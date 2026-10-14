"""Salvage 2–3 coherent views across rejected grids without accepting those grids."""

import json
import shutil
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import Field

from .models import StrictModel, PipelineConfig
from .stage_reviews import Score
from .store import atomic_json, now
from .view_inputs import VIEW_NAMES


class ViewAssessment(StrictModel):
    candidate_id: str
    direction: Literal["front", "back", "left", "right", "unknown"]
    quality: Score
    usable: bool
    reason: str = Field(min_length=1, max_length=1200)


class ViewSubsetReview(StrictModel):
    evidence_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessments: list[ViewAssessment] = Field(min_length=4, max_length=40)
    selected_ids: list[str] = Field(min_length=2, max_length=3)
    consistency_score: Score
    reasoning: str = Field(min_length=1, max_length=6000)


def grid_crop_boxes(image):
    """Use clear central gutters when an imperfect grid is not split at 50%.

    Pixels remain unchanged. Only choose an unambiguous near-white gap close to
    the expected separator; otherwise retain equal quadrants for critic review.
    """
    w, h = image.size
    ink = np.any(np.asarray(image.convert("RGB")) < 243, axis=2)

    def separator(density):
        length = len(density)
        lo, hi = int(length * 0.4), int(length * 0.6)
        clear = density < 0.01
        if clear[length // 2]:
            return length // 2
        runs = []
        start = None
        for i in range(lo, hi + 1):
            if i < hi and clear[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start >= max(3, int(length * 0.002)):
                    runs.append((start + i) // 2)
                start = None
        return min(runs, key=lambda i: abs(i - length / 2)) if runs else length // 2

    x, y = separator(ink.mean(axis=0)), separator(ink.mean(axis=1))
    return [(0, 0, x, y), (x, 0, w, y), (0, y, x, h), (x, y, w, h)]


def selection_request(store, job, directory):
    """Freeze full-resolution crops and labeled evidence for the fallback critic."""
    from .generation import record, waiting

    a = job["attempts"][-1]
    folder = directory / "view-selection"
    folder.mkdir(exist_ok=True)
    candidates, sheets = [], []
    for trial in a["multiview_trials"]:
        original = store.resolve(job["id"], trial["image"])
        with Image.open(original) as im:
            boxes = grid_crop_boxes(im)
            sheet = Image.new("RGB", (768, 820), "white")
            draw = ImageDraw.Draw(sheet)
            for cell, box in enumerate(boxes):
                candidate_id = f"t{trial['number']:02d}-p{cell}"
                crop = im.crop(box)
                destination = folder / f"{candidate_id}.png"
                crop.save(destination)
                artifact = record(store, job, f"view_crop_{candidate_id}", destination)
                candidates.append(
                    {
                        "id": candidate_id,
                        "trial": trial["number"],
                        "cell": cell,
                        "crop_box": list(box),
                        "extraction": "central_white_gutter_or_equal_quadrants",
                        "image": artifact,
                        "source_grid": trial["image"],
                        "prior_review": trial.get("review"),
                    }
                )
                thumbnail = crop.convert("RGB")
                thumbnail.thumbnail((384, 384))
                x, y = (cell % 2) * 384, (cell // 2) * 410
                sheet.paste(thumbnail, (x + (384 - thumbnail.width) // 2, y))
                draw.text((x + 10, y + 387), candidate_id, fill="black")
            path = folder / f"trial-{trial['number']:02d}-candidates.png"
            sheet.save(path)
            sheets.append(record(store, job, f"view_sheet_{trial['number']:02d}", path))
    request = {
        "stage": "view_selection",
        "mode": "best_subset",
        "min_views": 2,
        "max_views": PipelineConfig.model_validate(
            job["config"]
        ).generation.fallback_max_views,
        "candidates": candidates,
        "sheets": sheets,
        "scope": "Rank individual views, then select a mutually consistent set of 2–3 distinct directions. Cross-trial selection is allowed. Scores rank imperfect inputs; the full-grid acceptance threshold does not apply. No new views, mirrored duplicates, blank padding or quality-pass claim.",
    }
    atomic_json(folder / "request.json", request)
    record(store, job, "view_selection_request", folder / "request.json")
    waiting(job, "awaiting_view_selection")


def recover_multiview(store, job_id):
    """Explicitly adopt the new fallback for an older exhausted job; retain its snapshot."""
    from .pipeline import verify_evidence
    from .generation import record

    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] != "exhausted" or job.get("exhausted_stage") != "multiview":
            raise ValueError(
                "Recovery requires a job exhausted at the multi-view stage"
            )
        verify_evidence(store, job)
        directory = (
            store.directory(job_id) / f"attempt-{job['attempts'][-1]['number']:02d}"
        )
        snapshot = directory / "before-view-fallback.json"
        if snapshot.exists():
            raise ValueError("Fallback recovery was already started")
        shutil.copyfile(store.directory(job_id) / "job.json", snapshot)
        record(store, job, "before_view_fallback", snapshot)
        policy = job["config"].setdefault("generation", {})
        policy["multiview_fallback"] = "best_subset"
        policy.setdefault("fallback_max_views", 3)
        job.pop("exhausted_stage", None)
        job["attempts"][-1]["view_fallback_recovery"] = {
            "started_at": now(),
            "previous_status": "exhausted",
            "policy": "best_subset",
        }
        selection_request(store, job, directory)
        store.save(job)
        return job


def submit_view_selection(store, job_id, review: ViewSubsetReview):
    from .generation import record, advance
    from .pipeline import run_candidate, verify_evidence

    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] != "awaiting_view_selection":
            raise ValueError("Candidate must be awaiting_view_selection")
        verify_evidence(store, job)
        a = job["attempts"][-1]
        if review.evidence_digest != a["evidence_digest"]:
            raise ValueError("View selection refers to stale evidence")
        request = json.loads(
            store.resolve(job_id, a["artifacts"]["view_selection_request"]).read_text()
        )
        pool = {c["id"]: c for c in request["candidates"]}
        assessments = {c.candidate_id: c for c in review.assessments}
        if len(assessments) != len(review.assessments) or set(assessments) != set(pool):
            raise ValueError("Assess every frozen candidate exactly once")
        chosen = review.selected_ids
        if (
            len(set(chosen)) != len(chosen)
            or not 2 <= len(chosen) <= request["max_views"]
            or not set(chosen) <= set(pool)
        ):
            raise ValueError(
                "Select 2–3 distinct frozen candidates within the configured maximum"
            )
        directions = [assessments[c].direction for c in chosen]
        if (
            len(set(directions)) != len(directions)
            or any(d not in VIEW_NAMES for d in directions)
            or any(not assessments[c].usable for c in chosen)
        ):
            raise ValueError(
                "Selected images need distinct known directions and recognizable usable content"
            )
        directory = (
            store.directory(job_id) / f"attempt-{a['number']:02d}" / "view-selection"
        )
        atomic_json(directory / "review.json", review.model_dump())
        record(store, job, "view_selection_review", directory / "review.json")
        selected = directory / "selected"
        selected.mkdir(exist_ok=True)
        entries, ordered = (
            [],
            sorted(chosen, key=lambda c: VIEW_NAMES.index(assessments[c].direction)),
        )
        # Display only. The model receives the named PNGs, never this preview.
        preview = Image.new("RGB", (512 * len(chosen), 512), "white")
        for i, candidate_id in enumerate(ordered):
            c, assessment = pool[candidate_id], assessments[candidate_id]
            path = selected / f"input-{assessment.direction}.png"
            shutil.copyfile(store.resolve(job_id, c["image"]), path)
            artifact = record(store, job, f"selected_view_{assessment.direction}", path)
            entries.append(
                {
                    "name": assessment.direction,
                    "file": path.name,
                    "sha256": artifact["sha256"],
                    "candidate_id": candidate_id,
                    "source_trial": c["trial"],
                    "source_grid_sha256": c["source_grid"]["sha256"],
                    "crop_box": c["crop_box"],
                    "quality": assessment.quality,
                }
            )
            with Image.open(path) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((512, 512))
                preview.paste(
                    thumb,
                    (i * 512 + (512 - thumb.width) // 2, (512 - thumb.height) // 2),
                )
        preview.save(selected / "preview.png")
        record(store, job, "multiview", selected / "preview.png")
        manifest = {
            "schema_version": 1,
            "kind": "named_view_subset",
            "views": entries,
            "selection_review_sha256": a["artifacts"]["view_selection_review"][
                "sha256"
            ],
        }
        atomic_json(selected / "manifest.json", manifest)
        record(store, job, "view_manifest", selected / "manifest.json")
        a["view_selection"] = {
            **review.model_dump(),
            "mode": "best_subset",
            "selected_count": len(entries),
            "grid_accepted": False,
        }
        token = advance(job)
        store.save(job)
    return run_candidate(store, job_id, token)
