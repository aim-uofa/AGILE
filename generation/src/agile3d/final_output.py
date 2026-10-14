"""Deliver the best complete mesh/texture pair independently of quality acceptance.

Comparison uses original observations and actual mesh renders, never generated
conditioning images as ground truth. Historical reviews and budgets are untouched.
"""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from filelock import FileLock
from PIL import Image, ImageDraw
from pydantic import Field

from .mesh import export_obj, inspect_mesh, render_views
from .models import PipelineConfig, StrictModel
from .store import atomic_json, now


TERMINAL = {"accepted", "exhausted", "rejected", "failed"}


class AssetAssessment(StrictModel):
    candidate_id: str
    geometry_fidelity: float = Field(ge=0, le=10)
    texture_fidelity: float = Field(ge=0, le=10)
    surface_usability: float = Field(ge=0, le=10)
    remaining_issues: str = Field(min_length=1, max_length=4000)

    def score(self):
        return round(
            0.60 * self.geometry_fidelity
            + 0.35 * self.texture_fidelity
            + 0.05 * self.surface_usability,
            6,
        )


class AssetSelection(StrictModel):
    evidence_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessments: list[AssetAssessment] = Field(min_length=1, max_length=12)
    selected_id: str
    reasoning: str = Field(min_length=1, max_length=6000)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _snapshot(job):
    # Delivery records must not invalidate generation evidence or feed themselves
    # into a different comparison on every resume.
    return {
        k: job.get(k)
        for k in (
            "references",
            "reference_selection",
            "config",
            "attempts",
            "status",
            "accepted_attempt",
            "exhausted_stage",
            "delivery_texture",
        )
    }


def candidate_pool(store, job):
    """Keep textures paired with their own GLB's embedded UV/material bindings."""
    candidates, excluded, seen = [], [], set()
    config = PipelineConfig.model_validate(job["config"])
    accepted_hashes = {
        a["artifacts"]["mesh"]["sha256"]
        for a in job["attempts"]
        if a.get("number") == job.get("accepted_attempt")
        and (a.get("review") or {}).get("decision") == "accept"
        and "mesh" in a["artifacts"]
    }
    for ref in job["references"]:
        store.resolve(job["id"], ref)
    if job.get("reference_selection"):
        store.resolve(job["id"], job["reference_selection"])

    def add(attempt, candidate_id, artifacts, mesh_key="mesh", texture_review=None):
        mesh_ref = artifacts.get(mesh_key)
        if mesh_ref is None:
            return
        source = store.resolve(job["id"], mesh_ref)
        if mesh_ref["sha256"] in seen:
            return
        # Codex's neutral UV scaffold is not a generated texture. A job requesting
        # image-generated texture cannot silently deliver this placeholder.
        if (
            config.mesh.backend == "codex"
            and config.texture.provider in ("provided", "image_api")
            and (mesh_key != "mesh" or not artifacts.get("texture_atlas"))
        ):
            excluded.append(
                {"id": candidate_id, "reason": "Generated texture still required"}
            )
            return
        try:
            quality = inspect_mesh(
                source,
                config.quality.model_copy(
                    update={"max_faces": 2_000_000, "require_texture": True}
                ),
            )
        except (ValueError, TypeError, IndexError) as exc:
            excluded.append({"id": candidate_id, "reason": str(exc)})
            return
        if (
            not quality.get("textured")
            or "bounds" not in quality
            or not any(hi > lo for lo, hi in zip(*quality["bounds"]))
        ):
            excluded.append(
                {"id": candidate_id, "reason": "No usable textured geometry"}
            )
            return
        if quality["faces"] > config.quality.max_faces:
            quality["passed"] = False
            quality["reasons"].append(
                "Face count exceeds the configured acceptance limit"
            )
        a_art = attempt["artifacts"]
        pair = {"mesh": mesh_ref}
        if mesh_key != "source_mesh":
            for key in ("uv_layout", "surface_record", "prepared_mesh"):
                if key in a_art:
                    pair[key] = a_art[key]
            if mesh_key == "prepared_mesh" and "baked_albedo" in a_art:
                pair["texture_atlas"] = a_art["baked_albedo"]
        if mesh_key == "mesh":
            for key in ("texture_atlas", "texture_provenance", "mesh_views"):
                if key in artifacts:
                    pair[key] = artifacts[key]
        for ref in pair.values():
            store.resolve(job["id"], ref)
        is_current = mesh_ref == a_art.get("mesh")
        geometry_review = attempt.get("geometry_review")
        if not geometry_review and not a_art.get("mesh"):
            geometry_review = attempt.get("review")
        candidates.append(
            {
                "id": candidate_id,
                "attempt": attempt["number"],
                "artifacts": pair,
                "quality": quality,
                "geometry_review": geometry_review,
                "texture_review": texture_review,
                "final_review": attempt.get("review") if is_current else None,
                "accepted": mesh_ref["sha256"] in accepted_hashes,
                "white_views": a_art.get("untextured_views")
                if mesh_key != "source_mesh"
                else None,
            }
        )
        seen.add(mesh_ref["sha256"])

    for attempt in job["attempts"]:
        prefix = f"a{attempt['number']:02d}"
        for trial in attempt.get("texture_trials", []):
            add(
                attempt,
                f"{prefix}-t{trial['number']:02d}",
                trial.get("artifacts", {}),
                texture_review=trial.get("review"),
            )
        add(
            attempt,
            f"{prefix}-mesh",
            attempt["artifacts"],
            texture_review=attempt.get("texture_review"),
        )
        add(attempt, f"{prefix}-prepared", attempt["artifacts"], "prepared_mesh")
        add(attempt, f"{prefix}-source", attempt["artifacts"], "source_mesh")
    rescue = job.get("delivery_texture")
    if rescue and rescue.get("artifacts"):
        attempt = next(a for a in job["attempts"] if a["number"] == rescue["attempt"])
        add(attempt, "completion-texture", rescue["artifacts"])
    return candidates, excluded


def _render_candidate(store, job, candidate, directory):
    art = candidate["artifacts"]
    raw = directory / f"{candidate['id']}-views.png"
    config = PipelineConfig.model_validate(job["config"])
    if (
        art.get("mesh_views")
        and candidate["quality"]["faces"] <= config.quality.max_faces
    ):
        raw = store.resolve(job["id"], art["mesh_views"])
    else:
        render_views(
            store.resolve(job["id"], art["mesh"]),
            raw,
            up_axis="Z"
            if config.mesh.backend == "codex"
            and (config.texture.provider in ("provided", "image_api") or config.surface.enabled)
            else "Y",
        )
    # Label frozen, real renders. This is evidence composition, not image synthesis.
    paths = [raw]
    if candidate.get("white_views"):
        paths.append(store.resolve(job["id"], candidate["white_views"]))
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert("RGB"))
    sheet = Image.new(
        "RGB",
        (
            max(i.width for i in images),
            sum(i.height for i in images) + 36 * len(images),
        ),
        "white",
    )
    draw, y = ImageDraw.Draw(sheet), 0
    for index, im in enumerate(images):
        draw.text(
            (12, y + 8),
            f"{candidate['id']} | {'ACTUAL TEXTURED MESH' if index == 0 else 'SAME GEOMETRY - WHITE'}",
            fill="black",
        )
        sheet.paste(im, (0, y + 36))
        y += im.height + 36
    output = directory / f"{candidate['id']}-comparison.png"
    sheet.save(output)
    return store.artifact(job["id"], output)


def prepare_delivery(store, job_id):
    """Freeze a comparison pool. Safe to inspect or resume without new API calls."""
    with store.lock(job_id):
        job = store.load(job_id)
        if job["status"] not in TERMINAL:
            raise ValueError("Finish the bounded generation loop before final delivery")
        snapshot = _digest(_snapshot(job))
        existing = job.get("delivery_request")
        if existing and existing["snapshot_digest"] == snapshot:
            request = json.loads(
                store.resolve(job_id, existing["artifact"]).read_text()
            )
            _verify_request(store, job, request)
            return request
        candidates, excluded = candidate_pool(store, job)
        directory = store.directory(job_id) / f"delivery-{snapshot[:16]}"
        directory.mkdir(exist_ok=True)
        for candidate in candidates:
            candidate["comparison"] = _render_candidate(
                store, job, candidate, directory
            )
        request = {
            "schema_version": 1,
            "snapshot_digest": snapshot,
            "references": job["references"],
            "candidates": candidates,
            "excluded": excluded,
            "policy": {
                "geometry_fidelity": 0.60,
                "texture_fidelity": 0.35,
                "surface_usability": 0.05,
            },
            "scope": "Best available complete pair among this job's candidates, relative to supplied original video frames; not calibrated video or GT evaluation.",
        }
        request["evidence_digest"] = _digest(request)
        path = directory / "request.json"
        atomic_json(path, request)
        job["delivery_request"] = {
            "snapshot_digest": snapshot,
            "artifact": store.artifact(job_id, path),
        }
        store.save(job)
        return request


def _verify_request(store, job, request):
    if request["snapshot_digest"] != _digest(_snapshot(job)):
        raise ValueError("Final selection refers to stale candidate evidence")
    if request["evidence_digest"] != _digest(
        {k: v for k, v in request.items() if k != "evidence_digest"}
    ):
        raise ValueError("Final selection request digest changed")
    for ref in request["references"]:
        store.resolve(job["id"], ref)
    for c in request["candidates"]:
        for ref in [c["comparison"], *c["artifacts"].values()]:
            store.resolve(job["id"], ref)
        if c.get("white_views"):
            store.resolve(job["id"], c["white_views"])


def validate_selection(selection, candidates, evidence_digest):
    ids = [c["id"] for c in candidates]
    assessed = [a.candidate_id for a in selection.assessments]
    if selection.evidence_digest != evidence_digest:
        raise ValueError("Selection refers to stale evidence")
    if len(assessed) != len(set(assessed)) or set(assessed) != set(ids):
        raise ValueError("Assess every candidate exactly once")
    scores = {a.candidate_id: a.score() for a in selection.assessments}
    if (
        selection.selected_id not in scores
        or scores[selection.selected_id] < max(scores.values()) - 1e-6
    ):
        raise ValueError(
            "Selected candidate must have the best combined reference fidelity"
        )


def _prior_score(candidate):
    final = candidate.get("final_review") or {}
    geometry = candidate.get("geometry_review") or {}
    texture = candidate.get("texture_review") or {}
    return (
        0.60 * geometry.get("score", final.get("score", 0.0))
        + 0.35 * texture.get("score_overall", final.get("score", 0.0))
        + 0.05 * (10.0 if candidate["quality"]["passed"] else 0.0)
    )


def _choose(store, job, request, directory):
    """Bounded tournament, with one durable external call per immutable batch."""
    from .agent_backend import get_agent_backend

    pool = request["candidates"]
    if len(pool) == 1:
        return {
            "selected_id": pool[0]["id"],
            "method": "only_usable_textured_candidate",
            "rounds": [],
        }
    references = [store.resolve(job["id"], ref) for ref in request["references"]]
    agent_config = PipelineConfig.model_validate(job["config"]).agent
    backend = get_agent_backend(agent_config)
    width = min(12, 16 - len(references))
    rounds, remaining, errors = [], list(pool), []
    round_number = 0
    while len(remaining) > 1:
        winners = []
        for offset in range(0, len(remaining), width):
            group = remaining[offset : offset + width]
            if len(group) == 1:
                winners.extend(group)
                continue
            number = len(rounds) + 1
            batch = directory / f"comparison-{number:02d}"
            batch.mkdir(exist_ok=True)
            evidence = _digest(
                {
                    "pool": request["evidence_digest"],
                    "round": round_number,
                    "ids": [c["id"] for c in group],
                }
            )
            claim_path, response_path = batch / "claim.json", batch / "response.json"
            try:
                if response_path.exists():
                    selection = AssetSelection.model_validate_json(
                        response_path.read_text()
                    )
                else:
                    if claim_path.exists():
                        raise ValueError(
                            "Comparison already started; no duplicate external call"
                        )
                    atomic_json(
                        claim_path,
                        {
                            "status": "started",
                            "evidence_digest": evidence,
                            "started_at": now(),
                        },
                    )
                    selection, metadata = backend.select_final_asset(
                        references,
                        [store.resolve(job["id"], c["comparison"]) for c in group],
                        [c["id"] for c in group],
                        evidence,
                        batch / agent_config.backend,
                    )
                    # Persist a received answer even when later validation fails.
                    atomic_json(response_path, selection.model_dump())
                    atomic_json(
                        claim_path,
                        {
                            "status": "response_received",
                            "metadata": metadata,
                            "evidence_digest": evidence,
                        },
                    )
                validate_selection(selection, group, evidence)
                result = {
                    "method": f"{agent_config.backend}_original_frame_comparison",
                    **selection.model_dump(),
                }
            except Exception as exc:
                # A ranking service failure must not throw away completed assets.
                # This fallback is explicitly NOT a fresh visual assessment.
                best = max(
                    group, key=lambda c: (_prior_score(c), c["accepted"], c["id"])
                )
                errors.append({"batch": number, "error_type": type(exc).__name__})
                result = {
                    "method": "prior_review_fallback",
                    "selected_id": best["id"],
                    "reasoning": "New comparison unavailable; selected using saved geometry/texture reviews. No quality acceptance was granted.",
                    "prior_scores": {c["id"]: _prior_score(c) for c in group},
                }
                atomic_json(batch / "fallback.json", result)
            rounds.append(result)
            winners.append(next(c for c in group if c["id"] == result["selected_id"]))
        remaining = winners
        round_number += 1
    return {
        "selected_id": remaining[0]["id"],
        "method": f"{agent_config.backend}_original_frame_comparison"
        if not errors
        else "comparison_with_prior_review_fallback",
        "rounds": rounds,
        "comparison_errors": errors,
    }


def _export(store, job, request, selection, directory):
    selected = next(
        c for c in request["candidates"] if c["id"] == selection["selected_id"]
    )
    artifacts = selected["artifacts"]
    target = directory / "asset"
    if target.exists():
        # Recover a local interruption after atomic rename but before job.save.
        manifest = json.loads((target / "files.json").read_text())
        from .store import digest

        for item in manifest:
            if (
                Path(item["name"]).name != item["name"]
                or digest(target / item["name"]) != item["sha256"]
            ):
                raise ValueError("Previously exported file changed")
        prior = json.loads((target / "provenance.json").read_text())
        if (
            prior["evidence_digest"] != request["evidence_digest"]
            or prior["selection"] != selection
        ):
            raise ValueError("Existing output belongs to another selection")
        return _delivery_record(store, job, request, selection, selected, target)
    temporary = directory / f".export-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        source = store.resolve(job["id"], artifacts["mesh"])
        shutil.copyfile(source, temporary / "textured_mesh.glb")
        export_obj(source, temporary)
        for key, name in (
            ("uv_layout", "uv-layout.png"),
            ("surface_record", "surface.json"),
            ("texture_provenance", "texture-provenance.json"),
        ):
            if key in artifacts:
                shutil.copyfile(
                    store.resolve(job["id"], artifacts[key]), temporary / name
                )
        shutil.copyfile(
            store.resolve(job["id"], selected["comparison"]),
            temporary / "comparison.png",
        )
        quality = inspect_mesh(
            temporary / "textured_mesh.glb",
            PipelineConfig.model_validate(job["config"]).quality.model_copy(
                update={"max_faces": 2_000_000, "require_texture": True}
            ),
        )
        if not quality.get("textured"):
            raise ValueError("Final output must contain UV-bound bitmap textures")
        provenance = {
            "job_id": job["id"],
            "created_at": now(),
            "delivery_status": "accepted" if selected["accepted"] else "best_available",
            "generation_status": job["status"],
            "accepted_attempt": job.get("accepted_attempt"),
            "selection": selection,
            "selected_candidate": selected,
            "evidence_digest": request["evidence_digest"],
            "scope": request["scope"],
            "references": request["references"],
            "reference_selection": job.get("reference_selection"),
            "config": job["config"],
            "quality_gates_overridden": False,
            "texture_pairing": "Original self-contained GLB; no cross-mesh or cross-UV substitution",
        }
        atomic_json(temporary / "provenance.json", provenance)
        atomic_json(temporary / "selection.json", selection)
        from .store import digest

        atomic_json(
            temporary / "files.json",
            [
                {"name": p.name, "sha256": digest(p)}
                for p in sorted(temporary.iterdir())
                if p.is_file()
            ],
        )
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _delivery_record(store, job, request, selection, selected, target)


def _delivery_record(store, job, request, selection, selected, target):
    return {
        "status": "delivered",
        "quality_status": "accepted" if selected["accepted"] else "best_available",
        "selected_id": selected["id"],
        "method": selection["method"],
        "evidence_digest": request["evidence_digest"],
        "snapshot_digest": request["snapshot_digest"],
        "directory": str(target),
        "files": [
            store.artifact(job["id"], p)
            for p in sorted(target.iterdir())
            if p.is_file()
        ],
    }


def finalize_asset(store, job_id, *, selection=None):
    """Always deliver a usable textured candidate; request completion if none exists.

    This function never changes generation acceptance or clears prior defects.
    Only the finite generation loop decides quality acceptance.
    """
    with FileLock(store.directory(job_id) / ".delivery.lock", timeout=0):
        request = prepare_delivery(store, job_id)
        job = store.load(job_id)
        existing = job.get("delivery")
        if (
            existing
            and existing.get("snapshot_digest") == request["snapshot_digest"]
            and existing.get("status") == "delivered"
        ):
            for ref in existing["files"]:
                store.resolve(job_id, ref)
            return existing
        directory = store.resolve(job_id, job["delivery_request"]["artifact"]).parent
        if not request["candidates"]:
            result = _request_texture_completion(store, job, directory)
        else:
            if selection is None:
                result_selection = _choose(store, job, request, directory)
            else:
                validate_selection(
                    selection, request["candidates"], request["evidence_digest"]
                )
                result_selection = {
                    "method": "submitted_original_frame_comparison",
                    **selection.model_dump(),
                }
            with store.lock(job_id):
                job = store.load(job_id)
                _verify_request(store, job, request)
                result = _export(store, job, request, result_selection, directory)
                job["delivery"] = result
                store.save(job)
        return result


def _request_texture_completion(store, job, directory):
    """Do not mistake a white/neutral Codex scaffold for a finished texture."""
    from .surface import mesh_uv_request

    eligible = [
        a
        for a in job["attempts"]
        if all(
            k in a["artifacts"]
            for k in ("prepared_mesh", "uv_layout", "surface_record")
        )
    ]
    if not eligible:
        result = {
            "status": "needs_geometry",
            "reason": "No usable textured mesh or prepared UV mesh exists; generation must resume before delivery.",
        }
    else:
        attempt = max(
            eligible,
            key=lambda a: (a.get("geometry_review") or a.get("review") or {}).get(
                "score", 0
            ),
        )
        art = attempt["artifacts"]
        prepared = store.resolve(job["id"], art["prepared_mesh"])
        guide = store.resolve(job["id"], art["uv_layout"])
        record = json.loads(store.resolve(job["id"], art["surface_record"]).read_text())
        request = mesh_uv_request(
            job["description"],
            record,
            prepared,
            guide,
            feedback="Complete the best available geometry with a generated texture. Geometry has NOT been certified; keep all existing geometry/UV coordinates unchanged.",
        )
        request.update(
            attempt=attempt["number"],
            original_references=job["references"],
            geometry_review=attempt.get("geometry_review") or attempt.get("review"),
        )
        path = directory / "texture-completion-request.json"
        atomic_json(path, request)
        result = {
            "status": "awaiting_generated_texture",
            "attempt": attempt["number"],
            "request": store.artifact(job["id"], path),
            "prepared_mesh": art["prepared_mesh"],
            "uv_layout": art["uv_layout"],
            "reason": "Use the client image-generation tool with original references and this exact UV guide, then provide_final_texture. Quality failures remain recorded.",
        }
    with store.lock(job["id"]):
        current = store.load(job["id"])
        current["delivery"] = result
        store.save(current)
    return result


def provide_final_texture(store, job_id, image: Path, provenance: Path):
    """Complete an otherwise untextured best geometry without accepting its gate."""
    from .surface import apply_uv_texture, read_uv_texture
    from .store import digest

    with FileLock(store.directory(job_id) / ".delivery.lock", timeout=0):
        with store.lock(job_id):
            job = store.load(job_id)
            pending = job.get("delivery", {})
            if pending.get("status") != "awaiting_generated_texture":
                raise ValueError("Final delivery is not awaiting a generated texture")
            original_request = json.loads(
                store.resolve(job_id, job["delivery_request"]["artifact"]).read_text()
            )
            _verify_request(store, job, original_request)
            request = json.loads(store.resolve(job_id, pending["request"]).read_text())
            metadata = json.loads(provenance.read_text())
            for key in ("prepared_mesh_sha256", "uv_layout_sha256"):
                if metadata.get(key) != request[key]:
                    raise ValueError(
                        "Final texture provenance refers to different geometry/UV"
                    )
            if metadata.get("image_sha256") != digest(image):
                raise ValueError("Final texture image checksum mismatch")
            read_uv_texture(image)
            directory = store.resolve(job_id, pending["request"]).parent / "completion"
            directory.mkdir(exist_ok=True)
            shutil.copyfile(image, directory / "texture-atlas.png")
            shutil.copyfile(provenance, directory / "texture-provenance.json")
            apply_uv_texture(
                store.resolve(job_id, pending["prepared_mesh"]),
                directory / "texture-atlas.png",
                directory / "mesh.glb",
            )
            job["delivery_texture"] = {
                "attempt": pending["attempt"],
                "artifacts": {
                    k: store.artifact(job_id, directory / name)
                    for k, name in (
                        ("mesh", "mesh.glb"),
                        ("texture_atlas", "texture-atlas.png"),
                        ("texture_provenance", "texture-provenance.json"),
                    )
                },
            }
            store.save(job)
    return finalize_asset(store, job_id)
