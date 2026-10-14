"""A job-scoped HO3D bridge to the separately installed AGILE/HOI repository."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import StrictModel
from .pipeline import export_asset, verify_evidence
from .store import Store, atomic_json


class HOIConfig(StrictModel):
    repository: str
    python: str
    raw_data_root: str
    mano_root: str
    dino_model_path: str
    wilor_path: str
    foundationpose_path: str
    moge_path: str
    moge_env: str = "moge"
    wilor_env: str = "wilor"
    foundationpose_env: str = "foundationpose"
    hoi_env: str = "gsplat"
    max_frames: int = Field(default=1024, ge=2, le=2048)
    timeout_seconds: int = Field(default=86400, ge=60, le=86400)
    allow_best_available: bool = False


def prepare_hoi(
    store: Store,
    job_id: str,
    config: HOIConfig,
    split: Literal["train", "evaluation"],
    scene: str,
) -> dict:
    if split not in ("train", "evaluation") or not re.fullmatch(
        r"[A-Za-z0-9_-]+", scene
    ):
        raise ValueError("Invalid HO3D split or scene name")
    job = store.load(job_id)
    delivery = job.get("delivery", {})
    if delivery.get("status") == "delivered":
        verify_evidence(store, job)
        if delivery["quality_status"] != "accepted" and not config.allow_best_available:
            raise ValueError("Selected asset is best_available, not accepted; explicitly enable allow_best_available for this experiment")
        for artifact in delivery["files"]:
            store.resolve(job_id, artifact)
        export = delivery
        asset = {"source": "selected_delivery", "selected_id": delivery["selected_id"],
                 "quality_status": delivery["quality_status"],
                 "selection_evidence_digest": delivery["evidence_digest"]}
    else:
        export = export_asset(store, job_id)
        asset = {"source": "legacy_accepted_export", "quality_status": "accepted"}
    asset["mesh_sha256"] = next(f["sha256"] for f in export["files"] if Path(f["path"]).name == "textured_mesh.glb")
    with store.lock(job_id):
        directory = store.directory(job_id) / "hoi"
        if (directory / "handoff.json").exists():
            old = json.loads((directory / "handoff.json").read_text())
            if (
                HOIConfig.model_validate(old["config"]).model_dump() != config.model_dump()
                or old["scene"] != scene
                or old["split"] != split
                or old.get("asset") != asset
            ):
                raise ValueError("This job already has a different HOI handoff")
            return old
        repository = Path(config.repository).resolve()
        if not (repository / "preprocessing/preprocess_ho3d.py").is_file():
            raise ValueError("HOI repository is missing its HO3D preprocessor")
        # Upstream preprocessing constructs shell commands. Reject unsafe path
        # characters before those values can reach the legacy process runner.
        for key, value in config.model_dump().items():
            # This interpreter is launched as an argv element by run_process;
            # it is never interpolated into the legacy preprocessing shell.
            if key == "python":
                continue
            if isinstance(value, str) and not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
                raise ValueError(
                    f"HOI's legacy shell runner requires a simple path/name: {key}"
                )
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(directory)):
            raise ValueError("HOI job root must not contain spaces or shell characters")
        for key in (
            "repository",
            "python",
            "raw_data_root",
            "mano_root",
            "dino_model_path",
            "wilor_path",
            "moge_path",
            "foundationpose_path",
        ):
            if (
                not Path(getattr(config, key)).is_absolute()
                or not Path(getattr(config, key)).exists()
            ):
                raise ValueError(f"Missing absolute HOI resource: {key}")
        if (
            not Path(config.moge_path).is_absolute()
            or not (Path(config.moge_path) / "demo_moge.py").is_file()
        ):
            raise ValueError("Missing absolute standalone MoGe checkout: moge_path")
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"], text=True
            ).strip()
        )
        mesh_dir = directory / "meshes" / split / scene / "object"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for file in export["files"]:
            source = store.resolve(job_id, file)
            shutil.copyfile(source, mesh_dir / source.name)
        handoff = {
            "schema_version": 1,
            "job_id": job_id,
            "asset": asset,
            "config": config.model_dump(),
            "repository_revision": revision,
            "repository_dirty": dirty,
            "split": split,
            "scene": scene,
            "mesh_type": "object",
            "generated_mesh_root": str(directory / "meshes"),
            "processed_data_root": str(directory / "processed" / "HO3D"),
            "result_dir": str(directory / "results"),
            "stages": {name: "pending" for name in ("preprocess", "train", "evaluate")},
        }
        atomic_json(directory / "handoff.json", handoff)
        return handoff


def start_hoi_stage(
    store: Store, job_id: str, stage: Literal["preprocess", "train", "evaluate"]
) -> dict:
    order = ("preprocess", "train", "evaluate")
    if stage not in order:
        raise ValueError("Unknown HOI stage")
    with store.lock(job_id):
        path = store.directory(job_id) / "hoi" / "handoff.json"
        handoff = json.loads(path.read_text())
        if any(s == "running" for s in handoff["stages"].values()):
            raise ValueError(
                "An HOI stage is already running; inspect its process/log before recovery"
            )
        if handoff["stages"][stage] == "complete":
            return handoff
        if any(handoff["stages"][s] != "complete" for s in order[: order.index(stage)]):
            raise ValueError("Complete the preceding HOI stages first")
        handoff["stages"][stage] = "running"
        atomic_json(path, handoff)
        with path.with_name(f"{stage}-control.log").open("ab") as stream:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "agile3d.hoi", str(path), stage],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                handoff["process_id"] = process.pid
            except Exception:
                handoff["stages"][stage] = "failed"
                atomic_json(path, handoff)
                raise
        atomic_json(path, handoff)
        return handoff


def _execute(path: Path, stage: str):
    from .providers import run_process
    from .pipeline import _error_message

    handoff = json.loads(path.read_text())
    config = handoff["config"]
    error = None
    try:
        run_process(
            [
                config["python"],
                str(Path(__file__).with_name("hoi_worker.py")),
                str(path),
                stage,
            ],
            path.with_name(f"{stage}.log"),
            config["timeout_seconds"],
            cwd=config["repository"],
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    except Exception as exc:
        error = _error_message(exc)
    store = Store(path.parents[2])
    with store.lock(handoff["job_id"]):
        current = json.loads(path.read_text())
        current["stages"][stage] = "failed" if error else "complete"
        if error:
            current["error"] = error
        else:
            current.pop("error", None)
        current.pop("process_id", None)
        atomic_json(path, current)


if __name__ == "__main__":
    _execute(Path(sys.argv[1]), sys.argv[2])
