"""Run recorded shape/texture stages on prepared four-view scene directories.

Each scene directory contains multiview.png, or --view-manifest supplies 2–3
named PNGs. The controller starts an
isolated CUDA worker per stage, checks GPU availability, and preserves failures.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(path)


def git_state(repository):
    if not repository:
        return {"commit": None, "changes": None}
    try:
        command = ["git", "-C", repository]
        return {
            "commit": subprocess.check_output(
                command + ["rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip(),
            "changes": subprocess.check_output(
                command + ["status", "--short"], text=True
            ).splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "changes": None}


def conditioning_input(args, directory):
    """Validate every named input before reusing or starting a GPU stage."""
    path = getattr(args, "view_manifest", None)
    if path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from agile3d.view_inputs import load_view_manifest

        load_view_manifest(path)
        return path
    return directory / "multiview.png"


def run_stage(args, config, directory):
    mesh_config = config["mesh"]
    if mesh_config.get("repository"):
        sys.path.insert(0, str(Path(mesh_config["repository"]).expanduser().resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agile3d.hunyuan_worker import validate_model_selection
    validate_model_selection(mesh_config)
    image_path = conditioning_input(args, directory)
    record_path = directory / f"{args.stage}-metadata.json"
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old.get("status") == "completed":
            raise ValueError(
                "Completed stage exists; use --resume or a new scene directory"
            )
        number = len(list(directory.glob(f"{args.stage}-metadata.failed-*.json"))) + 1
        record_path.replace(directory / f"{args.stage}-metadata.failed-{number}.json")
    record = {
        "stage": args.stage,
        "case": directory.name,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "input": str(image_path),
        "input_sha256": digest(image_path),
        "config_sha256": digest(args.config),
        "mesh_config": mesh_config,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "previous_failures": [
            p.name
            for p in sorted(directory.glob(f"{args.stage}-metadata.failed-*.json"))
        ],
    }
    save_json(record_path, record)
    started = time.monotonic()
    try:
        import torch
        import trimesh
        from PIL import Image
        from hy3dgen.rembg import BackgroundRemover

        record["versions"] = {
            name: importlib.metadata.version(name)
            for name in ["torch", "diffusers", "transformers", "trimesh", "rembg"]
        }
        record["hunyuan_source"] = git_state(mesh_config.get("repository"))
        record["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        remover = BackgroundRemover()
        if args.view_manifest:
            from agile3d.view_inputs import load_view_manifest

            inputs = load_view_manifest(image_path, record["input_sha256"])
            views = {name: remover(image) for name, image in inputs.items()}
            record["input_format"] = "named_view_subset"
            record["view_manifest"] = json.loads(image_path.read_text())
        else:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            record["input_size"] = [width, height]
            boxes = [
                (0, 0, width // 2, height // 2),
                (width // 2, 0, width, height // 2),
                (0, height // 2, width // 2, height),
                (width // 2, height // 2, width, height),
            ]
            views = {
                name: remover(image.crop(box))
                for name, box in zip(["front", "back", "left", "right"], boxes)
            }
            record["input_format"] = "four_view_grid"
        record["input_view_names"] = list(views)
        save_json(record_path, record)
        for name, view in views.items():
            view.save(directory / f"{name}.png")
        del remover
        gc.collect()
        if args.stage == "shape":
            from hy3dgen.shapegen import (
                Hunyuan3DDiTFlowMatchingPipeline,
                FloaterRemover,
                DegenerateFaceRemover,
            )

            seed = mesh_config.get("seed", 1234)
            steps = mesh_config.get("steps", 30)
            resolution = mesh_config.get("octree_resolution", 128)
            record.update(seed=seed, steps=steps, octree_resolution=resolution)
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                mesh_config["shape_model"],
                subfolder=mesh_config["subfolder"],
                use_safetensors=True,
                device="cuda",
            )
            mesh = pipeline(
                image=views,
                num_inference_steps=steps,
                octree_resolution=resolution,
                generator=torch.Generator().manual_seed(seed),
            )[0]
            mesh = DegenerateFaceRemover()(FloaterRemover()(mesh))
            output = directory / "shape.glb"
        else:
            from agile3d.hunyuan_worker import load_texture_pipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            shape_record = json.loads((directory / "shape-metadata.json").read_text())
            if (
                shape_record.get("status") != "completed"
                or shape_record.get("input_sha256") != record["input_sha256"]
            ):
                raise ValueError(
                    "Texture requires a successful shape from these exact views"
                )
            mesh = trimesh.load(directory / "shape.glb", force="mesh")
            pipeline = load_texture_pipeline(
                Hunyuan3DPaintPipeline,
                mesh_config["texture_model"],
                device="cuda",
                low_vram=mesh_config.get("low_vram", True),
            )
            record["checkpoint_compatibility"] = getattr(
                pipeline, "_agile_weight_compatibility", None
            )
            mesh = pipeline(mesh, image=list(views.values()))
            record["texture_sampling"] = {
                "delight_seed": 42,
                "delight_requested_steps": 50,
                "paint_seed": 0,
                "paint_requested_steps": 30,
                "paint_turbo": True,
                "seed_source": "upstream fixed defaults",
                "delight_actual_steps": len(
                    pipeline.models["delight_model"].pipeline.scheduler.timesteps
                ),
                "paint_actual_steps": len(
                    pipeline.models["multiview_model"].pipeline.scheduler.timesteps
                ),
            }
            output = directory / "mesh.glb"
        mesh.export(output)
        record.update(
            status="completed",
            output=str(output),
            output_sha256=digest(output),
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        )
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(record_path, record)
        status = record["status"]
        if args.stage == "shape" and status == "completed":
            status = "shape_completed_waiting_texture"
        summary = {
            "method": "codex-multiview+hunyuan3d-2mv",
            "case": directory.name,
            "status": status,
        }
        for phase in ["shape", "texture"]:
            path = directory / f"{phase}-metadata.json"
            if path.exists():
                summary[phase] = json.loads(path.read_text())
        summary["attempt_history"] = [
            json.loads(p.read_text())
            for p in sorted(directory.glob("*-metadata.failed-*.json"))
        ]
        if args.checkpoint_manifest:
            summary["weight_manifest"] = json.loads(
                args.checkpoint_manifest.read_text()
            )
        save_json(directory / "generation.json", summary)


def record_interruption(directory, phase, error):
    """A terminated child cannot run its finally block; close its run record."""
    path = directory / f"{phase}-metadata.json"
    if not path.exists():
        return
    record = json.loads(path.read_text())
    if record.get("status") != "running":
        return
    finished = datetime.now(timezone.utc)
    record.update(
        status="failed",
        error=f"{type(error).__name__}: {error}",
        finished_utc=finished.isoformat(),
        terminated_by_controller=True,
    )
    if record.get("started_utc"):
        record["elapsed_seconds"] = (
            finished - datetime.fromisoformat(record["started_utc"])
        ).total_seconds()
    save_json(path, record)
    summary_path = directory / "generation.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else {"case": directory.name}
    )
    summary.update(status="failed")
    summary[phase] = record
    save_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_dirs", nargs="+", type=Path, help="Directories containing multiview.png"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="AGILE pipeline JSON configuration"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="Physical GPU index; checked before every worker",
    )
    parser.add_argument("--stage", choices=["shape", "texture", "both"], default="both")
    parser.add_argument(
        "--u2net-home",
        type=Path,
        required=True,
        help="Directory containing the existing u2net.onnx",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        help="Pinned model file manifest to include in provenance",
    )
    parser.add_argument(
        "--view-manifest",
        type=Path,
        help="Portable named-view bundle; use with exactly one scene",
    )
    parser.add_argument("--max-existing-gpu-mib", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    args.u2net_home = args.u2net_home.expanduser().resolve()
    args.scene_dirs = [p.expanduser().resolve() for p in args.scene_dirs]
    if args.view_manifest:
        if len(args.scene_dirs) != 1:
            parser.error("--view-manifest requires exactly one scene directory")
        args.view_manifest = args.view_manifest.expanduser().resolve()
    if args.checkpoint_manifest:
        args.checkpoint_manifest = args.checkpoint_manifest.expanduser().resolve()
    config = json.loads(args.config.read_text())
    if config["mesh"].get("device", "cuda") not in ("cuda", "cuda:0"):
        parser.error("Use device=cuda and select the physical card with --gpu")
    if not (args.u2net_home / "u2net.onnx").is_file():
        parser.error(
            "Existing u2net.onnx is required; this runner performs no downloads"
        )
    if args.worker:
        if len(args.scene_dirs) != 1 or args.stage == "both":
            parser.error("Internal worker needs one scene and one phase")
        run_stage(args, config, args.scene_dirs[0])
        return
    stages = ["shape", "texture"] if args.stage == "both" else [args.stage]
    for directory in args.scene_dirs:
        input_path = conditioning_input(args, directory)
        for phase in stages:
            metadata = directory / f"{phase}-metadata.json"
            if args.resume and metadata.exists():
                old = json.loads(metadata.read_text())
                if (
                    old.get("status") == "completed"
                    and old.get("input_sha256") == digest(input_path)
                    and old.get("config_sha256") == digest(args.config)
                ):
                    if Path(old.get("output", "")).is_file() and digest(
                        Path(old["output"])
                    ) == old.get("output_sha256"):
                        print(
                            json.dumps(
                                {
                                    "case": directory.name,
                                    "phase": phase,
                                    "status": "reused",
                                }
                            ),
                            flush=True,
                        )
                        continue
            used = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
            if used > args.max_existing_gpu_mib:
                raise RuntimeError(
                    f"GPU {args.gpu} is busy ({used} MiB); no worker started"
                )
            python = config["mesh"].get("python") or sys.executable
            command = [
                python,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                str(args.config),
                "--gpu",
                str(args.gpu),
                "--stage",
                phase,
                "--u2net-home",
                str(args.u2net_home),
                str(directory),
            ]
            if args.checkpoint_manifest:
                command += ["--checkpoint-manifest", str(args.checkpoint_manifest)]
            if args.view_manifest:
                command += ["--view-manifest", str(args.view_manifest)]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(args.gpu),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                PYTHONDONTWRITEBYTECODE="1",
                U2NET_HOME=str(args.u2net_home),
                OMP_NUM_THREADS="8",
            )
            print(
                json.dumps(
                    {
                        "case": directory.name,
                        "phase": phase,
                        "status": "running",
                        "gpu": args.gpu,
                    }
                ),
                flush=True,
            )
            with (directory / "mesh-worker.log").open("ab") as stream:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    code = process.wait(
                        timeout=config["mesh"].get("timeout_seconds", 1800)
                    )
                except BaseException as exc:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    record_interruption(directory, phase, exc)
                    raise
            if code:
                raise RuntimeError(
                    f"{directory.name}/{phase} failed; see mesh-worker.log and metadata"
                )
            print(
                json.dumps(
                    {"case": directory.name, "phase": phase, "status": "completed"}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
