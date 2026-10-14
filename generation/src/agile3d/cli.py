"""Small CLI using the same state machine as the agent tools."""

import argparse
import json
import sys
from pathlib import Path

from .hoi import HOIConfig, prepare_hoi, start_hoi_stage
from .models import PipelineConfig, Review
from .pipeline import (
    create_job,
    export_asset,
    queue_candidate,
    submit_review,
    submit_geometry_review,
    provide_texture,
)
from .store import Store
from .agent_loop import review_with_agent, run_agent_loop
from .generation import (
    create_from_frames,
    provide_multiview,
    provide_mesh,
    submit_stage_review,
)
from .stage_reviews import MultiviewReview, TextureReview
from .view_selection import ViewSubsetReview, submit_view_selection, recover_multiview
from .final_output import AssetSelection, TERMINAL, finalize_asset, prepare_delivery, provide_final_texture
from .doctor import inspect_file, format_report, PreflightError
from .workflow import generate, resume, format_status, WorkflowError


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    advanced_help = "--help-advanced" in argv
    if advanced_help:
        argv = [arg for arg in argv if arg != "--help-advanced"] + ["--help"]
    parser = argparse.ArgumentParser(
        prog="agile",
        description="Generate and review an object asset, or continue a saved job.",
        epilog="Use agile --help-advanced to list manual import, review, recovery and handoff commands. Existing advanced commands return JSON.",
    )
    parser.add_argument("--root", type=Path, default=Path("runs"))
    sub = parser.add_subparsers(dest="command", required=True, title="commands")
    doctor = sub.add_parser("doctor", help="Check all configured stages offline before any provider call")
    doctor.add_argument("--config", type=Path, required=True)
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable checks")
    generate_command = sub.add_parser("generate", help="Select references and run the configured generation route")
    generate_command.add_argument("--config", type=Path, required=True)
    generate_command.add_argument("--description", required=True)
    generate_command.add_argument("--frame", type=Path, action="append", required=True)
    generate_command.add_argument("--max-selected", type=int, default=4)
    generate_command.add_argument("--json", action="store_true", help="Emit the complete job state")
    resume_command = sub.add_parser("resume", help="Safely continue a saved job using its frozen configuration")
    resume_command.add_argument("job")
    resume_command.add_argument("--json", action="store_true", help="Emit the complete job state")
    status = sub.add_parser("status", help="Inspect a job (JSON by default for existing clients)")
    status.add_argument("job")
    output = status.add_mutually_exclusive_group()
    output.add_argument("--human", action="store_true", help="Show concise state, output and next action")
    output.add_argument("--json", action="store_true", help="Emit the complete job state (default)")
    create = sub.add_parser("create", help="Freeze references and a generation profile")
    create.add_argument("--reference", type=Path, action="append", required=True)
    create.add_argument("--description", required=True)
    create.add_argument("--config", type=Path, required=True)
    select = sub.add_parser(
        "create-from-frames",
        help="Configured VLM selects 1-4 references from at most 16 frozen candidate frames",
    )
    select.add_argument("--frame", type=Path, action="append", required=True)
    select.add_argument("--max-selected", type=int, default=4)
    select.add_argument("--description", required=True)
    select.add_argument("--config", type=Path, required=True)
    views = sub.add_parser("apply-multiview")
    views.add_argument("job")
    views.add_argument("--image", type=Path, required=True)
    views.add_argument("--provenance", type=Path)
    lifted = sub.add_parser("apply-mesh")
    lifted.add_argument("job")
    lifted.add_argument("--mesh", type=Path, required=True)
    lifted.add_argument("--provenance", type=Path, required=True)
    stage = sub.add_parser("stage-review")
    stage.add_argument("job")
    stage.add_argument("--file", type=Path, required=True)
    sub.add_parser("fallback-views", help="Continue a previously exhausted multi-view job using its historical views").add_argument("job")
    subset = sub.add_parser("select-views", help="Submit a scored 2–3-view subset selection")
    subset.add_argument("job")
    subset.add_argument("--file", type=Path, required=True)
    candidate = sub.add_parser(
        "candidate", help="Generate one candidate and stop for agent review"
    )
    candidate.add_argument("job")
    candidate.add_argument("--image", type=Path)
    candidate.add_argument("--mesh", type=Path)
    candidate.add_argument("--resume", action="store_true")
    candidate.add_argument("--retry-image", action="store_true")
    candidate.add_argument("--background", action="store_true")
    for command in ("export", "export-accepted", "codex-review", "codex-loop", "agent-review", "agent-loop"):
        sub.add_parser(command).add_argument("job")
    sub.add_parser("inspect-final", help="Freeze all complete mesh/texture candidates for final comparison").add_argument("job")
    final = sub.add_parser("finalize", help="Select and export the best available textured asset even after failed gates")
    final.add_argument("job")
    final.add_argument("--file", type=Path, help="Optional saved structured comparison")
    completion = sub.add_parser("apply-final-texture", help="Complete a UV mesh after geometry retries ran out")
    completion.add_argument("job")
    completion.add_argument("--image", type=Path, required=True)
    completion.add_argument("--provenance", type=Path, required=True)
    texture = sub.add_parser(
        "apply-texture", help="Embed a generated atlas and finish the waiting candidate"
    )
    texture.add_argument("job")
    texture.add_argument("--image", type=Path, required=True)
    texture.add_argument("--provenance", type=Path)
    review = sub.add_parser("review")
    review.add_argument("job")
    review.add_argument("--file", type=Path, required=True)
    shape_review = sub.add_parser("geometry-review")
    shape_review.add_argument("job")
    shape_review.add_argument("--file", type=Path, required=True)
    hoi = sub.add_parser("prepare-hoi")
    hoi.add_argument("job")
    hoi.add_argument("--config", type=Path, required=True)
    hoi.add_argument("--split", choices=["train", "evaluation"], default="train")
    hoi.add_argument("--scene", required=True)
    run = sub.add_parser("hoi-stage")
    run.add_argument("job")
    run.add_argument("stage", choices=["preprocess", "train", "evaluate"])
    if not advanced_help:
        # Hide advanced command descriptions, not their parsers or compatibility
        # interfaces. argparse exposes no separate grouping API for subcommands.
        common = {"doctor", "generate", "resume", "status"}
        sub._choices_actions[:] = [action for action in sub._choices_actions if action.dest in common]
        sub.metavar = "{doctor,generate,resume,status,...}"
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = inspect_file(args.config)
        print(json.dumps(result, indent=2) if args.json else format_report(result))
        if result["status"] != "ready":
            raise SystemExit(2)
        return
    store = Store(args.root)
    try:
        if args.command == "generate":
            result = generate(store, args.frame, args.description,
                              json.loads(args.config.read_text()), args.max_selected)
        elif args.command == "resume":
            result = resume(store, args.job)
        elif args.command == "create":
            result = create_job(
                store,
                args.reference,
                args.description,
                PipelineConfig.model_validate_json(args.config.read_text()),
            )
        elif args.command == "create-from-frames":
            result = create_from_frames(
                store,
                args.frame,
                args.description,
                PipelineConfig.model_validate_json(args.config.read_text()),
                args.max_selected,
            )
        elif args.command == "apply-multiview":
            result = provide_multiview(store, args.job, args.image, args.provenance)
        elif args.command == "apply-mesh":
            result = provide_mesh(store, args.job, args.mesh, args.provenance)
        elif args.command == "fallback-views":
            result = recover_multiview(store, args.job)
        elif args.command == "select-views":
            result = submit_view_selection(store, args.job, ViewSubsetReview.model_validate_json(args.file.read_text()))
        elif args.command == "stage-review":
            schema = (
                MultiviewReview
                if store.load(args.job)["status"] == "awaiting_multiview_review"
                else TextureReview
            )
            result = submit_stage_review(
                store, args.job, schema.model_validate_json(args.file.read_text())
            )
        elif args.command == "candidate":
            result = queue_candidate(
                store,
                args.job,
                image=args.image,
                mesh=args.mesh,
                resume=args.resume,
                retry_image=args.retry_image,
                background=args.background,
            )
        elif args.command == "apply-texture":
            result = provide_texture(store, args.job, args.image, args.provenance)
        elif args.command in ("codex-review", "agent-review"):
            result = review_with_agent(store, args.job)
        elif args.command in ("codex-loop", "agent-loop"):
            result = run_agent_loop(store, args.job)
        elif args.command == "review":
            result = submit_review(
                store, args.job, Review.model_validate_json(args.file.read_text())
            )
        elif args.command == "geometry-review":
            result = submit_geometry_review(
                store, args.job, Review.model_validate_json(args.file.read_text())
            )
        elif args.command == "export":
            result = finalize_asset(store, args.job)
        elif args.command == "export-accepted":
            result = export_asset(store, args.job)
        elif args.command == "inspect-final":
            result = prepare_delivery(store, args.job)
        elif args.command == "finalize":
            result = finalize_asset(store, args.job, selection=AssetSelection.model_validate_json(args.file.read_text()) if args.file else None)
        elif args.command == "apply-final-texture":
            result = provide_final_texture(store, args.job, args.image, args.provenance)
        elif args.command == "prepare-hoi":
            result = prepare_hoi(
                store,
                args.job,
                HOIConfig.model_validate_json(args.config.read_text()),
                args.split,
                args.scene,
            )
        elif args.command == "hoi-stage":
            result = start_hoi_stage(store, args.job, args.stage)
        else:
            result = store.load(args.job)
            path = store.directory(args.job) / "hoi/handoff.json"
            if path.is_file():
                result["hoi"] = json.loads(path.read_text())
        if args.command in ("review", "geometry-review", "stage-review", "codex-review", "agent-review") and result.get("status") in TERMINAL:
            finalize_asset(store, args.job)
            result = store.load(args.job)
        human = (args.command in ("generate", "resume") and not args.json) or (
            args.command == "status" and args.human
        )
        print(format_status(result) if human else json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if result.get("status") == "failed":
            raise SystemExit(1)
    except (PreflightError, WorkflowError) as exc:
        if getattr(args, "json", False):
            print(json.dumps(exc.report, indent=2))
            raise SystemExit(2)
        parser.exit(2, f"{exc}\n")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
