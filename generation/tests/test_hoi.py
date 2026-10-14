import json
from pathlib import Path
import subprocess
import sys

from agile3d.hoi import HOIConfig, prepare_hoi, start_hoi_stage
from agile3d.pipeline import submit_review
from test_pipeline import candidate, review_for


def test_hoi_contract_and_stage_order(sample, tmp_path, monkeypatch):
    store, job_id, *_ = sample
    job = candidate(sample)
    submit_review(store, job_id, review_for(job))
    repo = tmp_path / "hoi-code"
    (repo / "preprocessing").mkdir(parents=True)
    (repo / "preprocessing/preprocess_ho3d.py").touch()
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda argv, **kwargs: "aebc923\n" if "rev-parse" in argv else "",
    )
    interpreter = tmp_path / "Python 环境" / "python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    (tmp_path / "demo_moge.py").touch()
    config = HOIConfig(
        repository=str(repo),
        python=str(interpreter),
        raw_data_root=str(tmp_path),
        mano_root=str(tmp_path),
        dino_model_path=str(tmp_path),
        wilor_path=str(tmp_path),
        moge_path=str(tmp_path),
        foundationpose_path=str(tmp_path),
    )
    handoff = prepare_hoi(store, job_id, config, "train", "ABF14")
    assert Path(
        handoff["generated_mesh_root"], "train/ABF14/object/textured_mesh.obj"
    ).exists()
    assert "HO3D" in handoff["processed_data_root"]
    assert prepare_hoi(store, job_id, config, "train", "ABF14") == handoff
    import pytest

    with pytest.raises(ValueError, match="preceding"):
        start_hoi_stage(store, job_id, "train")
    with pytest.raises(ValueError, match="different"):
        prepare_hoi(store, job_id, config, "train", "ABF12")
    from agile3d.hoi import _execute
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    def run_isolated(argv, log, timeout, *, cwd, env):
        assert argv[0] == str(interpreter)
        assert env["PYTHONNOUSERSITE"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
    monkeypatch.setattr("agile3d.providers.run_process", run_isolated)
    _execute(store.directory(job_id) / "hoi/handoff.json", "preprocess")
    completed = json.loads((store.directory(job_id) / "hoi/handoff.json").read_text())
    assert completed["stages"]["preprocess"] == "complete"


def test_standalone_moge_handoff_needs_no_megasam_installation(sample, tmp_path, monkeypatch):
    store, job_id, *_ = sample
    job = candidate(sample)
    submit_review(store, job_id, review_for(job))
    repo = tmp_path / "integrated"
    (repo / "preprocessing").mkdir(parents=True)
    (repo / "preprocessing/preprocess_ho3d.py").touch()
    moge = tmp_path / "MoGe"; moge.mkdir()
    (moge / "demo_moge.py").touch()
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "revision")
    config = HOIConfig(repository=str(repo), python=sys.executable, raw_data_root=str(tmp_path),
        mano_root=str(tmp_path), dino_model_path=str(tmp_path), wilor_path=str(tmp_path),
        foundationpose_path=str(tmp_path), moge_path=str(moge), moge_env="moge")
    handoff = prepare_hoi(store, job_id, config, "train", "SYNTHETIC")
    assert handoff["config"]["moge_path"] == str(moge)
    assert "megasam_path" not in handoff["config"]


def test_preprocessing_false_is_a_nonzero_worker_exit(tmp_path):
    from agile3d import hoi_worker

    fake = tmp_path / "fake"
    (fake / "preprocessing").mkdir(parents=True)
    (fake / "preprocessing/__init__.py").touch()
    (fake / "preprocessing/preprocess_ho3d.py").write_text("""
class HO3DConfig:
    def __init__(self, **kwargs):
        assert kwargs["moge_path"]
        assert kwargs["moge_env"] == "moge"
class HO3DPreprocessor:
    def __init__(self, config): pass
    def process_scene(self, *args): return False
""")
    request = tmp_path / "handoff.json"
    config = {
        k: str(tmp_path)
        for k in (
            "mano_root",
            "raw_data_root",
            "dino_model_path",
            "wilor_path",
            "moge_path",
            "foundationpose_path",
        )
    }
    config.update(
        repository=str(fake),
        wilor_env="wilor",
        moge_env="moge",
        foundationpose_env="foundationpose",
        hoi_env="gsplat",
        max_frames=2,
    )
    request.write_text(
        json.dumps(
            dict(
                config=config,
                scene="ABF14",
                split="train",
                processed_data_root=str(tmp_path / "HO3D"),
                generated_mesh_root=str(tmp_path / "meshes"),
            )
        )
    )
    result = subprocess.run(
        [sys.executable, hoi_worker.__file__, str(request), "preprocess"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "preprocessing reported failure" in result.stderr
