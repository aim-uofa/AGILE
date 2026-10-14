#!/usr/bin/env python3
"""Create AGILE's local MANO configuration/topology from licensed model data.

The model pickle is read in place and is never copied into the source tree.
Generated assets remain subject to the MANO terms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import trimesh
import yaml


def closed_wrist_faces(faces: np.ndarray) -> np.ndarray:
    """Prepare the hand rendering topology, including its wrist cap."""
    faces = np.asarray(faces)
    if faces.shape != (1538, 3) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Expected the standard 1538-face MANO topology")
    cap_path = Path(__file__).resolve().parents[1] / "configs/mano_wrist_cap.json"
    cap = np.asarray(json.loads(cap_path.read_text())["triangles"], dtype=np.int64)
    if cap.shape != (14, 3) or cap.min() < 0 or cap.max() >= 778:
        raise ValueError("Invalid MANO wrist-cap vertex indices")
    closed = np.concatenate([faces, cap], axis=0)
    edges = np.sort(closed[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise ValueError("Wrist cap does not close this MANO topology")
    return closed


def prepare_assets(model_dir: Path, output_dir: Path, finger_mask: Path | None = None) -> dict:
    model_dir = model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    model_file = model_dir / "MANO_RIGHT.pkl"
    if not model_file.is_file():
        raise FileNotFoundError(f"Licensed right-hand model not found: {model_file}")
    targets = [output_dir / "mano_config.yaml", output_dir / "mano_hand.obj"]
    if finger_mask is not None:
        targets.append(output_dir / "finger_vertices_mask.npz")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Existing MANO assets will not be overwritten: " + ", ".join(existing))
    mask_values = None
    if finger_mask is not None:
        if finger_mask.suffix == '.json':
            selection = json.loads(finger_mask.read_text())
            indices = selection['selected_indices']
            if (selection['vertex_count'] != 778 or len(set(indices)) != len(indices)
                    or any(type(i) is not int or not 0 <= i < 778 for i in indices)):
                raise ValueError('Invalid MANO finger vertex selection')
            mask_values = np.zeros(778, dtype=bool)
            mask_values[indices] = True
        else:
            with np.load(finger_mask.expanduser(), allow_pickle=False) as archive:
                mask_values = archive["finger_vertices_mask"]
        if mask_values.shape != (778,) or mask_values.dtype != np.bool_:
            raise ValueError("finger_vertices_mask must be a boolean array of shape (778,)")

    model = smplx.MANO(str(model_dir), is_rhand=True, use_pca=False, flat_hand_mean=False)
    vertices = model.v_template.detach().cpu().numpy()
    faces = np.asarray(model.faces)
    if vertices.shape != (778, 3) or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Expected the standard 778-vertex MANO right-hand topology")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= 778:
        raise ValueError("MANO template has invalid vertices or face indices")
    faces = closed_wrist_faces(faces)
    config = {"MANO": {"MODEL_PATH": str(model_dir), "IS_RHAND": True,
                       "USE_PCA": False, "FLAT_HAND_MEAN": False}}
    contents = [yaml.safe_dump(config, sort_keys=False),
                trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(file_type="obj")]
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in zip(targets, contents):
            with path.open("x") as handle:
                created.append(path)
                handle.write(content)
        if mask_values is not None:
            with targets[-1].open("xb") as handle:
                created.append(targets[-1])
                np.savez(handle, finger_vertices_mask=mask_values)
    except Exception:
        for path in created:
            path.unlink()
        raise
    return {"model_file": str(model_file), "output_dir": str(output_dir),
            "vertices": len(vertices), "faces": len(faces), "model_copied": False,
            "wrist_cap_faces": 14,
            "finger_mask_imported": mask_values is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing your licensed MANO_RIGHT.pkl")
    parser.add_argument("--output-dir", type=Path,
                        help="Local asset directory; defaults to --model-dir. Existing outputs are refused.")
    parser.add_argument("--finger-mask", type=Path,
                        help="Finger selection JSON or existing finger_vertices_mask.npz; no substitute is inferred.")
    args = parser.parse_args()
    print(json.dumps(prepare_assets(args.model_dir, args.output_dir or args.model_dir, args.finger_mask), indent=2))


if __name__ == "__main__":
    main()
