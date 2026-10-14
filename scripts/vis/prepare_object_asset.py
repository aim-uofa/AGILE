"""Register the selected final surface against its pre-decimation canonical input."""
import argparse
import json
from pathlib import Path
from optimization.data.object_assets import prepare_object_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True, help='Final postprocessed GLB or OBJ; not the heavy generation intermediate')
    parser.add_argument('--canonical-input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', required=True, choices=('blender-final', 'postprocessed-final'))
    parser.add_argument('--provenance', type=Path, required=True, help='JSON recording selection/postprocessing evidence')
    parser.add_argument('--supporting-files', nargs='*', default=[])
    args = parser.parse_args()
    path = prepare_object_asset(args.mesh, args.canonical_input, args.output,
        stage=args.stage, provenance=json.loads(args.provenance.read_text()), supporting_files=args.supporting_files)
    print(path)


if __name__ == '__main__':
    main()
