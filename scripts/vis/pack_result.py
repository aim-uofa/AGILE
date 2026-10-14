"""Create a portable result bundle from a native AGILE export."""
import argparse
from optimization.data.result_bundle import pack_result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference-rgb-dir')
    p.add_argument('--label')
    p.add_argument('--source-fps', type=float)
    p.add_argument('--source-fps-provenance')
    p.add_argument('--start', type=int, default=0, help='Array index, not original frame ID')
    p.add_argument('--end', type=int, default=-1)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--handedness', choices=('left','right','unknown'))
    p.add_argument('--asset-manifest', help='Verified final Blender/postprocessed asset package')
    p.add_argument('--canonical-input', help='Original pre-decimation optimization input for asset verification')
    p.add_argument('--native-object-poses', help='Matching native object_poses.npz to recover scale from old v1 bundles')
    a = vars(p.parse_args()); a['source'] = a.pop('data_dir')
    m = pack_result(**a)
    print(f"Packed {len(m['frames'])} frames into {a['output']}")

if __name__ == '__main__':
    main()
