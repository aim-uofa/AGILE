"""Download, verify and safely extract selected AGILE HO3D archives."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive, destination, entry):
    destination = Path(destination).resolve()
    if checksum(archive) != entry['sha256']:
        raise ValueError('Archive SHA-256 mismatch')
    expected = {m['path']: m for m in entry['members']}
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Archive member list does not match the release manifest')
        for member in members:
            target = destination/member.name
            if not member.isfile() or destination not in target.resolve().parents:
                raise ValueError('Unsafe archive member')
            if member.size != expected[member.name]['bytes']:
                raise ValueError('Archive member size mismatch')
            if target.exists() and checksum(target) != expected[member.name]['sha256']:
                raise FileExistsError(f'Existing file differs from the release: {target}')
        for member in members:
            target = destination/member.name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as src, target.open('xb') as dst:
                for chunk in iter(lambda: src.read(8 << 20), b''):
                    dst.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected[member.name]['sha256']:
                target.unlink()
                raise ValueError(f'Extracted member checksum mismatch: {member.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', help='Default: all 18 released sequences')
    parser.add_argument('--kind', choices=['preprocessed','generated','both','optimized'], default='both',
                        help='Default: prepared inputs and generated assets; optimized downloads final reconstructions')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='main', help='Use a commit hash to pin the data release')
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi().repo_info('Chrisss/AGILE', repo_type='dataset', revision=args.revision).sha
    manifest_name = 'ho3d/optimized/manifest.json' if args.kind == 'optimized' else 'ho3d/manifest.json'
    manifest = Path(hf_hub_download('Chrisss/AGILE',manifest_name,repo_type='dataset',revision=revision))
    release = json.loads(manifest.read_text())
    available = {e['scene'] for e in release['archives']}
    if args.scenes and not set(args.scenes) <= available:
        raise ValueError('Requested scene is absent from the published HO3D manifest')
    for entry in release['archives']:
        if (args.scenes and entry['scene'] not in args.scenes) or (args.kind!='both' and entry['kind']!=args.kind):
            continue
        archive=hf_hub_download('Chrisss/AGILE',entry['path'],repo_type='dataset',revision=revision)
        extract_archive(archive,args.output,entry)
        print(f"Verified and extracted {entry['path']}")


if __name__=='__main__':
    main()
