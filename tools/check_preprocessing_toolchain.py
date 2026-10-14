#!/usr/bin/env python3
"""Check public fork CLI contracts without importing models or loading weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def inspect(repository, spec):
    path = Path(repository) / spec['entrypoint']
    if not path.is_file():
        return {'tool': spec['name'], 'passed': False, 'missing_entrypoint': spec['entrypoint']}
    source = path.read_bytes()
    flags = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            flags.update(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
    missing = sorted(set(spec['required_flags']) - flags)
    revision = subprocess.run(['git', '-C', str(Path(repository).resolve()), 'rev-parse', '--show-toplevel'],
                              capture_output=True, text=True)
    commit = None
    if revision.returncode == 0 and Path(revision.stdout.strip()).resolve() == Path(repository).resolve():
        commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    return {'tool': spec['name'], 'passed': not missing, 'missing_flags': missing,
            'checkout_commit': commit, 'audited_commit': spec['audited_commit'],
            'entrypoint_matches_audited_source': hashlib.sha256(source).hexdigest() == spec['entrypoint_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wilor', 'moge', 'foundationpose'):
        parser.add_argument('--' + name + '-path', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'configs/preprocessing-toolchain.json').read_text())
    results = [inspect(getattr(args, spec['name'].lower() + '_path'), spec) for spec in manifest['tools']]
    print(json.dumps({'tools': results, 'scope': 'Static entrypoint/flag audit; not model inference'}, indent=2))
    raise SystemExit(not all(r['passed'] for r in results))


if __name__ == '__main__':
    main()
