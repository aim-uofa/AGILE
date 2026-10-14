"""Check source-only HO3D release hygiene without importing GPU dependencies."""
import ast
import io
import json
from pathlib import Path
import re
import tokenize

ROOT = Path(__file__).resolve().parents[1]


def source_file(path):
    excluded={'.git', '.venv', 'external', '__pycache__', '.pytest_cache', 'build', 'dist'}
    return path.is_file() and not any(part in excluded or part.startswith('._') or part.endswith('.egg-info') for part in path.relative_to(ROOT).parts)


def check():
    problems = []
    python_files = [p for p in ROOT.rglob('*.py') if source_file(p)]
    for path in python_files:
        if any(part in {'.git', '.venv', 'external', '__pycache__'} for part in path.parts):
            continue
        source = path.read_text()
        relative = str(path.relative_to(ROOT))
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as error:
            problems.append(f'{relative}: {error}')
            continue
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and re.search(r'[\u4e00-\u9fff]', token.string):
                problems.append(f'{relative}:{token.start[0]}: comment is not English')
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc and re.search(r'[\u4e00-\u9fff]', doc):
                    problems.append(f'{relative}:{getattr(node, "lineno", 1)}: docstring is not English')
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                prefix = node.module.split('.')[0]
                if prefix in {'optimization', 'preprocessing', 'scripts', 'tools'}:
                    target = ROOT.joinpath(*node.module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        problems.append(f'{relative}: missing local module {node.module}')
        if re.search(r'/(?:mnt/nas|home/(?:sjc|ybh)|Users/jc)(?:/|\b)', source):
            problems.append(f'{relative}: private machine path')
        if re.search(r'\bhf_[A-Za-z0-9]{25,}\b|\bsk-[A-Za-z0-9]{30,}\b', source):
            problems.append(f'{relative}: possible embedded credential')
    required = ['preprocessing/README.md', 'generation/README.md', 'optimization/README.md',
                'configs/ho3d.json', 'configs/ho3d_scenes.json', 'configs/ho3d_evaluation.json']
    for name in required:
        if not (ROOT/name).is_file(): problems.append(f'Missing {name}')
    for path in ROOT.rglob('*.json'):
        if not source_file(path): continue
        json.loads(path.read_text())
        if 'dexycb' in path.name.lower(): problems.append(f'Unexpected deferred dataset configuration: {path.name}')
    for path in ROOT.rglob('*'):
        if not source_file(path) or path.suffix not in {'.py','.json','.yaml','.yml','.md','.toml','.sh'}:
            continue
        if re.search(r'\bhf_[A-Za-z0-9]{25,}\b|\bsk-[A-Za-z0-9]{30,}\b', path.read_text()):
            problems.append(f'{path.relative_to(ROOT)}: possible embedded credential')
    scenes=json.loads((ROOT/'configs/ho3d_scenes.json').read_text())
    if len(scenes) != 18: problems.append('Expected 18 released HO3D sequences')
    print(json.dumps({'python_files':len(python_files),'ho3d_scenes':len(scenes),'problems':problems},indent=2))
    return not problems


if __name__=='__main__':
    raise SystemExit(0 if check() else 1)
