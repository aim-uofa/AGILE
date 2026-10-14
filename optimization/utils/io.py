import random
import numpy as np
import torch

def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PrintColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

"""
Utility module for saving code snapshots.
Saves all source code files to the result directory for reproducibility.
"""
import os
import shutil
from pathlib import Path
from typing import List, Set
from datetime import datetime


def get_code_extensions() -> Set[str]:
    """
    Get the set of file extensions to include in code snapshot.

    Returns:
        Set of file extensions (including the dot)
    """
    return {
        '.py',      # Python source files
        '.sh',      # Shell scripts
        '.yaml',    # YAML configs
        '.yml',     # YAML configs
        '.json',    # JSON configs
        '.md',      # Markdown documentation
        '.txt',     # Text files (requirements, etc.)
    }


def get_exclude_patterns() -> Set[str]:
    """
    Get directory/file patterns to exclude from code snapshot.

    Returns:
        Set of directory/file names to exclude
    """
    return {
        # Cache directories
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.tox',
        'node_modules',

        # Build directories
        'build',
        'dist',
        '*.egg-info',

        # Data directories
        'data',
        'outputs',
        'results',
        'logs',
        'checkpoints',
        'ckpts',
        'runs',

        # Version control
        '.git',
        '.gitignore',
        '.gitattributes',

        # IDE directories
        '.vscode',
        '.idea',
        '.claude',

        # Environment directories
        'venv',
        'env',
        '.env',
        'conda-env',

        # System files
        '.DS_Store',
        'Thumbs.db',

        # Temporary files
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.Python',
        '*.so',
        '*.dylib',
        '*.swp',
        '*.swo',
        '*~',
    }


def should_exclude(path: Path, project_root: Path, exclude_patterns: Set[str]) -> bool:
    """
    Check if a path should be excluded from the snapshot.

    Args:
        path: Path to check
        project_root: Root directory of the project
        exclude_patterns: Set of patterns to exclude

    Returns:
        True if the path should be excluded, False otherwise
    """
    # Get relative path from project root
    try:
        rel_path = path.relative_to(project_root)
    except ValueError:
        # Path is not relative to project root
        return True

    # Check each component of the path
    for part in rel_path.parts:
        if part in exclude_patterns:
            return True

        # Check wildcard patterns
        for pattern in exclude_patterns:
            if '*' in pattern:
                import fnmatch
                if fnmatch.fnmatch(part, pattern):
                    return True

    # Check if filename matches any exclude pattern
    if path.name in exclude_patterns:
        return True

    return False


def collect_code_files(
    project_root: str,
    code_extensions: Set[str],
    exclude_patterns: Set[str]
) -> List[Path]:
    """
    Collect all code files from the project directory.

    Args:
        project_root: Root directory of the project
        code_extensions: Set of file extensions to include
        exclude_patterns: Set of patterns to exclude

    Returns:
        List of Path objects for code files
    """
    project_path = Path(project_root).resolve()
    code_files = []

    for root, dirs, files in os.walk(project_path):
        root_path = Path(root)

        # Filter out excluded directories (modify dirs in-place to prevent recursion)
        dirs[:] = [
            d for d in dirs
            if not should_exclude(root_path / d, project_path, exclude_patterns)
        ]

        # Collect code files
        for file in files:
            file_path = root_path / file

            # Check if file has a code extension
            if file_path.suffix in code_extensions:
                # Check if file should be excluded
                if not should_exclude(file_path, project_path, exclude_patterns):
                    code_files.append(file_path)

    return sorted(code_files)


def save_code_snapshot(
    project_root: str,
    result_dir: str,
    snapshot_name: str = "code_snapshot"
) -> str:
    """
    Save a snapshot of all source code to the result directory.

    Args:
        project_root: Root directory of the project
        result_dir: Directory where results are saved
        snapshot_name: Name of the snapshot subdirectory (default: "code_snapshot")

    Returns:
        Path to the created snapshot directory
    """
    project_path = Path(project_root).resolve()
    result_path = Path(result_dir).resolve()
    snapshot_dir = result_path / snapshot_name

    # Create snapshot directory
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Get code extensions and exclude patterns
    code_extensions = get_code_extensions()
    exclude_patterns = get_exclude_patterns()

    # Collect code files
    print(f"Collecting code files from {project_root}...")
    code_files = collect_code_files(project_root, code_extensions, exclude_patterns)

    print(f"Found {len(code_files)} code files to save")

    # Copy each file preserving directory structure
    for file_path in code_files:
        # Get relative path from project root
        rel_path = file_path.relative_to(project_path)

        # Determine destination path
        dest_path = snapshot_dir / rel_path

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(file_path, dest_path)

    # Create a metadata file
    metadata_path = snapshot_dir / "_snapshot_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Code Snapshot Metadata\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Project Root: {project_root}\n")
        f.write(f"Result Directory: {result_dir}\n")
        f.write(f"Total Files: {len(code_files)}\n\n")
        f.write(f"File List:\n")
        f.write("-" * 50 + "\n")

        for file_path in code_files:
            rel_path = file_path.relative_to(project_path)
            f.write(f"{rel_path}\n")

    print(f"Code snapshot saved to: {snapshot_dir}")
    print(f"Metadata saved to: {metadata_path}")

    return str(snapshot_dir)
