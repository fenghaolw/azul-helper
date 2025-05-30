#!/usr/bin/env python3
"""
Run mypy type checking only on modified Python files in the current branch.
This is useful for CI/CD pipelines and local development to speed up type checking.
"""

import configparser
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_project_root() -> Path:
    """
    Find the project root directory by looking for specific marker files.

    Returns:
        Path to the project root directory
    """
    current_dir = Path(__file__).parent.absolute()

    # Look for project markers (pyproject.toml, .git, etc.)
    markers = ["pyproject.toml", ".git", "requirements.txt"]

    # Start from the script directory and go up
    search_dir = current_dir
    while search_dir != search_dir.parent:
        for marker in markers:
            if (search_dir / marker).exists():
                return search_dir
        search_dir = search_dir.parent

    # If no markers found, assume parent of scripts directory is root
    return current_dir.parent


def read_mypy_config(config_file: str) -> Optional[List[str]]:
    """
    Read exclude patterns from mypy config file.

    Args:
        config_file: Path to mypy config file

    Returns:
        List of exclude patterns, or None if not found
    """
    if not Path(config_file).exists():
        return None

    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        if "mypy" in config and "exclude" in config["mypy"]:
            exclude_pattern = config["mypy"]["exclude"]
            # Remove parentheses and split by |
            exclude_pattern = exclude_pattern.strip("()")
            patterns = [pattern.strip() for pattern in exclude_pattern.split("|")]
            return patterns
    except Exception as e:
        print(f"Warning: Could not read mypy config: {e}")

    return None


def should_exclude_file(file_path: str, exclude_patterns: List[str]) -> bool:
    """
    Check if a file should be excluded based on exclude patterns.

    Args:
        file_path: Path to the file
        exclude_patterns: List of regex patterns to exclude

    Returns:
        True if file should be excluded
    """
    for pattern in exclude_patterns:
        if re.search(pattern, file_path):
            return True
    return False


def get_modified_files(base_branch: str = "main") -> List[str]:
    """
    Get list of modified Python files in current branch compared to base branch.

    Args:
        base_branch: The base branch to compare against (default: main)

    Returns:
        List of modified Python file paths
    """
    # Get the list of modified files
    cmd = ["git", "diff", "--name-only", f"origin/{base_branch}...HEAD"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error getting modified files: {result.stderr}")
        sys.exit(1)

    # Filter for Python files and apply exclusions
    modified_files = []
    for f in result.stdout.splitlines():
        if not f.endswith(".py") or not Path(f).exists():
            continue
        modified_files.append(f)

    return modified_files


def run_mypy(files: List[str], config_file: Optional[str] = None) -> int:
    """
    Run mypy on the specified files.

    Args:
        files: List of Python files to check
        config_file: Optional path to mypy config file

    Returns:
        Exit code from mypy
    """
    if not files:
        print("No modified Python files to check")
        return 0

    # Build mypy command
    cmd = ["mypy"]
    if config_file and Path(config_file).exists():
        cmd.extend(["--config-file", config_file])
    cmd.extend(files)

    # Run mypy
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    import argparse

    # Find project root and change to it
    project_root = find_project_root()
    os.chdir(project_root)

    # Default config file path relative to project root
    default_config = project_root / "mypy.ini"

    parser = argparse.ArgumentParser(
        description="Run mypy type checking on modified Python files"
    )
    parser.add_argument(
        "--base-branch",
        default="main",
        help="Base branch to compare against (default: main)",
    )
    parser.add_argument(
        "--config-file",
        default=str(default_config),
        help="Path to mypy config file (default: mypy.ini in project root)",
    )

    args = parser.parse_args()

    # Get modified files
    modified_files = get_modified_files(args.base_branch)

    # Read exclude patterns from config
    exclude_patterns = read_mypy_config(args.config_file)

    # Filter out excluded files
    if exclude_patterns:
        filtered_files = []
        excluded_files = []

        for file in modified_files:
            if should_exclude_file(file, exclude_patterns):
                excluded_files.append(file)
            else:
                filtered_files.append(file)

        modified_files = filtered_files

        if excluded_files:
            print(
                f"Excluded {len(excluded_files)} files based on mypy.ini exclude patterns:"
            )
            for f in excluded_files:
                print(f"  - {f}")
            print()

    if not modified_files:
        print("No modified Python files found (after applying exclude patterns)")
        return 0

    print(f"Checking {len(modified_files)} modified Python files:")
    for f in modified_files:
        print(f"  - {f}")
    print()

    # Run mypy
    return run_mypy(modified_files, args.config_file)


if __name__ == "__main__":
    sys.exit(main())
