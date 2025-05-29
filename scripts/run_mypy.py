#!/usr/bin/env python3
"""
Run mypy type checking only on modified Python files in the current branch.
This is useful for CI/CD pipelines and local development to speed up type checking.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_modified_files(
    base_branch: str = "main", exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Get list of modified Python files in current branch compared to base branch.

    Args:
        base_branch: The base branch to compare against (default: main)
        exclude_patterns: List of patterns to exclude (default: ['tests/', 'examples/'])

    Returns:
        List of modified Python file paths
    """
    if exclude_patterns is None:
        exclude_patterns = ["tests/", "examples/"]

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
        # Skip files matching exclude patterns
        if any(pattern in f for pattern in exclude_patterns):
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
    if config_file:
        cmd.extend(["--config-file", config_file])
    cmd.extend(files)

    # Run mypy
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    import argparse

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
        help="Path to mypy config file (default: mypy.ini in current directory)",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in type checking",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="Include example files in type checking",
    )

    args = parser.parse_args()

    # Determine exclude patterns
    exclude_patterns = []
    if not args.include_tests:
        exclude_patterns.append("tests/")
    if not args.include_examples:
        exclude_patterns.append("examples/")

    # Get modified files
    modified_files = get_modified_files(args.base_branch, exclude_patterns)

    if not modified_files:
        print("No modified Python files found")
        return 0

    print(f"Checking {len(modified_files)} modified Python files:")
    for f in modified_files:
        print(f"  - {f}")
    print()

    # Run mypy
    return run_mypy(modified_files, args.config_file)


if __name__ == "__main__":
    sys.exit(main())
