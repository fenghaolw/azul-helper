#!/usr/bin/env python3
"""Script to run all linting and formatting tools."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Run all linting and formatting tools."""
    print("🚀 Running linting and formatting tools...")

    # Change to project directory
    project_dir = Path(__file__).parent.parent
    print(f"Working directory: {project_dir}")

    commands = [
        (["python3", "-m", "isort", "."], "Sorting imports with isort"),
        (["python3", "-m", "black", "."], "Formatting code with Black"),
        (["python3", "-m", "flake8", ".", "--statistics"], "Linting with Flake8"),
    ]

    success_count = 0
    for cmd, description in commands:
        if run_command(cmd, description):
            success_count += 1

    print(f"\n📊 Results: {success_count}/{len(commands)} tools completed successfully")

    if success_count == len(commands):
        print("🎉 All checks passed!")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
