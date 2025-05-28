#!/usr/bin/env python3
"""
Run fast tests only (excluding tests marked with @pytest.mark.slow).
This script is used by pre-commit hooks to avoid running slow tests.
"""

import sys
import subprocess


def main():
    """Run pytest with fast tests only."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "-m", "not slow",  # Exclude slow tests
        "--tb=short",
        "-v"
    ]
    
    print("Running fast tests (excluding slow tests)...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main()) 