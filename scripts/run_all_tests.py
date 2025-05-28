#!/usr/bin/env python3
"""
Run all tests including slow ones.
This script is used in CI and before deployment to ensure all functionality works.
"""

import sys
import subprocess


def main():
    """Run pytest with all tests including slow ones."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=.", 
        "--cov-report=xml",
        "--cov-report=term-missing",
        "--tb=short",
        "-v"
    ]
    
    print("Running all tests (including slow tests)...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main()) 