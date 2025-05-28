#!/bin/bash

# Exit on error
set -e

echo "Running CI checks locally..."

# Code formatting check (Black)
echo -e "\n=== Running Black ==="
python3 -m black --check --diff .

# Import sorting check (isort)
echo -e "\n=== Running isort ==="
python3 -m isort --check-only --diff .

# Lint with flake8
echo -e "\n=== Running flake8 ==="
python3 -m flake8 .

# Type checking with mypy
echo -e "\n=== Running mypy ==="
python3 -m mypy .

# Run tests with coverage
echo -e "\n=== Running tests ==="
python3 -m pytest --cov=. --cov-report=xml --cov-report=term-missing

echo -e "\nAll checks passed! 🎉"
