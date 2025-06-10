.PHONY: help install install-dev format lint type-check test test-verbose clean all check

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install package dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Run flake8 linter"
	@echo "  type-check   - Run mypy type checker"
	@echo "  test         - Run tests with pytest"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  check        - Run all checks (format, lint, type-check, test)"
	@echo "  clean        - Clean up cache files"
	@echo "  pre-commit   - Install pre-commit hooks"

# Installation
install:
	pip3 install -e .

install-dev:
	pip3 install -e ".[dev,rl,viz]"

# Code formatting
format:
	@echo "Running isort..."
	python3 -m isort .
	@echo "Running black..."
	python3 -m black .
	@echo "✓ Code formatting complete"

# Linting
lint:
	@echo "Running flake8..."
	python3 -m flake8 .
	@echo "✓ Linting complete"

# Type checking
type-check:
	@echo "Running mypy..."
	python3 -m mypy .
	@echo "✓ Type checking complete"

# Testing
test:
	@echo "Running tests..."
	python3 -m pytest tests/ -m "not slow"
	@echo "✓ Tests complete"

test-verbose:
	@echo "Running tests with verbose output..."
	python3 -m pytest tests/ -v
	@echo "✓ Tests complete"

# Run all checks
check: format lint type-check test
	@echo "✓ All checks passed!"

# Pre-commit setup
pre-commit:
	@echo "Installing pre-commit hooks..."
	python3 -m pre_commit install
	@echo "✓ Pre-commit hooks installed"

# Clean up
clean:
	@echo "Cleaning up cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "✓ Cleanup complete"
