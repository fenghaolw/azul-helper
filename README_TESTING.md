# Testing Setup

## Overview

This project uses a two-tier testing approach to balance development speed with thorough testing:

- **Fast tests**: Run quickly during development and pre-commit hooks
- **Slow tests**: Include comprehensive integration tests that run before deployment

## Test Categories

### Fast Tests (Pre-commit)
- Run in ~12 seconds
- Include unit tests, basic functionality, and most integration tests
- Exclude neural network training and self-play tests
- **Usage**: `python3 scripts/run_fast_tests.py`

### Slow Tests (CI/Deployment)
- Run in ~2.5 minutes 
- Include neural network training and self-play training tests
- These tests verify the full ML pipeline works correctly
- **Usage**: `python3 scripts/run_all_tests.py`

## Commands

```bash
# Run fast tests only (for development)
python3 scripts/run_fast_tests.py

# Run slow tests only
python3 -m pytest -m "slow" -v

# Run all tests (for CI/deployment)
python3 scripts/run_all_tests.py

# Run all tests except slow ones (equivalent to fast tests)
python3 -m pytest -m "not slow" -v
```

## Pre-commit Integration

The pre-commit hooks automatically run fast tests only:

```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit checks (includes fast tests)
pre-commit run --all-files
```

## CI Integration

The GitHub Actions CI runs all tests including slow ones to ensure complete functionality before deployment.

## Adding New Tests

### For Regular Tests
Just create test functions normally - they'll be included in fast tests by default.

### For Slow Tests
Mark tests that involve neural network training, self-play, or other time-intensive operations:

```python
import pytest

@pytest.mark.slow
def test_neural_network_training():
    # Time-intensive test here
    pass
```

## Python Executable

This project uses `python3` explicitly to avoid issues on systems where `python` is not available.

If you prefer using `python`, you can:
```bash
# Temporary alias
source scripts/setup_python_alias.sh

# Or add to your shell profile
echo "alias python=python3" >> ~/.bashrc  # or ~/.zshrc
``` 