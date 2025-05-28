# Dependency Optimization for Faster CI

## Summary

Optimized dependencies to reduce CI installation time from ~1m30s to an estimated ~20-30s by removing unused packages.

## Changes Made

### 1. Core Dependencies (`azul_rl/requirements.txt`)
**Removed (commented out):**
- `torch>=1.9.0` - Not currently used (planned for future RL development)
- `gymnasium>=0.26.0` - Not currently used (planned for future RL development)
- `stable-baselines3>=1.6.0` - Not currently used (planned for future RL development)
- `matplotlib>=3.5.0` - Not currently used (planned for visualization)
- `seaborn>=0.11.0` - Not currently used (planned for visualization)
- `pandas>=1.3.0` - Not currently used (planned for data analysis)

**Kept:**
- `numpy>=1.21.0` - Used in `state_representation.py`
- `typing-extensions>=4.0.0` - Used throughout codebase

### 2. Development Dependencies (`azul_rl/requirements-dev.txt`)
**Removed (commented out):**
- `pytest-mock>=3.6.0` - Not actually used in tests
- `flake8-import-order>=0.18.0` - Not configured in `.flake8`
- `flake8-bugbear>=22.0.0` - Not configured in `.flake8`
- All `types-*` packages - Not needed (no imports of requests, setuptools, PyYAML, etc.)

**Kept:**
- Essential tools: `black`, `flake8`, `isort`, `mypy`, `pre-commit`
- Testing: `pytest`, `pytest-cov`
- Performance monitoring: `psutil` (used in performance tests)
- `flake8-docstrings` (configured in `.flake8`)

### 3. Updated pyproject.toml
Reorganized optional dependencies into logical groups:
- `[dev]` - Essential CI dependencies only
- `[rl]` - RL libraries for future development
- `[viz]` - Visualization libraries
- `[dev-extra]` - Additional dev tools not needed for CI

### 4. Updated CI Workflow
Changed from installing all dev dependencies to installing only essential ones:
```bash
# Before
pip install -r requirements.txt
pip install -r requirements-dev.txt

# After
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Expected Performance Improvement

**Heavy packages removed from CI:**
- PyTorch (~800MB) - Largest impact
- Stable Baselines3 (~100MB)
- Matplotlib/Seaborn/Pandas (~150MB combined)
- Various type stubs (~50MB combined)

**Estimated CI time reduction:**
- Before: ~1m30s installation time
- After: ~20-30s installation time
- **Improvement: ~60-70% faster dependency installation**

## How to Install Optional Dependencies

When you need the removed dependencies for development:

```bash
# For RL development
pip install -e ".[rl]"

# For visualization work
pip install -e ".[viz]"

# For additional dev tools
pip install -e ".[dev-extra]"

# Install everything
pip install -e ".[dev,rl,viz,dev-extra]"
```

## Verification

To verify the optimization worked:
1. Check CI logs for reduced installation time
2. Ensure all CI checks still pass
3. Confirm core functionality works with minimal dependencies

## Future Considerations

- When RL features are implemented, add `[rl]` dependencies to CI
- When visualization features are added, add `[viz]` dependencies to CI
- Monitor CI performance and adjust as needed
