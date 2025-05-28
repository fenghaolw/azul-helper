# Development Guide

This guide explains how to set up and use the development tools for the Azul RL project.

## Quick Setup

```bash
# Install development dependencies
pip3 install -r requirements-dev.txt

# Or install everything including RL and visualization dependencies
pip3 install -e ".[dev,rl,viz]"
```

## Available Tools

### Code Formatting

- **Black**: Automatic code formatter
- **isort**: Import statement organizer

### Code Quality

- **Flake8**: Linting and style checking
- **MyPy**: Static type checking (optional)

### Testing

- **pytest**: Test runner with coverage support

## Using Make Commands

The project includes a Makefile with convenient commands:

```bash
# Show all available commands
make help

# Format code (runs isort + black)
make format

# Run linter
make lint

# Run type checker
make type-check

# Run tests
make test

# Run tests with verbose output
make test-verbose

# Run all checks (format + lint + type-check + test)
make check

# Clean up cache files
make clean

# Run example game
make example
```

## Manual Tool Usage

You can also run tools manually:

```bash
# Format code
python3 -m black .
python3 -m isort .

# Check formatting without applying changes
python3 -m black --check --diff .

# Run linter
python3 -m flake8 .

# Run type checker
python3 -m mypy .

# Run tests
python3 -m pytest tests/
python3 -m pytest tests/ -v  # verbose
python3 -m pytest tests/ --cov=azul_rl  # with coverage
```

## Pre-commit Hooks (Optional)

Set up automatic formatting and linting on git commits:

```bash
# Install pre-commit hooks
python3 -m pre_commit install

# Run hooks manually on all files
python3 -m pre_commit run --all-files
```

## Configuration Files

The project includes configuration for all tools:

- `.flake8` - Flake8 linting configuration
- `pyproject.toml` - Black, isort, mypy, and pytest configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

## Development Workflow

1. **Before coding**: Run `make format` to ensure consistent formatting
2. **During development**: Use your IDE's integration with these tools
3. **Before committing**: Run `make check` to ensure all checks pass
4. **Optional**: Set up pre-commit hooks for automatic checking

## IDE Integration

### VS Code

Install these extensions:
- Python (Microsoft)
- Black Formatter
- Flake8
- isort

Add to your VS Code settings:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true
}
```

### PyCharm

1. Go to Settings → Tools → External Tools
2. Add Black and isort as external tools
3. Configure Flake8 in Settings → Editor → Inspections → Python

## Troubleshooting

### Tools not found
If you get "command not found" errors, the tools might not be in your PATH. Use `python3 -m` prefix:
```bash
python3 -m black .
python3 -m flake8 .
```

### Import order conflicts
We use isort for import organization and ignore Flake8's import order checks (I100, I101) to avoid conflicts.

### Type checking errors
MyPy is configured but optional. You can ignore type checking errors during initial development by skipping `make type-check`.

## Adding New Dependencies

When adding new dependencies:

1. Add to `requirements.txt` (core dependencies)
2. Add to `requirements-dev.txt` (development dependencies)
3. Update `pyproject.toml` optional dependencies if needed
4. Run `make install-dev` to install new dependencies

## Python Setup

### Python Executable
This project uses `python3` explicitly in scripts and configuration files to avoid issues on systems where `python` is not available.

If you prefer to use `python` instead of `python3`, you can:

1. **Temporary alias (current session only):**
   ```bash
   source scripts/setup_python_alias.sh
   ```

2. **Permanent alias (recommended):**
   Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):
   ```bash
   alias python=python3
   ```

3. **System-wide symlink (advanced users):**
   ```bash
   sudo ln -s $(which python3) /usr/local/bin/python
   ```
