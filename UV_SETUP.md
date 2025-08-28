# UV Setup Guide

This project now uses [UV](https://docs.astral.sh/uv/) for fast Python package management and environment handling.

## Installation

### Install UV
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Project Setup
```bash
# Clone and setup the project
cd glossary
uv sync  # Installs all dependencies and creates virtual environment

# Activate the environment (optional - uv commands work without activation)
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## Usage

### Running Commands
```bash
# Option 1: Direct uv run (recommended)
uv run glossary-lv0-s0  # Runs Level 0 Step 0
uv run glossary-lv1-s1  # Runs Level 1 Step 1
uv run glossary-web-miner --help  # Shows help for web mining

# Option 2: Traditional python -m (still works)
uv run python -m generate_glossary.generation.lv0.lv0_s0_get_college_names

# Option 3: After activation
glossary-lv0-s0  # Works after source .venv/bin/activate
```

### Development Commands
```bash
# Install development dependencies
uv sync --group dev

# Run tests
uv run pytest

# Code formatting
uv run black .
uv run isort .

# Type checking
uv run mypy .
```

### Adding Dependencies
```bash
# Add runtime dependency
uv add requests

# Add development dependency  
uv add --group dev pytest

# Remove dependency
uv remove requests
```

## Migration Benefits

### Before (Manual Path Management)
- ❌ Scripts broke when run from different directories
- ❌ Manual `sys.path.insert()` in every file  
- ❌ Deployment issues in containers/CI
- ❌ Import path fragility

### After (UV + pyproject.toml)
- ✅ Works from any directory
- ✅ Clean imports without path manipulation
- ✅ Production-ready deployment  
- ✅ Fast dependency resolution with UV
- ✅ Proper console scripts for all commands

## Available Console Scripts

All generation pipeline commands are now available as console scripts:

### Level 0 (Colleges)
- `glossary-lv0-s0` - Extract college names
- `glossary-lv0-s1` - Extract concepts
- `glossary-lv0-s2` - Filter by frequency
- `glossary-lv0-s3` - Verify single tokens

### Level 1 (Departments)  
- `glossary-lv1-s0` - Extract department names
- `glossary-lv1-s1` - Extract concepts
- `glossary-lv1-s2` - Filter by frequency
- `glossary-lv1-s3` - Verify single tokens

### Level 2 (Research Areas)
- `glossary-lv2-s0` - Extract research areas
- `glossary-lv2-s1` - Extract concepts
- `glossary-lv2-s2` - Filter by frequency  
- `glossary-lv2-s3` - Verify single tokens

### Level 3 (Conference Topics)
- `glossary-lv3-s0` - Extract conference topics
- `glossary-lv3-s1` - Extract concepts
- `glossary-lv3-s2` - Filter by frequency
- `glossary-lv3-s3` - Verify single tokens

### Post-Processing
- `glossary-web-miner` - Mine web content for concepts
- `glossary-validator` - Validate concept relevance
- `glossary-deduplicator` - Remove duplicate concepts

### Hierarchy Operations
- `glossary-hierarchy-build` - Build complete hierarchy
- `glossary-hierarchy-eval` - Evaluate hierarchy quality
- `glossary-hierarchy-viz` - Start web visualization

### Definition Generation
- `glossary-generate-definitions` - Generate definitions for terms

## Troubleshooting

### Common Issues

**Command not found**: Make sure you've run `uv sync` first
```bash
uv sync
uv run glossary-lv0-s0
```

**Import errors**: The package structure is now properly configured, old `sys.path` manipulation has been removed

**Environment issues**: UV automatically manages the virtual environment
```bash
uv sync  # Recreates environment if needed
```

### Performance
- UV is significantly faster than pip for dependency resolution
- Virtual environments are created automatically
- Lock files ensure reproducible builds across systems