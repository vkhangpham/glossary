# Firecrawl v2.2.0 Mining Module Testing Guide

This directory contains comprehensive tests for the Firecrawl v2.2.0 mining module integration. The testing infrastructure validates all v2.2.0 features, ensures backward compatibility, and provides performance benchmarking.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Test Categories](#test-categories)
- [Quick Start](#quick-start)
- [Test Execution](#test-execution)
- [Environment Setup](#environment-setup)
- [Test Coverage](#test-coverage)
- [Development Guidelines](#development-guidelines)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Testing Overview

### Testing Strategy

The testing strategy for the Firecrawl v2.2.0 mining module covers:

1. **Unit Tests**: Fast, isolated tests with no external dependencies
2. **Integration Tests**: Real API interactions validating v2.2.0 features
3. **Performance Tests**: Benchmarking and performance regression detection
4. **Compatibility Tests**: Backward compatibility and SDK validation
5. **Error Handling Tests**: Comprehensive error scenarios and fallbacks

### Key Features Tested

- ✅ **v2.2.0 Features**: PDF page limits, queue monitoring, fast mapping, webhooks
- ✅ **API Compatibility**: Search categories, batch scraping, structured extraction
- ✅ **Performance**: 500% batch scraping improvement, 15x Map endpoint speedup
- ✅ **Error Handling**: Graceful degradation, fallback mechanisms
- ✅ **Backward Compatibility**: Legacy function aliases, CLI compatibility
- ✅ **Logging & Metrics**: Feature usage tracking, correlation IDs

## Test Categories

### Unit Tests (`tests/mining/test_*.py`)

**No external dependencies required** - Run with `make test-unit`

| Test Module | Description | Key Areas |
|-------------|-------------|-----------|
| `test_models.py` | Pydantic model validation | ConceptDefinition, QueueStatus, WebhookConfig |
| `test_cli.py` | CLI argument parsing | v2.2.0 parameters, legacy compatibility |
| `test_client.py` | Client management | API key handling, singleton pattern |
| `test_backward_compatibility.py` | Legacy support | Function aliases, import paths |

### Integration Tests (`@pytest.mark.integration`)

**Requires FIRECRAWL_API_KEY** - Run with `make test-integration`

| Test Module | Description | API Features Tested |
|-------------|-------------|-------------------|
| `test_integration.py` | Live API validation | Search categories, batch scraping, queue monitoring |

### Performance Tests (`@pytest.mark.slow`)

**Long-running tests** - Run with `make test-performance`

| Test Module | Description | Performance Claims |
|-------------|-------------|-------------------|
| `test_performance.py` | Performance benchmarking | 500% batch improvement, 15x Map speedup |

### Error Handling Tests

| Test Module | Description | Error Scenarios |
|-------------|-------------|----------------|
| `test_error_handling.py` | Comprehensive error testing | Network failures, API fallbacks, graceful degradation |

### Compatibility Validation

| Test Module | Description | Validation Areas |
|-------------|-------------|-----------------|
| `test_firecrawl_compatibility.py` | SDK compatibility check | Method availability, parameter support |

## Quick Start

### 1. Basic Test Execution

```bash
# Run unit tests (fast, no API key needed)
make test-unit

# Run all tests
make test-all

# Run with coverage
make test-coverage
```

### 2. With API Key (Integration Tests)

```bash
# Set your Firecrawl API key
export FIRECRAWL_API_KEY=your_api_key_here

# Run integration tests
make test-integration

# Run complete test suite
make test-all
```

### 3. Quick Compatibility Check

```bash
# Validate Firecrawl SDK compatibility
make test-compatibility

# Or run directly
python tests/test_firecrawl_compatibility.py
```

## Test Execution

### Available Make Commands

```bash
# Test Execution
make test-unit              # Unit tests only (fast)
make test-integration       # Integration tests (requires API key)
make test-performance       # Performance tests (slow)
make test-all              # Complete test suite
make test-compatibility    # SDK compatibility check

# Coverage and Reporting
make test-coverage         # Tests with coverage reporting
make test-report          # Generate HTML coverage report
make test-ci              # CI-friendly execution

# Development
make test-watch           # Watch mode for development
make test-debug           # Debug output and logging
make test-clean           # Clean test artifacts

# Environment
make setup-test-env       # Set up test environment
make check-api-key        # Validate API key configuration
make test-env-check       # Environment validation
```

### Pytest Markers

Tests are organized using pytest markers:

```bash
# Run specific test categories
pytest -m unit tests/           # Unit tests only
pytest -m integration tests/    # Integration tests only
pytest -m slow tests/           # Performance tests only
pytest -m "not slow" tests/     # Skip slow tests
```

### Advanced Test Execution

```bash
# Parallel execution
make test-parallel

# Specific test
make test-specific TEST=test_models.py::TestConceptDefinition

# Failed tests only
make test-failed

# Modified files only
make test-modified
```

## Environment Setup

### Prerequisites

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up project environment
uv sync --dev

# Set up test environment
make setup-test-env
```

### Environment Variables

```bash
# Required for integration tests
export FIRECRAWL_API_KEY=your_api_key_here

# Optional configuration
export LOG_LEVEL=DEBUG
export MAX_CONCURRENT=10
```

### Validate Environment

```bash
# Check environment setup
make test-env-check

# Output example:
# ✓ pytest 7.4.0
# ✓ mining module importable
# ✓ FIRECRAWL_API_KEY configured
```

## Test Coverage

### Coverage Requirements

- **Minimum**: 80% overall coverage
- **Target**: 90%+ for core modules
- **Critical Paths**: 95%+ for error handling

### Coverage Reports

```bash
# Generate coverage report
make test-coverage

# View HTML report
make test-report

# Coverage is generated in htmlcov/index.html
```

### Coverage by Module

| Module | Target Coverage | Key Areas |
|--------|----------------|-----------|
| `models.py` | 95% | Pydantic validation, serialization |
| `client.py` | 90% | API key management, health checks |
| `mining.py` | 85% | Main mining pipeline |
| `cli.py` | 80% | Argument parsing, validation |

## Development Guidelines

### Writing New Tests

1. **Use Appropriate Markers**:
   ```python
   @pytest.mark.unit
   def test_unit_functionality():
       pass

   @pytest.mark.integration
   def test_api_integration():
       pass
   ```

2. **Use Shared Fixtures**:
   ```python
   def test_with_fixtures(sample_concepts, mock_firecrawl_app):
       # Fixtures provide consistent test data
       pass
   ```

3. **Follow Naming Conventions**:
   - Test files: `test_*.py`
   - Test classes: `TestClassName`
   - Test methods: `test_method_description`

4. **Use Descriptive Test Names**:
   ```python
   def test_concept_definition_validates_required_fields():
       """Test that ConceptDefinition requires concept, definition, and context."""
   ```

### Test Data Management

- **Fixtures**: Use `conftest.py` fixtures for shared test data
- **Mock Data**: Prefer mock data over real API calls in unit tests
- **Test Files**: Clean up temporary files in test teardown

### Performance Testing Guidelines

1. **Use Mock Data**: Don't hammer real APIs in performance tests
2. **Measure Consistently**: Use timing utilities from fixtures
3. **Set Realistic Targets**: Base targets on actual performance claims
4. **Include Variance**: Account for system performance differences

### Error Testing Best Practices

1. **Test All Error Paths**: Network, API, validation errors
2. **Verify Error Messages**: Ensure helpful error messages
3. **Test Fallback Behavior**: Validate graceful degradation
4. **Check Resource Cleanup**: Ensure proper cleanup on errors

## CI/CD Integration

### GitHub Actions

The test suite is designed for GitHub Actions integration:

```yaml
# Example GitHub Actions workflow
name: Test Firecrawl Mining Module

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up environment
        run: make github-test-setup

      - name: Run unit tests
        run: make github-test-unit

      - name: Run integration tests
        if: ${{ secrets.FIRECRAWL_API_KEY }}
        env:
          FIRECRAWL_API_KEY: ${{ secrets.FIRECRAWL_API_KEY }}
        run: make github-test-integration
```

### CI Commands

```bash
# CI-specific commands
make test-ci               # Complete CI test execution
make github-test-setup     # GitHub Actions setup
make github-test-unit      # GitHub Actions unit tests
make github-test-integration # GitHub Actions integration tests
```

### Test Reports

CI generates JUnit XML reports in `test-reports/`:

- `junit-unit.xml` - Unit test results
- `junit-integration.xml` - Integration test results
- `coverage-unit.xml` - Unit test coverage
- `coverage-integration.xml` - Integration test coverage

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'generate_glossary'
# Solution: Ensure you're in the project root and using uv
cd /path/to/glossary
uv run pytest tests/
```

#### 2. API Key Issues

```bash
# Error: Integration tests skipped (no API key)
# Solution: Set FIRECRAWL_API_KEY environment variable
export FIRECRAWL_API_KEY=fc-your-api-key-here
make check-api-key
```

#### 3. Firecrawl SDK Not Found

```bash
# Error: FirecrawlApp not available
# Solution: Install firecrawl-py
uv add firecrawl-py
```

#### 4. Test Failures

```bash
# Re-run only failed tests
make test-failed

# Run with debug output
make test-debug

# Check specific test
make test-specific TEST=test_models.py::TestConceptDefinition::test_validation
```

### Debug Commands

```bash
# Environment diagnostics
make test-info

# Clean and retry
make test-clean
make test-unit

# Verbose output
pytest tests/ -v -s --tb=long
```

### Performance Issues

```bash
# Skip slow tests
pytest -m "not slow" tests/

# Parallel execution
make test-parallel

# Profile test execution
pytest --durations=20 tests/
```

## Test Structure Reference

```
tests/
├── README.md                          # This file
├── conftest.py                        # Shared fixtures and configuration
├── test_firecrawl_compatibility.py    # SDK compatibility validation
└── mining/                            # Mining module tests
    ├── __init__.py                    # Package initialization
    ├── conftest.py                    # Mining-specific fixtures
    ├── test_models.py                 # Pydantic model tests
    ├── test_cli.py                    # CLI testing
    ├── test_client.py                 # Client management tests
    ├── test_integration.py            # Live API tests
    ├── test_performance.py            # Performance benchmarks
    ├── test_error_handling.py         # Error scenarios
    ├── test_logging_metrics.py        # Logging validation
    └── test_backward_compatibility.py # Legacy support tests
```

## Getting Help

### Documentation

- **API Documentation**: Check `generate_glossary/mining/` module docstrings
- **Test Examples**: Look at existing test files for patterns
- **Configuration**: Review `pytest.ini` and `conftest.py` files

### Common Test Patterns

```python
# Unit test with mocking
@patch('generate_glossary.mining.client.get_client')
def test_function_with_mock(mock_get_client):
    mock_get_client.return_value = Mock()
    # Test logic here

# Integration test with fixtures
@pytest.mark.integration
def test_live_api(live_firecrawl_client, sample_concepts):
    result = function_under_test(sample_concepts)
    assert result is not None

# Performance test with timing
@pytest.mark.slow
def test_performance(timing_utilities):
    with timing_utilities.time_operation() as timer:
        # Performance-critical operation
        pass
    assert timer.duration < 5.0  # Max 5 seconds
```

### Support

For issues with the testing infrastructure:

1. Check this README for common solutions
2. Review test output for specific error messages
3. Use debug commands to get more information
4. Check the project's main documentation

---

**Happy Testing!** 🧪

This comprehensive test suite ensures the Firecrawl v2.2.0 mining module works correctly, performs well, and maintains backward compatibility. Use the make commands for convenient test execution and refer to this guide for detailed testing workflows.