# Glossary Generation System Tests

This directory contains comprehensive tests for the glossary generation system, with a focus on testing the new simplified LLM integration using LiteLLM + Instructor.

## Test Structure

```
tests/
├── unit/                          # Unit tests (fast, no external dependencies)
│   ├── test_llm_simple.py        # Core LLM functionality tests
│   └── test_migration_compatibility.py  # Migration compatibility tests
├── integration/                   # Integration tests (require API keys)
│   └── test_llm_integration.py    # Real API call tests
├── fixtures/                      # Test data and fixtures
├── conftest.py                   # Pytest configuration and fixtures
├── run_tests.py                  # Test runner script
└── README.md                     # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Fast execution** (no network calls)
- **No external dependencies** (uses mocks)
- **Core functionality testing** of the LLM simple module
- **Migration compatibility** ensuring old patterns work with new system

### Integration Tests (`tests/integration/`)
- **Real API calls** to LLM providers
- **Require valid API keys** (OPENAI_API_KEY, GEMINI_API_KEY)
- **End-to-end functionality** testing
- **Provider compatibility** verification

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests (fast, no API keys needed)
python tests/run_tests.py --unit

# Run only integration tests (requires API keys)
python tests/run_tests.py --integration

# Run with coverage report
python tests/run_tests.py --coverage
```

### Using Pytest Directly
```bash
# All tests
uv run pytest tests/

# Unit tests only
uv run pytest tests/unit/ -m unit

# Integration tests only
uv run pytest tests/integration/ -m integration

# Specific test file
uv run pytest tests/unit/test_llm_simple.py

# With coverage
uv run pytest tests/ --cov=generate_glossary.utils.llm_simple --cov-report=html
```

### Test Markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Environment Setup

### For Unit Tests
No special setup required - unit tests use mocks and don't make external calls.

### For Integration Tests
Set environment variables with valid API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

**Note:** Integration tests will be automatically skipped if API keys are not available.

## Test Features

### Comprehensive Coverage
- ✅ **Core LLM functionality** (structured/text completion)
- ✅ **Provider compatibility** (OpenAI, Anthropic/Gemini)
- ✅ **Migration compatibility** (old patterns work with new system)
- ✅ **Error handling** (API failures, invalid inputs)
- ✅ **Configuration** (random provider/model selection)
- ✅ **Message formatting** (system prompts, user prompts)

### Migration Testing
- ✅ **Concept extraction patterns** from `lv0_s1_extract_concepts.py`
- ✅ **Definition generation patterns** from `generate_definitions.py`
- ✅ **Validation patterns** from `validator/cli.py`
- ✅ **Sense disambiguation patterns** from `splitter.py`
- ✅ **Backward compatibility** with `LLMResult` wrapper

### Integration Testing
- ✅ **Real API calls** with actual responses
- ✅ **Provider switching** (OpenAI ↔ Gemini)
- ✅ **Structured responses** with Pydantic models
- ✅ **Academic concept extraction** with realistic text
- ✅ **Temperature effects** on response variation
- ✅ **Error scenarios** (invalid models, API failures)

## Test Data and Fixtures

### Fixtures Available (`conftest.py`)
- `temp_dir` - Temporary directory for test files
- `sample_concepts` - Sample academic concepts
- `sample_academic_text` - Realistic academic text for testing
- `sample_glossary_metadata` - Glossary data structure
- `sample_web_resources` - Web resource data
- `mock_llm_response` - Mock LLM response objects
- `env_with_api_keys` / `env_without_api_keys` - Environment setup

### Test Models
Pydantic models matching those used in production:
- `ConceptExtraction` - Single concept extraction result
- `ConceptExtractionList` - Batch concept extraction results
- `SimpleResponse` - Basic structured response
- `ConceptList` - List of concepts with count

## Performance and Cost

### Unit Tests
- **~10 seconds** total execution time
- **Zero cost** (no API calls)
- **No rate limits** (uses mocks)

### Integration Tests  
- **~30-60 seconds** execution time
- **~$0.01-0.05 cost** (small test prompts)
- **Respects rate limits** (includes delays if needed)

## Continuous Integration

Tests are designed to work in CI environments:
- Unit tests always run (no API keys needed)
- Integration tests skip gracefully without API keys
- Proper exit codes for CI/CD pipelines
- Coverage reporting available

## Adding New Tests

### For New LLM Functions
1. Add unit tests in `tests/unit/test_llm_simple.py`
2. Add integration tests in `tests/integration/test_llm_integration.py`
3. Add compatibility tests in `tests/unit/test_migration_compatibility.py`

### For New Migration Patterns
1. Identify the old LLM usage pattern
2. Create a test in `test_migration_compatibility.py`
3. Mock the new system and verify equivalent functionality

### Test Naming Convention
- `test_function_name_scenario` - Basic functionality
- `test_function_name_with_parameters` - Parameter variations  
- `test_function_name_error_handling` - Error scenarios
- `test_migration_pattern_name` - Migration compatibility

## Debugging Tests

### Verbose Output
```bash
python tests/run_tests.py --verbose
```

### Specific Test Debugging
```bash
uv run pytest tests/unit/test_llm_simple.py::TestLLMResult::test_llm_result_creation -v -s
```

### Coverage Analysis
```bash
python tests/run_tests.py --coverage
# Open htmlcov/index.html in browser
```

## Migration Validation

The test suite validates that the LLM migration was successful:

1. **All old patterns work** with the new system
2. **Same response structures** are maintained
3. **Error handling** behaves consistently  
4. **Configuration compatibility** is preserved
5. **Performance characteristics** are acceptable

Run migration tests specifically:
```bash
python tests/run_tests.py --migration
```

## Troubleshooting

### Common Issues

**ImportError for llm_simple module:**
```bash
# Ensure you're in the project root
cd /path/to/glossary
uv run pytest tests/
```

**Integration tests skipped:**
```bash
# Set API keys
export OPENAI_API_KEY="your-key"
uv run pytest tests/integration/
```

**Coverage command not found:**
```bash
uv add --dev pytest-cov
```

### Getting Help
1. Check the test output for specific error messages
2. Run with `--verbose` for more details  
3. Check `conftest.py` for available fixtures
4. Verify API keys for integration tests

## Test Philosophy

These tests follow the **testing pyramid** approach:
- **Many unit tests** (fast, isolated, comprehensive)
- **Fewer integration tests** (slower, realistic, end-to-end)
- **Focused migration tests** (compatibility, regression prevention)

The goal is to ensure the LLM migration maintains functionality while providing a foundation for future development.