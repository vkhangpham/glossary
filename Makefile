# Makefile for Firecrawl v2.2.0 Mining Module Test Execution and CI Integration
#
# This Makefile provides convenient commands for running different types of tests,
# managing test environments, and integrating with CI/CD pipelines.

# Configuration
PYTHON = python
UV = uv
PYTEST = $(UV) run pytest
PROJECT_ROOT = .
TEST_DIR = tests
MINING_TEST_DIR = tests/mining
COVERAGE_DIR = htmlcov
REPORTS_DIR = test-reports

# Test markers
UNIT_MARKER = unit
INTEGRATION_MARKER = integration
SLOW_MARKER = slow
SKIP_CI_MARKER = skip_ci

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Firecrawl v2.2.0 Mining Module Test Commands"
	@echo "============================================"
	@echo ""
	@echo "Test Execution:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Test Execution Targets

.PHONY: test-unit
test-unit: ## Run unit tests only (no API key required)
	@echo "Running unit tests..."
	$(PYTEST) -m "$(UNIT_MARKER)" $(TEST_DIR) -v \
		--tb=short \
		--durations=10 \
		--junitxml=$(REPORTS_DIR)/unit-tests.xml

.PHONY: test-integration
test-integration: check-api-key ## Run integration tests (requires FIRECRAWL_API_KEY)
	@echo "Running integration tests..."
	$(PYTEST) -m "$(INTEGRATION_MARKER)" $(TEST_DIR) -v \
		--tb=short \
		--durations=10 \
		--junitxml=$(REPORTS_DIR)/integration-tests.xml

.PHONY: test-performance
test-performance: ## Run performance tests (slow, optional)
	@echo "Running performance tests..."
	$(PYTEST) -m "$(SLOW_MARKER)" $(TEST_DIR) -v \
		--tb=short \
		--durations=20 \
		--junitxml=$(REPORTS_DIR)/performance-tests.xml

.PHONY: test-all
test-all: ## Run complete test suite
	@echo "Running complete test suite..."
	$(PYTEST) $(TEST_DIR) -v \
		--tb=short \
		--durations=20 \
		--junitxml=$(REPORTS_DIR)/all-tests.xml

.PHONY: test-compatibility
test-compatibility: ## Run Firecrawl SDK compatibility check
	@echo "Running Firecrawl SDK compatibility check..."
	$(PYTHON) $(TEST_DIR)/test_firecrawl_compatibility.py

.PHONY: test-mining
test-mining: ## Run only mining module tests
	@echo "Running mining module tests..."
	$(PYTEST) $(MINING_TEST_DIR) -v \
		--tb=short \
		--durations=10

# Test Coverage and Reporting

.PHONY: test-coverage
test-coverage: ## Run tests with coverage reporting
	@echo "Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) \
		--cov=generate_glossary.mining \
		--cov-report=html:$(COVERAGE_DIR) \
		--cov-report=xml:$(REPORTS_DIR)/coverage.xml \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		--junitxml=$(REPORTS_DIR)/coverage-tests.xml

.PHONY: test-report
test-report: test-coverage ## Generate HTML coverage report
	@echo "Coverage report generated at $(COVERAGE_DIR)/index.html"
	@if command -v open >/dev/null 2>&1; then \
		echo "Opening coverage report..."; \
		open $(COVERAGE_DIR)/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		echo "Opening coverage report..."; \
		xdg-open $(COVERAGE_DIR)/index.html; \
	else \
		echo "Open $(COVERAGE_DIR)/index.html in your browser"; \
	fi

.PHONY: test-ci
test-ci: setup-reports-dir ## CI-friendly test execution with proper exit codes
	@echo "Running CI test suite..."
	@set -e; \
	echo "=== Unit Tests ==="; \
	$(PYTEST) -m "$(UNIT_MARKER)" $(TEST_DIR) \
		--tb=short \
		--junitxml=$(REPORTS_DIR)/ci-unit-tests.xml || exit 1; \
	if [ -n "$$FIRECRAWL_API_KEY" ]; then \
		echo "=== Integration Tests ==="; \
		$(PYTEST) -m "$(INTEGRATION_MARKER)" $(TEST_DIR) \
			--tb=short \
			--junitxml=$(REPORTS_DIR)/ci-integration-tests.xml || exit 1; \
	else \
		echo "Skipping integration tests (no API key)"; \
	fi; \
	echo "=== Compatibility Check ==="; \
	$(PYTHON) $(TEST_DIR)/test_firecrawl_compatibility.py || echo "Compatibility check completed with warnings"; \
	echo "CI test suite completed successfully"

# Development Targets

.PHONY: test-watch
test-watch: ## Watch mode for continuous testing during development
	@echo "Starting test watch mode..."
	$(PYTEST) $(TEST_DIR) -f --tb=short

.PHONY: test-debug
test-debug: ## Run tests with debug output and logging
	@echo "Running tests with debug output..."
	$(PYTEST) $(TEST_DIR) -v -s \
		--tb=long \
		--log-cli-level=DEBUG \
		--capture=no

.PHONY: test-clean
test-clean: ## Clean test artifacts and cache
	@echo "Cleaning test artifacts..."
	rm -rf $(COVERAGE_DIR)
	rm -rf $(REPORTS_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "firecrawl_compatibility_report.json" -delete
	@echo "Test artifacts cleaned"

# Environment Setup

.PHONY: setup-test-env
setup-test-env: ## Set up test environment and dependencies
	@echo "Setting up test environment..."
	$(UV) sync --dev
	@echo "Installing additional test dependencies..."
	$(UV) add --dev pytest pytest-cov pytest-mock pytest-asyncio pytest-xdist
	@echo "Test environment setup complete"

.PHONY: check-api-key
check-api-key: ## Validate FIRECRAWL_API_KEY is configured
	@if [ -z "$$FIRECRAWL_API_KEY" ]; then \
		echo "ERROR: FIRECRAWL_API_KEY environment variable is not set"; \
		echo "Integration tests require a valid Firecrawl API key"; \
		echo "Set it with: export FIRECRAWL_API_KEY=your_api_key_here"; \
		exit 1; \
	else \
		echo "✓ FIRECRAWL_API_KEY is configured"; \
	fi

.PHONY: test-env-check
test-env-check: ## Validate test environment setup
	@echo "Checking test environment..."
	@$(UV) run python -c "import pytest; print(f'✓ pytest {pytest.__version__}')" || \
		(echo "✗ pytest not available" && exit 1)
	@$(UV) run python -c "import generate_glossary.mining; print('✓ mining module importable')" || \
		(echo "✗ mining module not importable" && exit 1)
	@if [ -n "$$FIRECRAWL_API_KEY" ]; then \
		echo "✓ FIRECRAWL_API_KEY configured"; \
	else \
		echo "! FIRECRAWL_API_KEY not set (integration tests will be skipped)"; \
	fi
	@echo "Test environment check complete"

.PHONY: setup-reports-dir
setup-reports-dir: ## Create reports directory
	@mkdir -p $(REPORTS_DIR)

# Advanced Test Targets

.PHONY: test-parallel
test-parallel: ## Run tests in parallel for faster execution
	@echo "Running tests in parallel..."
	$(PYTEST) $(TEST_DIR) -n auto \
		--tb=short \
		--durations=10 \
		--junitxml=$(REPORTS_DIR)/parallel-tests.xml

.PHONY: test-specific
test-specific: ## Run specific test (usage: make test-specific TEST=test_name)
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Specify test with TEST=test_name"; \
		echo "Example: make test-specific TEST=test_models.py::TestConceptDefinition"; \
		exit 1; \
	fi
	@echo "Running specific test: $(TEST)"
	$(PYTEST) $(TEST_DIR) -k "$(TEST)" -v --tb=short

.PHONY: test-failed
test-failed: ## Re-run only failed tests from last run
	@echo "Re-running failed tests..."
	$(PYTEST) --lf $(TEST_DIR) -v --tb=short

.PHONY: test-modified
test-modified: ## Run tests for modified files only
	@echo "Running tests for modified files..."
	@if command -v git >/dev/null 2>&1; then \
		MODIFIED_FILES=$$(git diff --name-only HEAD -- "*.py" | grep -E "(test_|_test\.py|tests/)" || true); \
		if [ -n "$$MODIFIED_FILES" ]; then \
			echo "Testing modified files: $$MODIFIED_FILES"; \
			$(PYTEST) $$MODIFIED_FILES -v --tb=short; \
		else \
			echo "No modified test files found"; \
		fi; \
	else \
		echo "Git not available, running all tests"; \
		$(MAKE) test-unit; \
	fi

# Benchmarking and Analysis

.PHONY: test-benchmark
test-benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(PYTEST) -m "$(SLOW_MARKER)" $(TEST_DIR) \
		--benchmark-only \
		--benchmark-sort=mean \
		--benchmark-html=$(REPORTS_DIR)/benchmark.html

.PHONY: test-profile
test-profile: ## Run tests with profiling
	@echo "Running tests with profiling..."
	$(PYTEST) $(TEST_DIR) --profile \
		--profile-svg \
		-m "not $(SLOW_MARKER)"

.PHONY: test-memory
test-memory: ## Run tests with memory profiling
	@echo "Running tests with memory profiling..."
	$(UV) run python -m pytest $(TEST_DIR) \
		--memray \
		--memray-bin-path=$(REPORTS_DIR) \
		-m "not $(SLOW_MARKER)"

# Docker Support

.PHONY: test-docker
test-docker: ## Run tests in Docker container
	@echo "Running tests in Docker..."
	docker build -t mining-tests -f Dockerfile.test .
	docker run --rm \
		-e FIRECRAWL_API_KEY \
		-v $(PWD)/$(REPORTS_DIR):/app/$(REPORTS_DIR) \
		mining-tests

# Quality Checks

.PHONY: test-lint
test-lint: ## Run linting on test files
	@echo "Linting test files..."
	$(UV) run ruff check $(TEST_DIR)
	$(UV) run ruff format --check $(TEST_DIR)

.PHONY: test-type-check
test-type-check: ## Run type checking on test files
	@echo "Type checking test files..."
	$(UV) run mypy $(TEST_DIR) --ignore-missing-imports

.PHONY: test-security
test-security: ## Run security checks on test files
	@echo "Running security checks..."
	$(UV) run bandit -r $(TEST_DIR) -ll

# Documentation

.PHONY: test-docs
test-docs: ## Test documentation examples
	@echo "Testing documentation examples..."
	$(UV) run python -m doctest $(TEST_DIR)/README.md

# GitHub Actions Integration

.PHONY: github-test-setup
github-test-setup: ## Setup for GitHub Actions
	@echo "Setting up for GitHub Actions..."
	$(UV) sync --dev
	mkdir -p $(REPORTS_DIR)

.PHONY: github-test-unit
github-test-unit: github-test-setup ## GitHub Actions unit test target
	$(PYTEST) -m "$(UNIT_MARKER)" $(TEST_DIR) \
		--junitxml=$(REPORTS_DIR)/junit-unit.xml \
		--cov=generate_glossary.mining \
		--cov-report=xml:$(REPORTS_DIR)/coverage-unit.xml

.PHONY: github-test-integration
github-test-integration: github-test-setup ## GitHub Actions integration test target
	@if [ -n "$$FIRECRAWL_API_KEY" ]; then \
		$(PYTEST) -m "$(INTEGRATION_MARKER)" $(TEST_DIR) \
			--junitxml=$(REPORTS_DIR)/junit-integration.xml \
			--cov=generate_glossary.mining \
			--cov-report=xml:$(REPORTS_DIR)/coverage-integration.xml; \
	else \
		echo "Skipping integration tests (no API key)"; \
		touch $(REPORTS_DIR)/junit-integration.xml; \
	fi

# Maintenance

.PHONY: test-deps-update
test-deps-update: ## Update test dependencies
	@echo "Updating test dependencies..."
	$(UV) add --dev pytest@latest pytest-cov@latest pytest-mock@latest

.PHONY: test-info
test-info: ## Show test configuration information
	@echo "Test Configuration Information"
	@echo "============================="
	@echo "Python: $$($(PYTHON) --version)"
	@echo "UV: $$($(UV) --version 2>/dev/null || echo 'not available')"
	@echo "Pytest: $$($(UV) run pytest --version | head -1)"
	@echo "Project Root: $(PROJECT_ROOT)"
	@echo "Test Directory: $(TEST_DIR)"
	@echo "Mining Tests: $(MINING_TEST_DIR)"
	@echo "Reports Directory: $(REPORTS_DIR)"
	@echo "Coverage Directory: $(COVERAGE_DIR)"
	@echo ""
	@echo "Available Test Markers:"
	@echo "  $(UNIT_MARKER) - Unit tests (fast, no external deps)"
	@echo "  $(INTEGRATION_MARKER) - Integration tests (require API key)"
	@echo "  $(SLOW_MARKER) - Slow tests (performance tests)"
	@echo "  $(SKIP_CI_MARKER) - Tests to skip in CI"
	@echo ""
	@if [ -n "$$FIRECRAWL_API_KEY" ]; then \
		echo "API Key: ✓ Configured"; \
	else \
		echo "API Key: ✗ Not configured"; \
	fi

# Aliases for convenience
.PHONY: test tests
test: test-unit  ## Alias for test-unit
tests: test-all  ## Alias for test-all

.PHONY: coverage cov
coverage: test-coverage  ## Alias for test-coverage
cov: test-coverage       ## Short alias for test-coverage

.PHONY: ci
ci: test-ci             ## Alias for test-ci

.PHONY: clean
clean: test-clean       ## Alias for test-clean