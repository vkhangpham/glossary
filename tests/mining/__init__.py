"""
Mining module test package initialization.

This package contains comprehensive tests for the Firecrawl v2.2.0 mining module integration,
including unit tests, integration tests, performance tests, and compatibility validation.

Test Categories:
- Unit tests: Test individual components without external dependencies
- Integration tests: Test actual Firecrawl API interactions (requires API key)
- Performance tests: Validate v2.2.0 performance improvements
- Compatibility tests: Ensure backward compatibility and SDK validation

Fixtures and utilities are available through conftest.py for shared test functionality.
"""

import logging
import sys
from pathlib import Path

# Configure test logging to be less verbose by default
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Common test utilities
def enable_debug_logging():
    """Enable debug logging for tests that need detailed output."""
    logging.getLogger().setLevel(logging.DEBUG)

def disable_logging():
    """Disable logging for tests that need clean output."""
    logging.getLogger().setLevel(logging.CRITICAL)

# Test data helpers
TEST_CONCEPTS = [
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "neural networks",
    "computer vision"
]

TEST_URLS = [
    "https://example.com/ai-research",
    "https://example.com/ml-paper.pdf",
    "https://example.com/cs-department"
]

# Version info for testing
TEST_VERSION = "2.2.0"
EXPECTED_V220_FEATURES = {
    "batch_scraping",
    "queue_monitoring",
    "fast_mapping",
    "pdf_page_limit",
    "webhooks",
    "structured_extraction"
}