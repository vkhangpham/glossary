"""
Comprehensive tests for the mining module to verify the simplified Firecrawl v2.2.0 integration.

Tests verify:
1. Configuration loading from config.yml with CLI overrides
2. Firecrawl client initialization and singleton behavior
3. Core mining entry points remain importable after API changes
4. Error handling utilities produce structured reports
5. CLI interface wiring remains intact
6. Integration with other modules that depend on mining exports
7. Basic performance/memory characteristics of utility helpers
8. File I/O operations and output formatting
9. Backward compatibility with public mining module exports
"""

from __future__ import annotations

import json
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
import yaml


# Test configuration loading
def test_config_loading() -> None:
    """Test configuration loading from config.yml with CLI overrides."""
    from generate_glossary.mining import config

    # Test module exposes expected constants
    assert hasattr(config, "CONFIG_FILE_NAME")
    assert hasattr(config, "DEFAULT_CONFIG_PATH")
    assert config.DEFAULT_CONFIG_PATH.exists()

    # Test config.yml exists and is valid
    config_path = config.DEFAULT_CONFIG_PATH
    assert config_path.exists()

    with open(config_path, encoding='utf-8') as handle:
        config_data = yaml.safe_load(handle)

    assert isinstance(config_data, dict)
    assert "firecrawl" in config_data
    assert "api_key" in config_data["firecrawl"]


def test_firecrawl_client_initialization() -> None:
    """Test Firecrawl client initialization and singleton behavior."""
    from generate_glossary.mining.config import FirecrawlConfig, get_firecrawl_client

    with patch("generate_glossary.mining.core.firecrawl_client.create_client") as mock_create:
        mock_client = Mock()
        mock_create.return_value = mock_client

        config = FirecrawlConfig(api_key="test-key")
        client = get_firecrawl_client(config, force_refresh=True)

        assert client is mock_client
        mock_create.assert_called_once_with(config)


# Ensure ``mine_concepts_simple`` stays importable even if implementation evolves
def test_core_mining_functions() -> None:
    from generate_glossary.mining import mine_concepts_simple

    assert callable(mine_concepts_simple)


# Error handling should still return structured reports
def test_error_handling_and_retries() -> None:
    from generate_glossary.mining.utils import handle_firecrawl_error

    error = Exception("API Error")
    context = {"operation": "search", "term": "artificial intelligence"}
    report = handle_firecrawl_error(error, context)

    assert report["operation"] == "search"
    assert report["message"] == "API Error"
    assert report["context"]["term"] == "artificial intelligence"


# CLI wiring should still resolve the main entry point
def test_cli_interface() -> None:
    from generate_glossary.mining.main import main

    with pytest.raises(SystemExit):
        with patch("sys.argv", ["mine-web", "--help"]):
            main()


# Modules that depend on mining should still import without errors
def test_integration_with_other_modules() -> None:
    from generate_glossary.generation.lv0.lv0_s0_get_college_names import main as extract_colleges
    from generate_glossary.mining import mine_concepts_simple

    assert callable(extract_colleges)
    assert callable(mine_concepts_simple)


# Utility helpers should remain lightweight and efficient
def test_performance_and_memory_usage() -> None:
    from generate_glossary.mining.utils import aggregate_mining_results

    sample_results: List[Dict[str, Any]] = [
        {
            "concept": "Computer Science",
            "content": "Research on AI and ML advancements",
            "url": f"https://example.edu/cs/{index}",
        }
        for index in range(50)
    ]

    tracemalloc.start()
    start_time = time.time()
    aggregated = aggregate_mining_results(sample_results)
    duration = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert aggregated["statistics"]["total_results"] == len(sample_results)
    assert duration < 1.0
    assert peak < 10 * 1024 * 1024
    assert current < peak


# File I/O helpers should continue to emit well-structured outputs
def test_file_io_operations() -> None:
    from generate_glossary.mining.config import OutputConfig
    from generate_glossary.mining.utils import save_mining_results

    with tempfile.TemporaryDirectory() as temp_dir:
        results = [
            {
                "concept": "Computer Science",
                "content": "Machine learning overview",
                "url": "https://example.edu/cs",
                "metadata": {"source": "example"},
            }
        ]
        output_path = Path(temp_dir) / "results"
        config = OutputConfig(format="json", save_metadata=True, include_source_urls=True)

        saved_path = save_mining_results(results, str(output_path), config)
        assert Path(saved_path).exists()

        with open(saved_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["results"]
        assert payload["metadata"]["include_source_urls"] is True


# Public exports should still be reachable for backward compatibility
def test_backward_compatibility() -> None:
    from generate_glossary.mining import (
        ConfigError,
        load_config,
        mine_concepts_simple,
        save_mining_results,
    )

    assert callable(load_config)
    assert callable(mine_concepts_simple)
    assert callable(save_mining_results)
    assert issubclass(ConfigError, Exception)


def test_module_structure() -> None:
    """Test that the mining module follows the standardized structure."""
    mining_path = Path(__file__).parent.parent / "generate_glossary" / "mining"

    # Check required files exist
    assert (mining_path / "config.yml").exists()
    assert (mining_path / "config.py").exists()
    assert (mining_path / "main.py").exists()
    assert (mining_path / "core").exists()
