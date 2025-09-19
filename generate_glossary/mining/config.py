"""Configuration management for the simplified mining module.

This module keeps configuration handling lightweight and explicit.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


CONFIG_FILE_NAME = "config.yml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(CONFIG_FILE_NAME)


class ConfigError(RuntimeError):
    """Raised when configuration loading or validation fails."""


@dataclass
class FirecrawlConfig:
    api_key: str
    timeout: int = 30
    max_retries: int = 3
    base_url: str = "https://api.firecrawl.dev"


@dataclass
class MiningConfig:
    batch_size: int = 25
    max_concurrent_operations: int = 5
    max_urls_per_concept: int = 3
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_age: int = 172800000
    use_summary: bool = True
    use_batch_scrape: bool = True
    max_pages: int = 10
    search_categories: list[str] = field(
        default_factory=lambda: ["research", "academic", "education"],
    )
    academic_domains: list[str] = field(
        default_factory=lambda: ["edu", "org", "gov"],
    )


@dataclass
class OutputConfig:
    format: str = "json"
    save_metadata: bool = True
    include_source_urls: bool = True


@dataclass
class MiningModuleConfig:
    firecrawl: FirecrawlConfig
    mining: MiningConfig
    output: OutputConfig
    logging: Dict[str, Any]


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            return {}
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Failed to parse configuration file: {config_path}") from exc

    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ConfigError("Configuration root must be a mapping")

    return data


def _resolve_api_key(value: Optional[str]) -> str:
    if os.environ.get("FIRECRAWL_API_KEY"):
        return os.environ["FIRECRAWL_API_KEY"].strip()
    if value:
        return str(value).strip()
    raise ConfigError("Firecrawl API key must be set via config or FIRECRAWL_API_KEY env var")


def _prepare_logging_config(raw_logging: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    default_logging = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None,
    }
    if not raw_logging:
        return default_logging

    merged: Dict[str, Any] = {**default_logging, **raw_logging}
    return merged


def load_config(config_path: Optional[str] = None) -> MiningModuleConfig:
    """Load configuration from YAML file and environment variables."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw_config = _load_yaml_config(path)

    firecrawl_config = FirecrawlConfig(
        api_key=_resolve_api_key(raw_config.get("firecrawl", {}).get("api_key")),
        timeout=int(raw_config.get("firecrawl", {}).get("timeout", 30)),
        max_retries=int(raw_config.get("firecrawl", {}).get("max_retries", 3)),
        base_url=str(raw_config.get("firecrawl", {}).get("base_url", "https://api.firecrawl.dev")),
    )

    mining_section = raw_config.get("mining", {})
    mining_config = MiningConfig(
        batch_size=int(mining_section.get("batch_size", 25)),
        max_concurrent_operations=int(mining_section.get("max_concurrent_operations", 5)),
        max_urls_per_concept=int(mining_section.get("max_urls_per_concept", 3)),
        request_timeout=int(mining_section.get("request_timeout", 30)),
        retry_attempts=int(mining_section.get("retry_attempts", 3)),
        retry_delay=float(mining_section.get("retry_delay", 1.0)),
        max_age=int(mining_section.get("max_age", 172800000)),
        use_summary=bool(mining_section.get("use_summary", True)),
        use_batch_scrape=bool(mining_section.get("use_batch_scrape", True)),
        max_pages=int(mining_section.get("max_pages", 10)),
        search_categories=list(mining_section.get(
            "search_categories", ["research", "academic", "education"],
        )),
        academic_domains=list(mining_section.get(
            "academic_domains", ["edu", "org", "gov"],
        )),
    )

    output_section = raw_config.get("output", {})
    output_config = OutputConfig(
        format=str(output_section.get("format", "json")),
        save_metadata=bool(output_section.get("save_metadata", True)),
        include_source_urls=bool(output_section.get("include_source_urls", True)),
    )

    logging_config = _prepare_logging_config(raw_config.get("logging"))

    config = MiningModuleConfig(
        firecrawl=firecrawl_config,
        mining= mining_config,
        output=output_config,
        logging=logging_config,
    )
    validate_config(config)
    return config


def override_with_cli_args(config: MiningModuleConfig, args: argparse.Namespace) -> MiningModuleConfig:
    """Override configuration values with CLI arguments."""

    def _maybe_get(name: str) -> Any:
        return getattr(args, name, None)

    batch_size = _maybe_get("batch_size")
    if batch_size is not None:
        config.mining.batch_size = int(batch_size)

    max_concurrent = _maybe_get("max_concurrent")
    if max_concurrent is not None:
        config.mining.max_concurrent_operations = int(max_concurrent)

    max_urls_per_concept = _maybe_get("max_urls")
    if max_urls_per_concept is not None:
        config.mining.max_urls_per_concept = int(max_urls_per_concept)

    max_age = _maybe_get("max_age")
    if max_age is not None:
        config.mining.max_age = int(max_age)

    max_pages = _maybe_get("max_pages")
    if max_pages is not None:
        config.mining.max_pages = int(max_pages)

    use_batch = _maybe_get("use_batch")
    if use_batch is not None:
        config.mining.use_batch_scrape = bool(use_batch)

    use_summary = _maybe_get("use_summary")
    if use_summary is not None:
        config.mining.use_summary = bool(use_summary)

    timeout = _maybe_get("request_timeout")
    if timeout is not None:
        config.mining.request_timeout = int(timeout)

    retry_attempts = _maybe_get("retry_attempts")
    if retry_attempts is not None:
        config.mining.retry_attempts = int(retry_attempts)

    retry_delay = _maybe_get("retry_delay")
    if retry_delay is not None:
        config.mining.retry_delay = float(retry_delay)

    search_categories = _maybe_get("search_categories")
    if search_categories:
        if isinstance(search_categories, str):
            config.mining.search_categories = [item.strip() for item in search_categories.split(",") if item.strip()]
        else:
            config.mining.search_categories = list(search_categories)

    academic_domains = _maybe_get("academic_domains")
    if academic_domains:
        if isinstance(academic_domains, str):
            config.mining.academic_domains = [item.strip() for item in academic_domains.split(",") if item.strip()]
        else:
            config.mining.academic_domains = list(academic_domains)

    log_level = _maybe_get("log_level")
    if log_level:
        config.logging["level"] = str(log_level)

    output_format = _maybe_get("output_format")
    if output_format:
        config.output.format = str(output_format)

    save_metadata = _maybe_get("save_metadata")
    if save_metadata is not None:
        config.output.save_metadata = bool(save_metadata)

    include_source_urls = _maybe_get("include_source_urls")
    if include_source_urls is not None:
        config.output.include_source_urls = bool(include_source_urls)

    return config


def validate_config(config: MiningModuleConfig) -> None:
    """Validate configuration values."""

    if not config.firecrawl.api_key:
        raise ConfigError("Firecrawl API key is required")

    if config.mining.batch_size <= 0:
        raise ConfigError("batch_size must be positive")

    if config.mining.max_concurrent_operations <= 0:
        raise ConfigError("max_concurrent_operations must be positive")

    if config.mining.max_urls_per_concept <= 0:
        raise ConfigError("max_urls_per_concept must be positive")

    if config.mining.max_age < 0:
        raise ConfigError("max_age cannot be negative")

    if config.mining.max_pages <= 0:
        raise ConfigError("max_pages must be positive")

    if config.output.format not in {"json", "jsonl", "csv"}:
        raise ConfigError("output format must be one of: json, jsonl, csv")


def get_firecrawl_client(config: FirecrawlConfig):
    """Instantiate and return a Firecrawl client."""

    try:
        from firecrawl import Firecrawl
    except ImportError as exc:  # pragma: no cover - import guard
        raise ConfigError(
            "Firecrawl SDK is not installed. Install `firecrawl` to use the mining module."
        ) from exc

    client_kwargs = {
        "api_key": config.api_key,
    }
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    client = Firecrawl(**client_kwargs)
    if hasattr(client, "set_request_timeout"):
        client.set_request_timeout(config.timeout)
    if hasattr(client, "set_max_retries"):
        client.set_max_retries(config.max_retries)

    return client


def configure_logging(logging_config: Dict[str, Any]) -> None:
    """Configure logging for the mining module."""

    level = logging_config.get("level", "INFO")
    fmt = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path = logging_config.get("file")

    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO), format=fmt)

    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(file_handler)

