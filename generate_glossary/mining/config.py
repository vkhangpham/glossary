"""Configuration management for the simplified mining module.

This module keeps configuration handling lightweight and explicit.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

import yaml


CONFIG_FILE_NAME = "config.yml"
DEFAULT_CONFIG_PATH = Path(__file__).with_name(CONFIG_FILE_NAME)
_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


_CACHED_DEFAULT_CONFIG: Optional["MiningModuleConfig"] = None
_CACHED_FIRECRAWL_CLIENT: Optional[Any] = None
_CACHE_LOCK = RLock()


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
    max_concurrent_operations: int = (
        5  # Controls parallel scrapes when batch mode is disabled
    )
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


def _sub_env(val: Optional[str]) -> Optional[str]:
    """Substitute environment placeholder values of the form ``${VAR}``."""

    if val is None:
        return None
    stripped = str(val).strip()
    match = _ENV_PLACEHOLDER.match(stripped)
    if not match:
        return stripped

    env_name = match.group(1)
    resolved = os.environ.get(env_name)
    if resolved is None:
        return None
    return resolved.strip() or None


def _resolve_api_key(value: Optional[str]) -> str:
    substituted = _sub_env(value)

    env_override = os.environ.get("FIRECRAWL_API_KEY")
    resolved = (env_override or substituted or "").strip()

    if "${" in resolved:
        raise ConfigError(
            "Firecrawl API key contains an unresolved placeholder after substitution; check config.yml and environment variables."
        )

    if not resolved:
        raise ConfigError(
            "Firecrawl API key must be provided via config.yml or the FIRECRAWL_API_KEY environment variable."
        )

    return resolved


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
    """Load configuration from YAML file, applying environment overrides and placeholders."""

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw_config = _load_yaml_config(path)

    firecrawl_config = FirecrawlConfig(
        api_key=_resolve_api_key(raw_config.get("firecrawl", {}).get("api_key")),
        timeout=int(raw_config.get("firecrawl", {}).get("timeout", 30)),
        max_retries=int(raw_config.get("firecrawl", {}).get("max_retries", 3)),
        base_url=str(
            raw_config.get("firecrawl", {}).get("base_url", "https://api.firecrawl.dev")
        ),
    )

    mining_section = raw_config.get("mining", {})
    mining_config = MiningConfig(
        batch_size=int(mining_section.get("batch_size", 25)),
        max_concurrent_operations=int(
            mining_section.get("max_concurrent_operations", 5)
        ),
        max_urls_per_concept=int(mining_section.get("max_urls_per_concept", 3)),
        request_timeout=int(mining_section.get("request_timeout", 30)),
        retry_attempts=int(mining_section.get("retry_attempts", 3)),
        retry_delay=float(mining_section.get("retry_delay", 1.0)),
        max_age=int(mining_section.get("max_age", 172800000)),
        use_summary=bool(mining_section.get("use_summary", True)),
        use_batch_scrape=bool(mining_section.get("use_batch_scrape", True)),
        max_pages=int(mining_section.get("max_pages", 10)),
        search_categories=list(
            mining_section.get(
                "search_categories",
                ["research", "academic", "education"],
            )
        ),
        academic_domains=list(
            mining_section.get(
                "academic_domains",
                ["edu", "org", "gov"],
            )
        ),
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
        mining=mining_config,
        output=output_config,
        logging=logging_config,
    )
    validate_config(config)
    return config


def get_cached_config(force_reload: bool = False) -> MiningModuleConfig:
    """Return the cached default mining configuration, reloading if requested."""

    global _CACHED_DEFAULT_CONFIG
    with _CACHE_LOCK:
        if force_reload or _CACHED_DEFAULT_CONFIG is None:
            _CACHED_DEFAULT_CONFIG = load_config()
        return _CACHED_DEFAULT_CONFIG


def override_with_cli_args(
    config: MiningModuleConfig, args: argparse.Namespace
) -> MiningModuleConfig:
    """Override configuration values with CLI arguments."""

    def _maybe_get(name: str) -> Any:
        return getattr(args, name, None)

    batch_size = _maybe_get("batch_size")
    if batch_size is not None:
        config.mining.batch_size = int(batch_size)

    max_concurrent = _maybe_get("max_concurrent")
    if max_concurrent is not None:
        value = int(max_concurrent)
        if value <= 0:
            raise ConfigError("max_concurrent must be positive")
        config.mining.max_concurrent_operations = value

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
            config.mining.search_categories = [
                item.strip() for item in search_categories.split(",") if item.strip()
            ]
        else:
            config.mining.search_categories = list(search_categories)

    academic_domains = _maybe_get("academic_domains")
    if academic_domains:
        if isinstance(academic_domains, str):
            config.mining.academic_domains = [
                item.strip() for item in academic_domains.split(",") if item.strip()
            ]
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


def get_firecrawl_client(
    config: Optional[FirecrawlConfig] = None,
    *,
    force_refresh: bool = False,
) -> Any:
    """Instantiate and return a Firecrawl client.

    When ``config`` is omitted, a cached default configuration is used. Setting
    ``force_refresh`` invalidates the cached client and reloads configuration
    from disk.
    """

    from .core.firecrawl_client import create_firecrawl_client

    try:
        if config is None:
            with _CACHE_LOCK:
                resolved_config = get_cached_config(
                    force_reload=force_refresh
                ).firecrawl

                global _CACHED_FIRECRAWL_CLIENT
                if not force_refresh and _CACHED_FIRECRAWL_CLIENT is not None:
                    return _CACHED_FIRECRAWL_CLIENT

                client = create_firecrawl_client(resolved_config)
                _CACHED_FIRECRAWL_CLIENT = client
                return client

        return create_firecrawl_client(config)
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise ConfigError(str(exc)) from exc


def reset_cached_firecrawl_client() -> None:
    """Clear cached Firecrawl client and configuration state."""

    global _CACHED_DEFAULT_CONFIG, _CACHED_FIRECRAWL_CLIENT
    with _CACHE_LOCK:
        _CACHED_FIRECRAWL_CLIENT = None
        _CACHED_DEFAULT_CONFIG = None


def get_firecrawl_api_key() -> str:
    """Return the resolved Firecrawl API key from configuration."""

    with _CACHE_LOCK:
        return get_cached_config().firecrawl.api_key


def configure_logging(logging_config: Dict[str, Any]) -> None:
    """Configure logging for the mining module in an idempotent manner."""

    level_name = str(logging_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = logging_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_path = logging_config.get("file")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(fmt)

    # Ensure a single stream handler managed by this module.
    mining_stream_handler = next(
        (
            handler
            for handler in root_logger.handlers
            if getattr(handler, "_mining_stream", False)
        ),
        None,
    )
    if mining_stream_handler is None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        stream_handler._mining_stream = True  # type: ignore[attr-defined]
        root_logger.addHandler(stream_handler)
    else:
        mining_stream_handler.setFormatter(formatter)
        mining_stream_handler.setLevel(level)

    if file_path:
        normalized_path = str(Path(file_path).resolve())
        existing_file_handler = next(
            (
                handler
                for handler in root_logger.handlers
                if isinstance(handler, logging.FileHandler)
                and getattr(handler, "_mining_path", None) == normalized_path
            ),
            None,
        )
        if existing_file_handler is None:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            file_handler._mining_path = normalized_path  # type: ignore[attr-defined]
            root_logger.addHandler(file_handler)
        else:
            existing_file_handler.setFormatter(formatter)
            existing_file_handler.setLevel(level)
