"""Minimal helpers that wrap Firecrawl SDK calls directly."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..config import FirecrawlConfig

logger = logging.getLogger(__name__)


def create_client(config: FirecrawlConfig):
    """Create a Firecrawl client using the provided configuration.

    The ``timeout`` and ``max_retries`` fields in ``FirecrawlConfig`` are not
    applied at construction time; pass them via per-call options when supported
    by the Firecrawl SDK instead.
    """

    try:
        from firecrawl import Firecrawl
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Firecrawl SDK is not installed. Install `firecrawl` to use the mining module."
        ) from exc

    kwargs = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return Firecrawl(**kwargs)


def create_async_client(config: FirecrawlConfig):
    """Create an AsyncFirecrawl client using the provided configuration."""

    try:
        from firecrawl import AsyncFirecrawl
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Firecrawl SDK is not installed. Install `firecrawl` to use the mining module."
        ) from exc

    kwargs = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return AsyncFirecrawl(**kwargs)


def search_academic_concepts(
    client,
    query: str,
    limit: int = 5,
    scrape_options: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run a Firecrawl search; callers should shape the query upstream."""

    options = scrape_options or {"formats": ["markdown"]}

    try:
        return client.search(
            query=query,
            limit=limit,
            scrape_options=options,
        )
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Search failed for query '%s': %s", query, exc)
        return None


def batch_scrape_urls(
    client,
    urls: List[str],
    use_summary: bool = True,
    poll_interval: int = 2,
    wait_timeout: int = 180,
) -> List[Dict]:
    """Return the list emitted by Firecrawl's ``batch_scrape`` for ``urls``."""

    formats = ["markdown"]
    if use_summary:
        formats.append("summary")

    try:
        results = client.batch_scrape(
            urls=urls,
            formats=formats,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
        )
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Batch scrape failed for %d URLs: %s", len(urls), exc)
        return []
    return results or []


def scrape_single_url(client, url: str, use_summary: bool = True) -> Dict:
    """Scrape a single URL through Firecrawl's scrape helper."""

    formats = ["markdown"]
    if use_summary:
        formats.append("summary")

    try:
        result = client.scrape(url=url, formats=formats)
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Scrape failed for URL '%s': %s", url, exc)
        return {}

    return result or {}


def get_queue_status(client) -> Dict:
    """Return queue status when the Firecrawl client exposes it."""

    status_callable = getattr(client, "get_queue_status", None)
    if not callable(status_callable):
        return {"status": "queue_status_not_available"}

    try:
        status = status_callable()
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Queue status check failed: %s", exc)
        return {"status": "error", "error": str(exc)}

    return status or {}
