"""Minimal helpers that wrap Firecrawl SDK calls directly."""

from __future__ import annotations

import logging
from typing import Dict, List

from ..config import FirecrawlConfig

logger = logging.getLogger(__name__)


def create_client(config: FirecrawlConfig):
    """Create a Firecrawl client using the provided configuration."""

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


def search_academic_concepts(client, query: str, limit: int = 5) -> List[Dict]:
    """Run a Firecrawl search with lightweight academic shaping."""

    try:
        results = client.search(
            query=f"{query} (definition OR explanation OR academic)",
            limit=limit,
            categories=["research", "academic", "education"],
        )
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Search failed for query '%s': %s", query, exc)
        return []

    if isinstance(results, dict):
        return results.get("web", []) or []
    return results or []


def batch_scrape_urls(client, urls: List[str], use_summary: bool = True) -> List[Dict]:
    """Scrape multiple URLs using Firecrawl's native batch endpoint."""

    formats = ["markdown"]
    if use_summary:
        formats.append("summary")

    try:
        job = client.batch_scrape(
            urls=urls,
            formats=formats,
            poll_interval=2,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Batch scrape failed for %d URLs: %s", len(urls), exc)
        return []

    data = getattr(job, "data", job)
    return data or []


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
