"""Lightweight helpers for interacting with the Firecrawl v2.2.0 SDK.

This module keeps Firecrawl access focused on the essentials: creating
configured clients and exposing simple wrappers around the core SDK
operations used by the mining pipeline.
"""

from __future__ import annotations

import logging
import inspect
from typing import Any, Dict, Iterable, List, Optional

from ..config import FirecrawlConfig

logger = logging.getLogger(__name__)


def create_firecrawl_client(config: FirecrawlConfig):
    """Instantiate a Firecrawl client using the provided configuration."""

    try:
        from firecrawl import Firecrawl
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Firecrawl SDK is not installed. Install `firecrawl` to use the mining module."
        ) from exc

    client_kwargs: Dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    client = Firecrawl(**client_kwargs)
    _apply_client_config(client, config)
    return client


def _apply_client_config(client: Any, config: FirecrawlConfig) -> None:
    """Apply timeout and retry settings when the SDK exposes helpers."""

    if hasattr(client, "set_request_timeout"):
        try:
            client.set_request_timeout(config.timeout)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to set request timeout: %s", exc)
    elif hasattr(client, "timeout"):
        try:
            client.timeout = config.timeout
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to assign timeout attribute: %s", exc)

    if hasattr(client, "set_max_retries"):
        try:
            client.set_max_retries(config.max_retries)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to set max retries: %s", exc)
    elif hasattr(client, "max_retries"):
        try:
            client.max_retries = config.max_retries
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to assign max_retries attribute: %s", exc)


def search_concepts(
    client: Any,
    query: str,
    limit: int = 5,
    categories: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Search Firecrawl for academic concepts, returning an empty list on failure."""

    composed_query = f"{query} (definition OR explanation OR academic)"
    search_kwargs: Dict[str, Any] = {"query": composed_query, "limit": limit}
    if categories is not None:
        search_kwargs["categories"] = list(categories)
    else:
        search_kwargs["categories"] = ["research", "academic", "education"]
    search_kwargs.update(kwargs)

    try:
        results = client.search(**search_kwargs)
        return results or []
    except TypeError as exc:
        logger.error(
            "Search failed for query '%s' due to argument mismatch: %s", query, exc
        )
        fallback_kwargs = {key: search_kwargs[key] for key in ("query", "limit")}
        try:
            results = client.search(**fallback_kwargs)
            return results or []
        except Exception as fallback_exc:  # pragma: no cover - depends on SDK
            logger.error(
                "Fallback search failed for query '%s': %s", query, fallback_exc
            )
            return []
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Search failed for query '%s': %s", query, exc)
        return []


def scrape_url(
    client: Any,
    url: str,
    formats: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Scrape a single URL, returning an empty mapping on failure."""

    scrape_kwargs: Dict[str, Any] = {"url": url, "formats": list(formats or ["markdown"])}
    scrape_kwargs.update(kwargs)

    try:
        result = client.scrape(**scrape_kwargs)
        return result or {}
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Scrape failed for URL '%s': %s", url, exc)
        return {}


def batch_scrape_urls(
    client: Any,
    urls: Iterable[str],
    formats: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Batch scrape URLs, returning an empty list on failure."""

    batch_kwargs: Dict[str, Any] = {
        "urls": list(urls),
        "formats": list(formats or ["markdown"]),
    }
    batch_kwargs.update(kwargs)

    try:
        results = client.batch_scrape(**batch_kwargs)
        return results or []
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Batch scrape failed for %d URLs: %s", len(batch_kwargs["urls"]), exc)
        return []


def map_website(client: Any, url: str, **kwargs: Any) -> List[str]:
    """Discover links from a website using Firecrawl's map API."""

    try:
        result = client.map(url=url, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Map failed for URL '%s': %s", url, exc)
        return []

    if isinstance(result, dict):
        links = result.get("links") or result.get("data")
        if isinstance(links, list):
            return links
        return []
    if isinstance(result, list):
        return result
    return []


def get_queue_status(client: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Retrieve queue status if the SDK exposes the helper; otherwise return an empty dict."""

    status_callable = getattr(client, "get_queue_status", None)
    if not callable(status_callable):
        return {}

    try:
        status = status_callable(*args, **kwargs)
        return status or {}
    except Exception as exc:  # pragma: no cover - depends on SDK
        logger.error("Queue status check failed: %s", exc)
        return {}


class AsyncFirecrawlClient:
    """Async wrapper that surfaces the same helpers using AsyncFirecrawl."""

    def __init__(self, config: FirecrawlConfig):
        try:
            from firecrawl import AsyncFirecrawl
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Firecrawl SDK is not installed. Install `firecrawl` to use the mining module."
            ) from exc

        client_kwargs: Dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = AsyncFirecrawl(**client_kwargs)
        _apply_client_config(self.client, config)

    async def search_concepts(
        self,
        query: str,
        limit: int = 5,
        categories: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        composed_query = f"{query} (definition OR explanation OR academic)"
        search_kwargs: Dict[str, Any] = {"query": composed_query, "limit": limit}
        if categories is not None:
            search_kwargs["categories"] = list(categories)
        else:
            search_kwargs["categories"] = ["research", "academic", "education"]
        search_kwargs.update(kwargs)

        try:
            results = await self.client.search(**search_kwargs)
            return results or []
        except TypeError as exc:
            logger.error(
                "Async search failed for query '%s' due to argument mismatch: %s",
                query,
                exc,
            )
            fallback_kwargs = {key: search_kwargs[key] for key in ("query", "limit")}
            try:
                results = await self.client.search(**fallback_kwargs)
                return results or []
            except Exception as fallback_exc:  # pragma: no cover - depends on SDK
                logger.error(
                    "Async fallback search failed for query '%s': %s",
                    query,
                    fallback_exc,
                )
                return []
        except Exception as exc:  # pragma: no cover - depends on SDK
            logger.error("Async search failed for query '%s': %s", query, exc)
            return []

    async def scrape_url(
        self,
        url: str,
        formats: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        scrape_kwargs: Dict[str, Any] = {"url": url, "formats": list(formats or ["markdown"])}
        scrape_kwargs.update(kwargs)

        try:
            result = await self.client.scrape(**scrape_kwargs)
            return result or {}
        except Exception as exc:  # pragma: no cover - depends on SDK
            logger.error("Async scrape failed for URL '%s': %s", url, exc)
            return {}

    async def batch_scrape_urls(
        self,
        urls: Iterable[str],
        formats: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        batch_kwargs: Dict[str, Any] = {
            "urls": list(urls),
            "formats": list(formats or ["markdown"]),
        }
        batch_kwargs.update(kwargs)

        try:
            results = await self.client.batch_scrape(**batch_kwargs)
            return results or []
        except Exception as exc:  # pragma: no cover - depends on SDK
            logger.error(
                "Async batch scrape failed for %d URLs: %s", len(batch_kwargs["urls"]), exc
            )
            return []

    async def map_website(self, url: str, **kwargs: Any) -> List[str]:
        try:
            result = await self.client.map(url=url, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on SDK
            logger.error("Async map failed for URL '%s': %s", url, exc)
            return []

        if isinstance(result, dict):
            links = result.get("links") or result.get("data")
            if isinstance(links, list):
                return links
            return []
        if isinstance(result, list):
            return result
        return []

    async def get_queue_status(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        status_callable = getattr(self.client, "get_queue_status", None)
        if not callable(status_callable):
            return {}
        try:
            result = status_callable(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result or {}
        except Exception as exc:  # pragma: no cover - depends on SDK
            logger.error("Async queue status check failed: %s", exc)
            return {}

    async def close(self) -> None:
        """Close the underlying async client when supported."""

        close_callable = getattr(self.client, "close", None)
        if not callable(close_callable):
            return

        try:
            result = close_callable()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # pragma: no cover - depends on SDK
            logger.debug("Async Firecrawl client close failed: %s", exc)
