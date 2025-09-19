"""Utility helpers for the simplified mining module.

The functions in this module avoid tight coupling with the rest of the system
and focus on data transformation, validation, and convenience helpers for the
mining workflow.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict
from urllib.parse import urlparse

from .config import MiningConfig, OutputConfig


ConceptDefinition = Dict[str, Any]
logger = logging.getLogger("generate_glossary.mining")


class MetricEntry(TypedDict):
    success: int
    failure: int
    duration_sum: float


def _metrics_factory() -> MetricEntry:
    return {"success": 0, "failure": 0, "duration_sum": 0.0}


MINING_METRICS: defaultdict[str, MetricEntry] = defaultdict(_metrics_factory)
ACADEMIC_KEYWORDS = {
    "research",
    "paper",
    "study",
    "university",
    "conference",
    "journal",
    "curriculum",
    "course",
    "laboratory",
    "analysis",
    "methodology",
    "experiments",
}


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


def validate_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_academic_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme or "https"
    hostname = (parsed.netloc or "").lower()
    path = parsed.path.rstrip("/")
    normalized = f"{scheme}://{hostname}{path}"
    if parsed.query:
        normalized = f"{normalized}?{parsed.query}"
    return normalized


def extract_domain_info(url: str) -> Dict[str, str]:
    parsed = urlparse(url)
    hostname = parsed.netloc.lower()
    parts = hostname.split(".")
    domain = parts[-2] if len(parts) >= 2 else hostname
    tld = parts[-1] if parts else ""
    subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
    return {
        "hostname": hostname,
        "domain": domain,
        "tld": tld,
        "subdomain": subdomain,
    }


def is_academic_domain(url: str) -> bool:
    info = extract_domain_info(url)
    academic_suffixes = {"edu", "ac", "org", "gov"}
    if any(
        info["hostname"].endswith(f".{suffix}") or info["tld"] == suffix
        for suffix in academic_suffixes
    ):
        return True
    return any(
        segment in info["hostname"]
        for segment in ("university", "college", "institute")
    )


def filter_academic_urls(urls: Sequence[str]) -> List[str]:
    filtered: List[str] = []
    for url in urls:
        if not url:
            continue
        if validate_url(url) and is_academic_domain(url):
            filtered.append(normalize_academic_url(url))
    return filtered


# ---------------------------------------------------------------------------
# Result processing and formatting
# ---------------------------------------------------------------------------


def format_concept_results(
    results: Iterable[Dict[str, Any]],
) -> List[ConceptDefinition]:
    formatted: List[ConceptDefinition] = []
    for result in results:
        concept = result.get("concept") or result.get("term") or "unknown"
        formatted.append(
            {
                "concept": concept,
                "content": result.get("content") or result.get("summary"),
                "url": result.get("url"),
                "metadata": result.get("metadata", {}),
                "score": calculate_result_quality_score(result),
            }
        )
    return formatted


def aggregate_mining_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "results": [],
            "statistics": {
                "total_results": 0,
                "unique_urls": 0,
                "avg_quality_score": 0.0,
            },
        }

    deduped = deduplicate_results(results)
    formatted = format_concept_results(deduped)
    scores = [item["score"] for item in formatted]

    return {
        "results": formatted,
        "statistics": {
            "total_results": len(formatted),
            "unique_urls": len(
                {item.get("url") for item in formatted if item.get("url")}
            ),
            "avg_quality_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        },
    }


def deduplicate_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for result in results:
        url = result.get("url")
        content_hash = result.get("content_hash")
        dedupe_key = url or content_hash
        if not dedupe_key:
            dedupe_key = json.dumps(result, sort_keys=True)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(result)
    return deduped


def calculate_result_quality_score(result: Dict[str, Any]) -> float:
    quality = 0.0
    content = result.get("content") or result.get("summary") or ""
    if not content:
        return quality

    length_bonus = min(len(content) / 1000.0, 1.0)
    quality += length_bonus * 0.4

    keyword_matches = sum(
        1 for keyword in ACADEMIC_KEYWORDS if keyword in content.lower()
    )
    quality += min(keyword_matches / 5.0, 1.0) * 0.4

    if result.get("url") and is_academic_domain(result["url"]):
        quality += 0.2

    return round(min(quality, 1.0), 4)


# ---------------------------------------------------------------------------
# Error handling and logging
# ---------------------------------------------------------------------------


def setup_mining_logger(config: Dict[str, Any]) -> logging.Logger:
    # Logging configuration is managed centrally via ``configure_logging``.
    return logging.getLogger("generate_glossary.mining")


def log_mining_operation(operation: str, details: Dict[str, Any]) -> None:
    logger.info("%s | details=%s", operation, json.dumps(details, sort_keys=True))


def handle_firecrawl_error(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    logger.error(
        "Firecrawl error during %s: %s", context.get("operation", "unknown"), error
    )
    return create_error_report(error, context.get("operation", "unknown"), context)


def create_error_report(
    error: Exception, operation: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "operation": operation,
        "error_type": error.__class__.__name__,
        "message": str(error),
        "context": context,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# File / data management
# ---------------------------------------------------------------------------


def load_terms_from_file(file_path: str) -> List[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Terms file not found: {file_path}")

    terms: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term)

    if not terms:
        raise ValueError("Terms file is empty or contains only comments")
    return terms


def create_output_directory(output_path: str) -> str:
    path = Path(output_path)
    directory = path if path.is_dir() else path.parent
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def generate_output_filename(base_path: str, format: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = Path(base_path)
    if base.is_dir():
        return str(base / f"mining_results_{timestamp}.{format}")
    return str(base)


def save_mining_results(
    results: Sequence[Dict[str, Any]], output_path: str, config: OutputConfig
) -> str:
    create_output_directory(output_path)
    resolved_path = generate_output_filename(output_path, config.format)
    path = Path(resolved_path)

    processed_results = _apply_source_policy(results, config.include_source_urls)

    if config.format == "json":
        payload = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "include_source_urls": config.include_source_urls,
                "save_metadata": config.save_metadata,
            },
            "results": processed_results,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    elif config.format == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for entry in processed_results:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
    elif config.format == "csv":
        keys = sorted({key for item in processed_results for key in item.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for entry in processed_results:
                writer.writerow(entry)
    else:
        raise ValueError(f"Unsupported output format: {config.format}")

    logger.info("Saved mining results to %s", path)
    return str(path)


def _apply_source_policy(
    results: Sequence[Dict[str, Any]],
    include_source_urls: bool,
) -> List[Dict[str, Any]]:
    if include_source_urls:
        return [dict(entry) for entry in results]

    sanitized: List[Dict[str, Any]] = []
    for entry in results:
        redacted = dict(entry)
        for key in ("url", "source_url", "sourceUrl", "sourceURL"):
            redacted.pop(key, None)

        metadata = redacted.get("metadata")
        if isinstance(metadata, dict):
            filtered_metadata = {
                key: value
                for key, value in metadata.items()
                if not _metadata_contains_url(key, value)
            }
            redacted["metadata"] = filtered_metadata

        sanitized.append(redacted)
    return sanitized


def _metadata_contains_url(key: str, value: Any) -> bool:
    key_lower = key.lower()
    if "url" in key_lower or "link" in key_lower:
        return True
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return True
    return False


# ---------------------------------------------------------------------------
# Performance and monitoring utilities
# ---------------------------------------------------------------------------


def track_mining_metrics(operation: str, duration: float, success: bool) -> None:
    entry = MINING_METRICS[operation]
    if success:
        entry["success"] += 1
    else:
        entry["failure"] += 1
    entry["duration_sum"] += float(duration)


def calculate_processing_stats(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "avg_score": 0.0,
            "median_score": 0.0,
            "domains": {},
        }

    scores = [calculate_result_quality_score(result) for result in results]
    domain_counts = Counter(
        extract_domain_info(result.get("url", ""))["domain"]
        for result in results
        if result.get("url")
    )
    return {
        "count": len(results),
        "avg_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "median_score": round(statistics.median(scores), 4) if scores else 0.0,
        "domains": dict(domain_counts),
    }


def estimate_remaining_time(completed: int, total: int, elapsed: float) -> float:
    if completed == 0:
        return math.inf
    rate = completed / max(elapsed, 1e-6)
    remaining = total - completed
    return remaining / rate if rate > 0 else math.inf


def format_progress_message(completed: int, total: int, elapsed: float) -> str:
    remaining_seconds = estimate_remaining_time(completed, total, elapsed)
    if math.isinf(remaining_seconds):
        remaining_text = "unknown"
    else:
        remaining_text = time.strftime(
            "%H:%M:%S", time.gmtime(max(remaining_seconds, 0))
        )
    return f"Processed {completed}/{total} concepts in {elapsed:.1f}s (ETA {remaining_text})"


# ---------------------------------------------------------------------------
# Academic content processing
# ---------------------------------------------------------------------------


def extract_academic_keywords(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    counter = Counter(words)
    keywords = [
        word for word, _ in counter.most_common(25) if word in ACADEMIC_KEYWORDS
    ]
    return keywords


def classify_content_type(content: str) -> str:
    lowered = content.lower()
    if "conference" in lowered or "proceedings" in lowered:
        return "conference"
    if "syllabus" in lowered or "curriculum" in lowered:
        return "course"
    if "journal" in lowered or "doi" in lowered:
        return "journal"
    if "lecture" in lowered or "notes" in lowered:
        return "lecture_notes"
    return "article"


def extract_citation_info(content: str) -> Dict[str, Optional[str]]:
    doi_match = re.search(
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", content, flags=re.IGNORECASE
    )
    year_match = re.search(r"(19|20)\d{2}", content)
    return {
        "doi": doi_match.group(0) if doi_match else None,
        "year": year_match.group(0) if year_match else None,
    }


def score_academic_relevance(content: str, concept: str) -> float:
    lowered = content.lower()
    concept_words = [part for part in concept.lower().split() if part.isalpha()]
    if not content or not concept_words:
        return 0.0

    keyword_score = sum(lowered.count(word) for word in concept_words)
    academic_score = sum(lowered.count(keyword) for keyword in ACADEMIC_KEYWORDS)
    total = keyword_score + academic_score
    if total == 0:
        return 0.0
    normalized = min(total / (len(content) / 250), 1.0)
    return round(normalized, 4)


# ---------------------------------------------------------------------------
# Firecrawl helpers
# ---------------------------------------------------------------------------


def prepare_firecrawl_params(
    config: MiningConfig, *, use_snake_case: bool = False, **kwargs: Any
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "maxAge": config.max_age,
        "maxPages": config.max_pages,
        "timeout": config.request_timeout,
        "retry": {
            "attempts": config.retry_attempts,
            "delay": config.retry_delay,
        },
    }
    params.update(kwargs)

    if not use_snake_case:
        return params

    return _convert_keys_to_snake(params)


def _convert_keys_to_snake(payload: Any) -> Any:
    if isinstance(payload, dict):
        converted: Dict[str, Any] = {}
        for key, value in payload.items():
            converted[_to_snake(key)] = _convert_keys_to_snake(value)
        return converted
    if isinstance(payload, list):
        return [_convert_keys_to_snake(item) for item in payload]
    return payload


def _to_snake(name: str) -> str:
    if not name:
        return name
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return snake


def prioritize_urls_by_domain(
    urls: Sequence[str], preferred_domains: Sequence[str]
) -> List[str]:
    if not urls:
        return []
    if not preferred_domains:
        return list(dict.fromkeys(urls))

    preferred = [domain.lower() for domain in preferred_domains if domain]

    def sort_key(item: tuple[int, str]) -> tuple[int, int]:
        index, url = item
        info = extract_domain_info(url)
        hostname = info["hostname"]
        tld = info["tld"]
        for rank, domain in enumerate(preferred):
            if hostname.endswith(f".{domain}") or hostname == domain:
                return rank, index
            if info["domain"] == domain or tld == domain:
                return rank, index
        return len(preferred), index

    ordered = sorted(enumerate(urls), key=sort_key)

    result: List[str] = []
    seen: set[str] = set()
    for _, url in ordered:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def handle_batch_scrape_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not response:
        return []

    if isinstance(response, list):
        records = response
    else:
        records = response.get("results") or response.get("data") or []

    processed: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        processed.append(
            {
                "url": record.get("url"),
                "content": record.get("content") or record.get("markdown"),
                "summary": record.get("summary"),
                "metadata": record.get("metadata", {}),
            }
        )
    return processed


def monitor_queue_status(firecrawl_client: Any, job_id: str) -> Dict[str, Any]:
    if hasattr(firecrawl_client, "queue_status"):
        return firecrawl_client.queue_status(job_id)
    if hasattr(firecrawl_client, "get_job_status"):
        return firecrawl_client.get_job_status(job_id)
    raise RuntimeError("Firecrawl client does not support queue monitoring")


def setup_webhook_config(
    webhook_url: Optional[str], events: Optional[Sequence[str]] = None
) -> Optional[Dict[str, Any]]:
    if not webhook_url:
        return None
    return {
        "url": webhook_url,
        "events": list(events or ["completed", "failed"]),
    }
