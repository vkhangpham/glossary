"""Simplified CLI entry point for the mining module."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import (
    ConfigError,
    MiningConfig,
    MiningModuleConfig,
    configure_logging,
    get_firecrawl_client,
    load_config,
    override_with_cli_args,
)
from .utils import (
    aggregate_mining_results,
    calculate_processing_stats,
    filter_academic_urls,
    format_progress_message,
    handle_batch_scrape_response,
    handle_firecrawl_error,
    load_terms_from_file,
    log_mining_operation,
    monitor_queue_status,
    prepare_firecrawl_params,
    prioritize_urls_by_domain,
    save_mining_results,
    score_academic_relevance,
    setup_webhook_config,
    track_mining_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mine-web",
        description="Mine web content for academic glossary terms using Firecrawl v2.2.0",
    )
    parser.add_argument(
        "terms_file", help="Path to a file containing terms to mine (one per line)"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output path (file or directory)"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, help="Number of terms per batch"
    )
    parser.add_argument(
        "--max-urls", type=int, help="Maximum URLs to fetch per concept"
    )
    parser.add_argument("--max-age", type=int, help="Cache duration in milliseconds")
    parser.add_argument(
        "--max-pages", type=int, help="Maximum pages to extract per document"
    )
    parser.add_argument(
        "-c",
        "--max-concurrent",
        dest="max_concurrent",
        type=int,
        help=(
            "Maximum concurrent Firecrawl scrape operations when batch mode is disabled (must be > 0)"
        ),
    )
    parser.add_argument(
        "--no-batch",
        dest="use_batch",
        action="store_false",
        help="Disable Firecrawl batch scraping",
        default=None,
    )
    parser.add_argument(
        "--batch",
        dest="use_batch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-summary",
        dest="use_summary",
        action="store_false",
        help="Disable summary generation",
        default=None,
    )
    parser.add_argument(
        "--summary",
        dest="use_summary",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--request-timeout", type=int, help="Request timeout in seconds"
    )
    parser.add_argument("--retry-attempts", type=int, help="Number of retry attempts")
    parser.add_argument(
        "--retry-delay", type=float, help="Delay between retries in seconds"
    )
    parser.add_argument(
        "--search-categories",
        help="Comma-separated list of Firecrawl search categories",
    )
    parser.add_argument(
        "--academic-domains",
        help="Comma-separated list of preferred academic domains",
    )
    parser.add_argument("--config", help="Path to alternate mining config.yml")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Silence non-error logs"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "csv"],
        dest="output_format",
        help="Output format",
    )
    parser.add_argument("--webhook-url", help="Webhook URL for Firecrawl notifications")
    parser.add_argument("--queue-job", help="Optional Firecrawl job ID to monitor")
    parser.add_argument("--save-metadata", dest="save_metadata", action="store_true")
    parser.add_argument(
        "--no-save-metadata", dest="save_metadata", action="store_false"
    )
    parser.add_argument(
        "--include-source-urls", dest="include_source_urls", action="store_true"
    )
    parser.add_argument(
        "--no-include-source-urls", dest="include_source_urls", action="store_false"
    )

    parser.set_defaults(save_metadata=None, include_source_urls=None)
    return parser


def apply_logging_overrides(
    config: MiningModuleConfig, args: argparse.Namespace
) -> None:
    if args.verbose:
        config.logging["level"] = "DEBUG"
    if args.quiet:
        config.logging["level"] = "ERROR"
    if args.log_level:
        config.logging["level"] = args.log_level


def mine_concepts_simple(
    terms: Iterable[str],
    firecrawl_client: Any,
    mining_config: MiningConfig,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    term_list = list(terms)
    start_time = time.time()
    mined_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    webhook_config = setup_webhook_config(webhook_url)
    total_terms = len(term_list) or 1

    snake_case_params = _infer_snake_case_params(firecrawl_client)
    supported_batch_kwargs = _collect_supported_kwargs(
        firecrawl_client, "batch_scrape"
    )
    supported_scrape_kwargs = _collect_supported_kwargs(firecrawl_client, "scrape")
    if supported_batch_kwargs and supported_scrape_kwargs:
        combined_supported_kwargs = supported_batch_kwargs | supported_scrape_kwargs
    else:
        combined_supported_kwargs = supported_batch_kwargs or supported_scrape_kwargs
    batch_size = max(1, mining_config.batch_size)
    batches = [
        term_list[i : i + batch_size] for i in range(0, len(term_list), batch_size)
    ] or [term_list]
    processed_terms = 0

    for batch_index, batch_terms in enumerate(batches, start=1):
        for batch_offset, term in enumerate(batch_terms, start=1):
            global_index = processed_terms + batch_offset
            operation_context = {"operation": "search", "term": term}
            op_start = time.time()
            try:
                search_results = _execute_search(
                    firecrawl_client,
                    query=f"{term} (definition OR explanation OR academic)",
                    limit=mining_config.max_urls_per_concept,
                    categories=mining_config.search_categories,
                )
                track_mining_metrics("search", time.time() - op_start, True)
            except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
                track_mining_metrics("search", time.time() - op_start, False)
                errors.append(handle_firecrawl_error(exc, operation_context))
                continue

            urls = [item.get("url") for item in search_results or [] if item.get("url")]
            academic_urls = prioritize_urls_by_domain(
                filter_academic_urls(urls) or urls,
                mining_config.academic_domains,
            )
            if not academic_urls:
                continue

            scrape_formats = ["markdown"]
            if mining_config.use_summary:
                scrape_formats.append("summary")

            extra_kwargs = {"formats": scrape_formats}
            if webhook_config:
                extra_kwargs["webhook"] = webhook_config
            scrape_params = prepare_firecrawl_params(
                mining_config,
                use_snake_case=snake_case_params,
                allowed_keys=combined_supported_kwargs,
                **extra_kwargs,
            )

            if mining_config.use_batch_scrape and hasattr(
                firecrawl_client, "batch_scrape"
            ):
                op_start = time.time()
                response = None
                try:
                    response = firecrawl_client.batch_scrape(
                        urls=academic_urls, **scrape_params
                    )
                except TypeError as type_error:
                    log_mining_operation(
                        "sdk_mismatch",
                        {
                            "operation": "batch_scrape",
                            "hint": "Retrying with alternate parameter casing",
                            "error": str(type_error),
                            "term": term,
                        },
                    )
                    alternate_case = not snake_case_params
                    fallback_params = prepare_firecrawl_params(
                        mining_config,
                        use_snake_case=alternate_case,
                        allowed_keys=supported_batch_kwargs,
                        **extra_kwargs,
                    )
                    try:
                        response = firecrawl_client.batch_scrape(
                            urls=academic_urls, **fallback_params
                        )
                        snake_case_params = alternate_case
                        scrape_params = fallback_params
                    except TypeError as pared_type_error:
                        log_mining_operation(
                            "sdk_mismatch",
                            {
                                "operation": "batch_scrape",
                                "hint": "Retrying with minimal kwargs",
                                "error": str(pared_type_error),
                                "term": term,
                            },
                        )
                        minimal_kwargs = dict(extra_kwargs)
                        try:
                            response = firecrawl_client.batch_scrape(
                                urls=academic_urls, **minimal_kwargs
                            )
                            snake_case_params = alternate_case
                            scrape_params = minimal_kwargs
                        except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
                            log_mining_operation(
                                "sdk_mismatch",
                                {
                                    "operation": "batch_scrape",
                                    "hint": "Minimal kwargs retry failed",
                                    "error": str(exc),
                                    "term": term,
                                },
                            )
                            track_mining_metrics(
                                "batch_scrape", time.time() - op_start, False
                            )
                            errors.append(
                                handle_firecrawl_error(
                                    exc,
                                    {"operation": "batch_scrape", "term": term},
                                )
                            )
                            continue
                    except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
                        track_mining_metrics(
                            "batch_scrape", time.time() - op_start, False
                        )
                        errors.append(
                            handle_firecrawl_error(
                                exc,
                                {"operation": "batch_scrape", "term": term},
                            )
                        )
                        continue
                except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
                    track_mining_metrics("batch_scrape", time.time() - op_start, False)
                    errors.append(
                        handle_firecrawl_error(
                            exc, {"operation": "batch_scrape", "term": term}
                        )
                    )
                    continue

                track_mining_metrics("batch_scrape", time.time() - op_start, True)
                extracted = handle_batch_scrape_response(response)
            else:
                extracted = []
                max_workers = min(
                    max(1, mining_config.max_concurrent_operations),
                    len(academic_urls),
                )

                if max_workers <= 1:
                    for url in academic_urls:
                        records, style_used, error = _scrape_with_fallbacks(
                            firecrawl_client,
                            mining_config,
                            extra_kwargs,
                            snake_case_params,
                            supported_scrape_kwargs,
                            term,
                            url,
                        )
                        if style_used is not None:
                            snake_case_params = style_used
                        if error:
                            errors.append(error)
                        extracted.extend(records)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(
                                _scrape_with_fallbacks,
                                firecrawl_client,
                                mining_config,
                                extra_kwargs,
                                snake_case_params,
                                supported_scrape_kwargs,
                                term,
                                url,
                            )
                            for url in academic_urls
                        ]
                        for future in as_completed(futures):
                            records, style_used, error = future.result()
                            if style_used is not None:
                                snake_case_params = style_used
                            if error:
                                errors.append(error)
                            extracted.extend(records)

                scrape_params = prepare_firecrawl_params(
                    mining_config,
                    use_snake_case=snake_case_params,
                    allowed_keys=combined_supported_kwargs,
                    **extra_kwargs,
                )

            for record in extracted:
                record["concept"] = term
                record.setdefault(
                    "score",
                    score_academic_relevance(record.get("content", "") or "", term),
                )
                mined_results.append(record)

            log_mining_operation(
                "mining_progress",
                {
                    "term": term,
                    "index": global_index,
                    "total": total_terms,
                    "urls_considered": len(academic_urls),
                },
            )

        processed_terms += len(batch_terms)
        elapsed = time.time() - start_time
        log_mining_operation(
            "mining_batch_progress",
            {
                "batch_index": batch_index,
                "batch_size": len(batch_terms),
                "completed_terms": processed_terms,
                "total_terms": total_terms,
                "message": format_progress_message(
                    processed_terms, total_terms, elapsed
                ),
            },
        )

    aggregated = aggregate_mining_results(mined_results)
    aggregated["statistics"].update(calculate_processing_stats(mined_results))
    aggregated["errors"] = errors
    aggregated["runtime_seconds"] = round(time.time() - start_time, 2)
    return aggregated


def _infer_snake_case_params(client: Any) -> bool:
    candidate = getattr(client, "batch_scrape", None) or getattr(client, "scrape", None)
    if candidate is None:
        return False
    try:
        parameters = inspect.signature(candidate).parameters
    except (TypeError, ValueError):  # pragma: no cover - reflect failures
        return False
    names = set(parameters)
    if {"max_age", "max_pages"} & names:
        return True
    if {"maxAge", "maxPages"} & names:
        return False
    return False


def _collect_supported_kwargs(client: Any, method_name: str) -> Optional[set[str]]:
    method = getattr(client, method_name, None)
    if method is None:
        return None
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - reflect failures
        return None
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return None
    supported = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return supported or None


def _scrape_with_fallbacks(
    firecrawl_client: Any,
    mining_config: MiningConfig,
    extra_kwargs: Dict[str, Any],
    initial_snake_case: bool,
    supported_scrape_kwargs: Optional[set[str]],
    term: str,
    url: str,
) -> Tuple[List[Dict[str, Any]], Optional[bool], Optional[Dict[str, Any]]]:
    op_start = time.time()
    local_snake_case = initial_snake_case

    def _on_success(response: Any, style: bool) -> Tuple[List[Dict[str, Any]], Optional[bool], Optional[Dict[str, Any]]]:
        track_mining_metrics("scrape", time.time() - op_start, True)
        return handle_batch_scrape_response(response), style, None

    def _on_failure(exc: Exception) -> Tuple[List[Dict[str, Any]], Optional[bool], Optional[Dict[str, Any]]]:
        track_mining_metrics("scrape", time.time() - op_start, False)
        return (
            [],
            None,
            handle_firecrawl_error(
                exc,
                {
                    "operation": "scrape",
                    "term": term,
                    "url": url,
                },
            ),
        )

    params = prepare_firecrawl_params(
        mining_config,
        use_snake_case=local_snake_case,
        allowed_keys=supported_scrape_kwargs,
        **extra_kwargs,
    )
    try:
        response = firecrawl_client.scrape(url=url, **params)
        return _on_success(response, local_snake_case)
    except TypeError as type_error:
        log_mining_operation(
            "sdk_mismatch",
            {
                "operation": "scrape",
                "hint": "Retrying with positional URL argument",
                "error": str(type_error),
                "term": term,
                "url": url,
            },
        )
        try:
            response = firecrawl_client.scrape(url, **params)
            return _on_success(response, local_snake_case)
        except TypeError as positional_error:
            log_mining_operation(
                "sdk_mismatch",
                {
                    "operation": "scrape",
                    "hint": "Retrying with alternate parameter casing",
                    "error": str(positional_error),
                    "term": term,
                    "url": url,
                },
            )
            alternate_case = not local_snake_case
            alternate_params = prepare_firecrawl_params(
                mining_config,
                use_snake_case=alternate_case,
                allowed_keys=supported_scrape_kwargs,
                **extra_kwargs,
            )
            try:
                response = firecrawl_client.scrape(url, **alternate_params)
                return _on_success(response, alternate_case)
            except TypeError as pared_type_error:
                log_mining_operation(
                    "sdk_mismatch",
                    {
                        "operation": "scrape",
                        "hint": "Retrying with minimal kwargs",
                        "error": str(pared_type_error),
                        "term": term,
                        "url": url,
                    },
                )
                minimal_kwargs = dict(extra_kwargs)
                try:
                    response = firecrawl_client.scrape(url, **minimal_kwargs)
                    # Minimal payload skips Firecrawl config params, so style preference stays flipped.
                    return _on_success(response, alternate_case)
                except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
                    log_mining_operation(
                        "sdk_mismatch",
                        {
                            "operation": "scrape",
                            "hint": "Minimal kwargs retry failed",
                            "error": str(exc),
                            "term": term,
                            "url": url,
                        },
                    )
                    return _on_failure(exc)
            except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
                return _on_failure(exc)
        except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
            return _on_failure(exc)
    except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
        return _on_failure(exc)

def _execute_search(firecrawl_client: Any, **kwargs: Any) -> Any:
    try:
        return firecrawl_client.search(**kwargs)
    except TypeError as type_error:
        log_mining_operation(
            "sdk_mismatch",
            {
                "operation": "search",
                "hint": "Retrying without categories parameter",
                "error": str(type_error),
            },
        )
        fallback_kwargs = {
            key: value for key, value in kwargs.items() if key in {"query", "limit"}
        }
        return firecrawl_client.search(**fallback_kwargs)


def monitor_job_if_requested(
    client: Any, job_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    try:
        status = monitor_queue_status(client, job_id)
        log_mining_operation("queue_status", {"job_id": job_id, "status": status})
        return status
    except Exception as exc:  # pragma: no cover - depends on Firecrawl SDK
        handle_firecrawl_error(exc, {"operation": "queue_status", "job_id": job_id})
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    config = override_with_cli_args(config, args)
    apply_logging_overrides(config, args)

    configure_logging(config.logging)
    log_mining_operation("startup", {"args": vars(args)})

    try:
        terms = load_terms_from_file(args.terms_file)
    except (FileNotFoundError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    try:
        client = get_firecrawl_client(config.firecrawl)
    except ConfigError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    monitor_job_if_requested(client, args.queue_job)

    start = time.time()
    aggregated = mine_concepts_simple(
        terms, client, config.mining, webhook_url=args.webhook_url
    )
    duration = time.time() - start

    output_file = save_mining_results(aggregated["results"], args.output, config.output)
    aggregated["statistics"]["processing_time_seconds"] = round(duration, 2)

    with open(
        Path(output_file).with_suffix(Path(output_file).suffix + ".meta.json"),
        "w",
        encoding="utf-8",
    ) as meta_file:
        json.dump(aggregated, meta_file, indent=2, sort_keys=True)

    logging.getLogger(__name__).info(
        "Mining complete | %s",
        format_progress_message(len(terms), len(terms), duration),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
