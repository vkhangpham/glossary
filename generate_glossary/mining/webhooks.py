"""Webhook management and security module.

This module provides comprehensive webhook setup, signature verification,
event handling, and failure recovery for real-time notifications in the
Firecrawl-based mining system.
"""

import os
import time
import hmac
import hashlib
import logging
import collections
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from firecrawl import FirecrawlApp

from .models import WebhookConfig
from .client import get_client, get_firecrawl_api_key
from generate_glossary.utils.error_handler import (
    ExternalServiceError, handle_error, processing_context
)
from generate_glossary.utils.logger import get_logger, log_processing_step, log_with_context


logger = get_logger(__name__)

# Global webhook state
_active_webhooks: Dict[str, WebhookConfig] = {}
# Use bounded deque to enforce hard memory limits (max 1000 events)
_webhook_events: collections.deque[Dict[str, Any]] = collections.deque(maxlen=1000)
_webhook_stats: Dict[str, Any] = {
    'total_events': 0,
    'successful_events': 0,
    'failed_events': 0,
    'signature_verifications': 0,
    'signature_failures': 0
}


def setup_webhooks(webhook_config: WebhookConfig, app: Optional[FirecrawlApp] = None) -> bool:
    """Setup enhanced webhooks with signatures for real-time job notifications.

    Configures webhooks with v2.2.0 enhancements including:
    - Event subscriptions (started, page, completed, failed)
    - Signature verification for security
    - Event failure handling and retry mechanisms
    - Actual registration with Firecrawl via SDK or REST API

    Args:
        webhook_config: WebhookConfig containing webhook settings
        app: Optional FirecrawlApp instance for SDK-based registration

    Returns:
        True if webhook setup successful, False otherwise
    """
    if app is None:
        app = get_client()
        if app is None:
            return False

    with processing_context("setup_webhooks") as correlation_id:
        log_processing_step(
            logger,
            "setup_webhooks",
            "started",
            {
                "webhook_url": webhook_config.url,
                "events": webhook_config.events,
                "verify_signature": webhook_config.verify_signature,
                "has_app": app is not None
            },
            correlation_id=correlation_id
        )

        try:
            # Validate webhook URL
            parsed_url = urlparse(webhook_config.url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid webhook URL: {webhook_config.url}")

            # Validate events
            valid_events = ["started", "page", "completed", "failed", "error"]
            for event in webhook_config.events:
                if event not in valid_events:
                    logger.warning(f"Unknown event type: {event}")

            # Log webhook configuration
            log_with_context(
                logger,
                logging.INFO,
                f"Setting up webhook for events: {', '.join(webhook_config.events)}",
                correlation_id=correlation_id
            )

            success = False
            max_retries = 3
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    # Try SDK-based registration first if app is available
                    if app:
                        success = _try_sdk_webhook_registration(app, webhook_config, correlation_id)
                        if success:
                            break

                    # Fallback to direct REST API registration
                    success = _try_rest_webhook_registration(webhook_config, correlation_id)
                    if success:
                        break

                except Exception as e:
                    log_with_context(
                        logger,
                        logging.WARNING,
                        f"Webhook setup attempt {attempt + 1} failed: {e}",
                        correlation_id=correlation_id
                    )

                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        log_with_context(
                            logger,
                            logging.INFO,
                            f"Retrying webhook setup in {delay}s...",
                            correlation_id=correlation_id
                        )
                        time.sleep(delay)

            if success:
                # Store webhook configuration
                _active_webhooks[webhook_config.url] = webhook_config

                log_processing_step(
                    logger,
                    "setup_webhooks",
                    "completed",
                    {
                        "webhook_url": webhook_config.url,
                        "events": webhook_config.events,
                        "registration_method": "sdk" if app else "rest"
                    },
                    correlation_id=correlation_id
                )
            else:
                log_with_context(
                    logger,
                    logging.ERROR,
                    f"Failed to setup webhook after {max_retries} attempts",
                    correlation_id=correlation_id
                )

            return success

        except Exception as e:
            handle_error(
                ExternalServiceError(f"Webhook setup failed: {e}", service="firecrawl"),
                context={
                    "webhook_url": webhook_config.url,
                    "events": webhook_config.events
                },
                operation="setup_webhooks"
            )
            log_with_context(
                logger,
                logging.ERROR,
                f"Webhook setup failed: {e}",
                correlation_id=correlation_id
            )
            return False


def _try_sdk_webhook_registration(app: FirecrawlApp, webhook_config: WebhookConfig,
                                  correlation_id: str) -> bool:
    """Try to register webhook using SDK methods.

    Args:
        app: Firecrawl app instance
        webhook_config: Webhook configuration
        correlation_id: Correlation ID for logging

    Returns:
        True if registration successful, False otherwise
    """
    try:
        # Try to use SDK webhook registration method (if available)
        if hasattr(app, 'register_webhook'):
            webhook_response = app.register_webhook(
                url=webhook_config.url,
                events=webhook_config.events,
                secret=webhook_config.secret
            )
            log_with_context(
                logger,
                logging.INFO,
                f"Webhook registered via SDK: {webhook_response}",
                correlation_id=correlation_id
            )
            return True
        elif hasattr(app, 'set_webhook'):
            webhook_response = app.set_webhook(
                webhook_config.url,
                events=webhook_config.events
            )
            log_with_context(
                logger,
                logging.INFO,
                f"Webhook set via SDK: {webhook_response}",
                correlation_id=correlation_id
            )
            return True
        else:
            log_with_context(
                logger,
                logging.DEBUG,
                "SDK webhook methods not available",
                correlation_id=correlation_id
            )
            return False

    except (AttributeError, TypeError) as sdk_error:
        log_with_context(
            logger,
            logging.WARNING,
            f"SDK webhook registration not available: {sdk_error}",
            correlation_id=correlation_id
        )
        return False


def _try_rest_webhook_registration(webhook_config: WebhookConfig, correlation_id: str) -> bool:
    """Try to register webhook using direct REST API.

    Args:
        webhook_config: Webhook configuration
        correlation_id: Correlation ID for logging

    Returns:
        True if registration successful, False otherwise
    """
    try:
        import requests
    except ImportError:
        log_with_context(
            logger,
            logging.ERROR,
            "requests library not available for REST webhook registration",
            correlation_id=correlation_id
        )
        return False

    api_key = get_firecrawl_api_key()
    if not api_key:
        log_with_context(
            logger,
            logging.ERROR,
            "Firecrawl API key not available for REST webhook registration",
            correlation_id=correlation_id
        )
        return False

    webhook_payload = {
        "url": webhook_config.url,
        "events": webhook_config.events
    }

    if webhook_config.secret:
        webhook_payload["secret"] = webhook_config.secret

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        # Try different potential webhook endpoints
        potential_endpoints = [
            "https://api.firecrawl.dev/v1/webhook/register",
            "https://api.firecrawl.dev/v2/webhook/register",
            "https://api.firecrawl.dev/webhook/register"
        ]

        for endpoint in potential_endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json=webhook_payload,
                    headers=headers,
                    timeout=30
                )

                if response.status_code in [200, 201]:
                    log_with_context(
                        logger,
                        logging.INFO,
                        f"Webhook registered via REST API ({endpoint}): {response.json()}",
                        correlation_id=correlation_id
                    )
                    return True
                elif response.status_code == 404:
                    # Endpoint not found, try next one
                    continue
                else:
                    log_with_context(
                        logger,
                        logging.WARNING,
                        f"Webhook registration failed at {endpoint}: {response.status_code} {response.text}",
                        correlation_id=correlation_id
                    )

            except requests.exceptions.RequestException as e:
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"Request failed for {endpoint}: {e}",
                    correlation_id=correlation_id
                )
                continue

        log_with_context(
            logger,
            logging.ERROR,
            "All webhook registration endpoints failed",
            correlation_id=correlation_id
        )
        return False

    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"REST webhook registration failed: {e}",
            correlation_id=correlation_id
        )
        return False


def verify_webhook_signature(payload_bytes: bytes, signature: str, secret: str) -> bool:
    """Verify webhook signature for security (v2.2.0 feature).

    Args:
        payload_bytes: Raw webhook payload as bytes (exact HTTP request body)
        signature: Signature from webhook headers
        secret: Webhook secret for verification

    Returns:
        True if signature is valid, False otherwise
    """
    _webhook_stats['signature_verifications'] += 1

    if not secret:
        logger.warning("No webhook secret provided for signature verification")
        _webhook_stats['signature_failures'] += 1
        return False

    if not signature:
        logger.warning("No signature provided for verification")
        _webhook_stats['signature_failures'] += 1
        return False

    try:
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()

        # Remove 'sha256=' prefix if present
        clean_signature = signature.replace('sha256=', '')

        is_valid = hmac.compare_digest(expected_signature, clean_signature)

        if not is_valid:
            _webhook_stats['signature_failures'] += 1
            logger.warning("Webhook signature verification failed")

        return is_valid

    except Exception as e:
        logger.error(f"Webhook signature verification error: {e}")
        _webhook_stats['signature_failures'] += 1
        return False


def handle_webhook_event(event_data: Dict[str, Any], webhook_url: str,
                        signature: Optional[str] = None,
                        raw_body: Optional[bytes] = None) -> bool:
    """Handle incoming webhook event.

    Args:
        event_data: Event data from webhook
        webhook_url: URL of the webhook that received the event
        signature: Optional signature for verification
        raw_body: Raw HTTP request body as bytes for signature verification

    Returns:
        True if event handled successfully, False otherwise
    """
    with processing_context("handle_webhook_event") as correlation_id:
        _webhook_stats['total_events'] += 1

        try:
            # Get webhook configuration
            webhook_config = _active_webhooks.get(webhook_url)
            if not webhook_config:
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"Received event for unknown webhook: {webhook_url}",
                    correlation_id=correlation_id
                )
                return False

            # Verify signature if required
            if webhook_config.verify_signature and webhook_config.secret:
                if not signature:
                    log_with_context(
                        logger,
                        logging.ERROR,
                        "Signature required but not provided",
                        correlation_id=correlation_id
                    )
                    _webhook_stats['failed_events'] += 1
                    return False

                if not raw_body:
                    log_with_context(
                        logger,
                        logging.ERROR,
                        "Raw body required for signature verification but not provided",
                        correlation_id=correlation_id
                    )
                    _webhook_stats['failed_events'] += 1
                    return False

                if not verify_webhook_signature(raw_body, signature, webhook_config.secret):
                    log_with_context(
                        logger,
                        logging.ERROR,
                        "Webhook signature verification failed",
                        correlation_id=correlation_id
                    )
                    _webhook_stats['failed_events'] += 1
                    return False

            # Process the event
            event_type = event_data.get('type', 'unknown')
            job_id = event_data.get('job_id', 'unknown')

            log_with_context(
                logger,
                logging.INFO,
                f"Processing webhook event: {event_type} for job {job_id}",
                correlation_id=correlation_id
            )

            # Store event for tracking
            event_record = {
                'timestamp': time.time(),
                'webhook_url': webhook_url,
                'event_type': event_type,
                'job_id': job_id,
                'data': event_data,
                'correlation_id': correlation_id
            }
            # Add to bounded deque (automatically removes oldest when full)
            _webhook_events.append(event_record)

            # Process based on event type
            success = _process_event_by_type(event_type, event_data, correlation_id)

            if success:
                _webhook_stats['successful_events'] += 1
            else:
                _webhook_stats['failed_events'] += 1

            return success

        except Exception as e:
            handle_error(
                ExternalServiceError(f"Webhook event handling failed: {e}", service="webhook"),
                context={
                    "webhook_url": webhook_url,
                    "event_type": event_data.get('type', 'unknown')
                },
                operation="handle_webhook_event"
            )
            _webhook_stats['failed_events'] += 1
            return False


def _process_event_by_type(event_type: str, event_data: Dict[str, Any], correlation_id: str) -> bool:
    """Process webhook event based on its type.

    Args:
        event_type: Type of the event
        event_data: Event data
        correlation_id: Correlation ID for logging

    Returns:
        True if processing successful, False otherwise
    """
    try:
        if event_type == "started":
            log_with_context(
                logger,
                logging.INFO,
                f"Job started: {event_data.get('job_id')}",
                correlation_id=correlation_id
            )
            return True

        elif event_type == "page":
            pages_processed = event_data.get('pages_processed', 0)
            log_with_context(
                logger,
                logging.DEBUG,
                f"Job progress: {event_data.get('job_id')} - {pages_processed} pages processed",
                correlation_id=correlation_id
            )
            return True

        elif event_type == "completed":
            pages_total = event_data.get('pages_total', 0)
            log_with_context(
                logger,
                logging.INFO,
                f"Job completed: {event_data.get('job_id')} - {pages_total} pages total",
                correlation_id=correlation_id
            )
            return True

        elif event_type == "failed":
            error_message = event_data.get('error', 'Unknown error')
            log_with_context(
                logger,
                logging.ERROR,
                f"Job failed: {event_data.get('job_id')} - {error_message}",
                correlation_id=correlation_id
            )
            return True

        elif event_type == "error":
            error_message = event_data.get('error', 'Unknown error')
            log_with_context(
                logger,
                logging.ERROR,
                f"Job error: {event_data.get('job_id')} - {error_message}",
                correlation_id=correlation_id
            )
            return True

        else:
            log_with_context(
                logger,
                logging.WARNING,
                f"Unknown event type: {event_type}",
                correlation_id=correlation_id
            )
            return False

    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"Event processing failed for {event_type}: {e}",
            correlation_id=correlation_id
        )
        return False


def get_webhook_stats() -> Dict[str, Any]:
    """Get webhook statistics.

    Returns:
        Dictionary with webhook statistics
    """
    stats = _webhook_stats.copy()
    stats.update({
        'active_webhooks': len(_active_webhooks),
        'recent_events': len(_webhook_events),
        'success_rate': (stats['successful_events'] / max(1, stats['total_events'])) * 100,
        'signature_success_rate': (
            (stats['signature_verifications'] - stats['signature_failures']) /
            max(1, stats['signature_verifications'])
        ) * 100 if stats['signature_verifications'] > 0 else 100
    })
    return stats


def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent webhook events.

    Args:
        limit: Maximum number of events to return

    Returns:
        List of recent webhook events
    """
    return _webhook_events[-limit:]


def list_active_webhooks() -> List[WebhookConfig]:
    """Get list of active webhook configurations.

    Returns:
        List of active webhook configurations
    """
    return list(_active_webhooks.values())


def remove_webhook(webhook_url: str) -> bool:
    """Remove webhook configuration.

    Args:
        webhook_url: URL of webhook to remove

    Returns:
        True if removed successfully, False otherwise
    """
    if webhook_url in _active_webhooks:
        del _active_webhooks[webhook_url]
        logger.info(f"Removed webhook: {webhook_url}")
        return True
    else:
        logger.warning(f"Webhook not found: {webhook_url}")
        return False


def test_webhook_connectivity(webhook_url: str) -> Dict[str, Any]:
    """Test webhook endpoint connectivity.

    Args:
        webhook_url: Webhook URL to test

    Returns:
        Dictionary with connectivity test results
    """
    try:
        import requests
    except ImportError:
        return {"status": "error", "message": "requests library not available"}

    test_result = {
        "webhook_url": webhook_url,
        "timestamp": time.time(),
        "status": "unknown",
        "response_time": None,
        "status_code": None,
        "message": ""
    }

    try:
        start_time = time.time()

        # Send a test payload
        test_payload = {
            "type": "test",
            "job_id": "test-job-123",
            "timestamp": time.time(),
            "message": "Webhook connectivity test"
        }

        response = requests.post(
            webhook_url,
            json=test_payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )

        response_time = time.time() - start_time
        test_result.update({
            "status": "success" if response.status_code < 400 else "failed",
            "response_time": response_time,
            "status_code": response.status_code,
            "message": f"Response: {response.text[:200]}"
        })

    except requests.exceptions.Timeout:
        test_result.update({
            "status": "timeout",
            "message": "Request timed out after 10 seconds"
        })
    except requests.exceptions.ConnectionError:
        test_result.update({
            "status": "connection_error",
            "message": "Failed to connect to webhook URL"
        })
    except Exception as e:
        test_result.update({
            "status": "error",
            "message": f"Test failed: {e}"
        })

    return test_result


def reset_webhook_state() -> None:
    """Reset webhook state - useful for testing."""
    global _active_webhooks, _webhook_events, _webhook_stats
    _active_webhooks.clear()
    _webhook_events.clear()
    _webhook_stats = {
        'total_events': 0,
        'successful_events': 0,
        'failed_events': 0,
        'signature_verifications': 0,
        'signature_failures': 0
    }


__all__ = [
    'setup_webhooks',
    'verify_webhook_signature',
    'handle_webhook_event',
    'get_webhook_stats',
    'get_recent_events',
    'list_active_webhooks',
    'remove_webhook',
    'test_webhook_connectivity',
    'reset_webhook_state'
]