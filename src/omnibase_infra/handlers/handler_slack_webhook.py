# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Slack Webhook Handler - Infrastructure alerting via Slack webhooks.

This handler sends alerts to Slack channels using incoming webhooks,
with support for Block Kit formatting, retry with exponential backoff,
and rate limit handling.

Architecture:
    This handler follows the ONEX operation handler pattern:
    - Receives typed input (ModelSlackAlert)
    - Executes a single responsibility (Slack webhook delivery)
    - Returns typed output (ModelSlackAlertResult)
    - Uses error sanitization for security
    - Stateless and coroutine-safe for concurrent calls

Handler Responsibilities:
    - Format alerts as Slack Block Kit messages
    - Send webhooks with configurable retry logic
    - Handle 429 rate limiting gracefully
    - Sanitize errors to prevent credential exposure
    - Track operation timing and retry counts

Configuration:
    The webhook URL is resolved from the SLACK_WEBHOOK_URL environment
    variable. This keeps credentials out of code and configuration files.

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances.

Related Tickets:
    - OMN-1905: Add declarative Slack webhook handler to omnibase_infra
    - OMN-1895: Wiring Health Monitor alerting (blocked by this)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import aiohttp

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
    ModelSlackAlertResult,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)

# Default retry configuration
_DEFAULT_MAX_RETRIES: int = 3
_DEFAULT_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)
_DEFAULT_TIMEOUT_SECONDS: float = 10.0

# Slack Block Kit emoji mapping for severity levels
_SEVERITY_EMOJI: dict[EnumAlertSeverity, str] = {
    EnumAlertSeverity.CRITICAL: ":red_circle:",
    EnumAlertSeverity.ERROR: ":red_circle:",
    EnumAlertSeverity.WARNING: ":large_yellow_circle:",
    EnumAlertSeverity.INFO: ":large_blue_circle:",
}

# Default titles for each severity level
_SEVERITY_TITLES: dict[EnumAlertSeverity, str] = {
    EnumAlertSeverity.CRITICAL: "Critical Alert",
    EnumAlertSeverity.ERROR: "Error Alert",
    EnumAlertSeverity.WARNING: "Warning",
    EnumAlertSeverity.INFO: "Info",
}


class HandlerSlackWebhook:
    """Handler for Slack webhook alert delivery.

    Encapsulates all Slack-specific alerting logic for declarative
    node compliance. Supports Block Kit formatting, retry with exponential
    backoff, and rate limit handling.

    Error Handling:
        All errors are sanitized before inclusion in the result to prevent
        credential exposure. The handler never raises exceptions during
        normal operation - errors are captured in ModelSlackAlertResult.

    Rate Limiting:
        HTTP 429 responses trigger automatic retry with backoff. After
        max retries are exhausted, the operation fails gracefully with
        an error result rather than raising an exception.

    Attributes:
        _webhook_url: Slack webhook URL (from env or constructor)
        _http_session: Optional shared aiohttp session
        _max_retries: Maximum retry attempts for failed requests
        _retry_backoff: Tuple of backoff delays in seconds
        _timeout: HTTP request timeout in seconds

    Example:
        >>> import asyncio
        >>> handler = HandlerSlackWebhook()
        >>> alert = ModelSlackAlert(
        ...     severity=EnumAlertSeverity.ERROR,
        ...     message="Circuit breaker opened for Consul",
        ...     title="Infrastructure Alert",
        ... )
        >>> # result = await handler.handle(alert)
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: tuple[float, ...] = _DEFAULT_RETRY_BACKOFF_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize handler with webhook URL and optional HTTP session.

        Args:
            webhook_url: Slack webhook URL. If not provided, reads from
                SLACK_WEBHOOK_URL environment variable.
            http_session: Optional shared aiohttp ClientSession. If not
                provided, a new session is created per request.
            max_retries: Maximum retry attempts for failed requests.
                Default is 3.
            retry_backoff: Tuple of backoff delays in seconds for each
                retry attempt. Default is (1.0, 2.0, 4.0).
            timeout: HTTP request timeout in seconds. Default is 10.0.
        """
        self._webhook_url: str = (
            webhook_url if webhook_url else os.getenv("SLACK_WEBHOOK_URL", "")
        )
        self._http_session = http_session
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._timeout = timeout

    async def handle(
        self,
        alert: ModelSlackAlert,
    ) -> ModelSlackAlertResult:
        """Execute Slack webhook alert delivery.

        Formats the alert as a Slack Block Kit message and sends it
        to the configured webhook URL with retry logic.

        Args:
            alert: Alert payload containing severity, message, and optional
                details.

        Returns:
            ModelSlackAlertResult with:
                - success: True if alert was delivered
                - duration_ms: Time taken for the operation
                - correlation_id: From the input alert
                - error: Sanitized error message (only on failure)
                - error_code: Error code for programmatic handling
                - retry_count: Number of retry attempts made

        Note:
            This handler does not raise exceptions during normal operation.
            All errors are captured and returned in ModelSlackAlertResult
            to support graceful degradation in alerting scenarios.
        """
        start_time = time.perf_counter()
        correlation_id = alert.correlation_id
        retry_count = 0

        # Validate webhook URL is configured
        if not self._webhook_url:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ModelSlackAlertResult(
                success=False,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                error="SLACK_WEBHOOK_URL not configured",
                error_code="SLACK_NOT_CONFIGURED",
                retry_count=0,
            )

        # Format the alert as Block Kit message
        slack_payload = self._format_block_kit_message(alert)

        # Create or use existing HTTP session
        session_created = False
        session = self._http_session
        if session is None:
            session = aiohttp.ClientSession()
            session_created = True

        try:
            return await self._send_with_retry(
                session=session,
                slack_payload=slack_payload,
                correlation_id=correlation_id,
                start_time=start_time,
            )
        finally:
            # Close session if we created it
            if session_created and session is not None:
                await session.close()

    async def _send_with_retry(
        self,
        session: aiohttp.ClientSession,
        slack_payload: dict[str, object],
        correlation_id: UUID,
        start_time: float,
    ) -> ModelSlackAlertResult:
        """Send webhook with retry logic and rate limit handling.

        Args:
            session: aiohttp ClientSession for HTTP requests
            slack_payload: Formatted Slack Block Kit payload
            correlation_id: UUID for distributed tracing
            start_time: Performance timer start for duration calculation

        Returns:
            ModelSlackAlertResult with operation outcome
        """
        retry_count = 0
        last_error: str | None = None
        last_error_code: str | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with session.post(
                    self._webhook_url,
                    json=slack_payload,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    if response.status == 200:
                        # Success
                        logger.info(
                            "Slack alert delivered successfully",
                            extra={
                                "correlation_id": str(correlation_id),
                                "duration_ms": round(duration_ms, 2),
                                "retry_count": retry_count,
                            },
                        )
                        return ModelSlackAlertResult(
                            success=True,
                            duration_ms=duration_ms,
                            correlation_id=correlation_id,
                            retry_count=retry_count,
                        )

                    elif response.status == 429:
                        # Rate limited - retry with backoff
                        last_error = "Slack rate limit (429)"
                        last_error_code = "SLACK_RATE_LIMITED"
                        logger.warning(
                            "Slack rate limited, will retry",
                            extra={
                                "correlation_id": str(correlation_id),
                                "attempt": attempt + 1,
                                "max_attempts": self._max_retries + 1,
                            },
                        )

                    elif response.status >= 400:
                        # Other HTTP error
                        response_text = await response.text()
                        last_error = f"HTTP {response.status}: {response_text[:100]}"
                        last_error_code = f"SLACK_HTTP_{response.status}"
                        logger.warning(
                            "Slack webhook error",
                            extra={
                                "correlation_id": str(correlation_id),
                                "status_code": response.status,
                                "attempt": attempt + 1,
                            },
                        )

            except TimeoutError:
                last_error = "Request timeout"
                last_error_code = "SLACK_TIMEOUT"
                logger.warning(
                    "Slack webhook timeout",
                    extra={
                        "correlation_id": str(correlation_id),
                        "timeout_seconds": self._timeout,
                        "attempt": attempt + 1,
                    },
                )

            except aiohttp.ClientConnectorError as e:
                last_error = sanitize_error_message(e)
                last_error_code = "SLACK_CONNECTION_ERROR"
                logger.warning(
                    "Slack webhook connection error",
                    extra={
                        "correlation_id": str(correlation_id),
                        "attempt": attempt + 1,
                    },
                )

            except aiohttp.ClientError as e:
                last_error = sanitize_error_message(e)
                last_error_code = "SLACK_CLIENT_ERROR"
                logger.warning(
                    "Slack webhook client error",
                    extra={
                        "correlation_id": str(correlation_id),
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                    },
                )

            # Retry with backoff if we have retries remaining
            if attempt < self._max_retries:
                backoff_index = min(attempt, len(self._retry_backoff) - 1)
                backoff_seconds = self._retry_backoff[backoff_index]
                logger.info(
                    "Retrying Slack webhook",
                    extra={
                        "correlation_id": str(correlation_id),
                        "backoff_seconds": backoff_seconds,
                        "attempt": attempt + 1,
                    },
                )
                await asyncio.sleep(backoff_seconds)
                retry_count += 1

        # All retries exhausted
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Slack alert delivery failed after retries",
            extra={
                "correlation_id": str(correlation_id),
                "duration_ms": round(duration_ms, 2),
                "retry_count": retry_count,
                "error_code": last_error_code,
            },
        )

        return ModelSlackAlertResult(
            success=False,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            error=last_error,
            error_code=last_error_code,
            retry_count=retry_count,
        )

    def _format_block_kit_message(self, alert: ModelSlackAlert) -> dict[str, object]:
        """Format alert as Slack Block Kit message.

        Creates a rich formatted message using Slack's Block Kit API
        with header, message body, and optional detail fields.

        Args:
            alert: Alert payload to format

        Returns:
            Dict containing Slack Block Kit blocks structure
        """
        emoji = _SEVERITY_EMOJI.get(alert.severity, ":white_circle:")
        title = alert.title or _SEVERITY_TITLES.get(alert.severity, "Alert")

        blocks: list[dict[str, object]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True,
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message[:3000],  # Slack limit
                },
            },
        ]

        # Add detail fields if provided
        if alert.details:
            fields: list[dict[str, object]] = []
            for key, value in list(alert.details.items())[:10]:  # Limit to 10 fields
                # Convert value to string, truncate if needed
                value_str = str(value)[:100]
                fields.append({"type": "mrkdwn", "text": f"*{key}:*\n{value_str}"})

            # Slack allows max 10 fields per section
            if fields:
                blocks.append({"type": "section", "fields": fields})

        # Add correlation ID for traceability
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Correlation: `{str(alert.correlation_id)[:16]}...`",
                    }
                ],
            }
        )

        return {"blocks": blocks}


__all__: list[str] = ["HandlerSlackWebhook"]
