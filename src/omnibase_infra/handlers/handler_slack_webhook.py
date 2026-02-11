# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Slack Webhook Handler - Infrastructure alerting via Slack.

This handler sends alerts to Slack channels using either incoming webhooks
or the Slack Web API (chat.postMessage), with support for Block Kit
formatting, retry with exponential backoff, rate limit handling, and
message threading.

Architecture:
    This handler follows the ONEX operation handler pattern:
    - Receives typed input (ModelSlackAlert)
    - Executes a single responsibility (Slack message delivery)
    - Returns typed output (ModelSlackAlertResult)
    - Uses error sanitization for security
    - Stateless and coroutine-safe for concurrent calls

Delivery Modes:
    - **Web API mode** (preferred): When ``bot_token`` is configured, uses
      ``chat.postMessage`` which returns a ``ts`` value for threading.
    - **Webhook mode** (fallback): When only ``webhook_url`` is configured,
      uses incoming webhooks. No threading support (webhooks don't return ts).

Handler Responsibilities:
    - Format alerts as Slack Block Kit messages
    - Send via Web API or webhook with configurable retry logic
    - Handle 429 rate limiting gracefully
    - Sanitize errors to prevent credential exposure
    - Track operation timing and retry counts
    - Support message threading via thread_ts

Configuration:
    - Web API: SLACK_BOT_TOKEN + SLACK_CHANNEL_ID environment variables
    - Webhook: SLACK_WEBHOOK_URL environment variable

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different request instances.

Related Tickets:
    - OMN-1905: Add declarative Slack webhook handler to omnibase_infra
    - OMN-2157: Extend with Web API support for threading
    - OMN-1895: Wiring Health Monitor alerting (blocked by OMN-1905)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

import aiohttp

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

# Slack Web API endpoint
_SLACK_WEB_API_URL: str = "https://slack.com/api/chat.postMessage"

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
    """Handler for Slack alert delivery via webhook or Web API.

    Encapsulates all Slack-specific alerting logic for declarative
    node compliance. Supports Block Kit formatting, retry with exponential
    backoff, rate limit handling, and message threading (Web API mode).

    Delivery Mode Resolution:
        - If ``bot_token`` is set: uses Slack Web API (chat.postMessage).
          Supports threading via ``thread_ts``.
        - If only ``webhook_url`` is set: uses incoming webhooks (original
          behavior). No threading support.
        - If neither is set: returns SLACK_NOT_CONFIGURED error.

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
        _bot_token: Slack Bot Token for Web API (from env or constructor)
        _default_channel: Default channel ID for Web API posts
        _http_session: Optional shared aiohttp session
        _max_retries: Maximum retry attempts for failed requests
        _retry_backoff: Tuple of backoff delays in seconds
        _timeout: HTTP request timeout in seconds

    Example:
        >>> import asyncio
        >>> # Web API mode (supports threading)
        >>> handler = HandlerSlackWebhook(
        ...     bot_token="xoxb-...",
        ...     default_channel="C01234567",
        ... )
        >>> alert = ModelSlackAlert(
        ...     severity=EnumAlertSeverity.ERROR,
        ...     message="Circuit breaker opened for Consul",
        ...     title="Infrastructure Alert",
        ... )
        >>> # result = await handler.handle(alert)
        >>> # result.thread_ts  # ts of the posted message
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: tuple[float, ...] = _DEFAULT_RETRY_BACKOFF_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        bot_token: str | None = None,
        default_channel: str | None = None,
    ) -> None:
        """Initialize handler with webhook URL or bot token.

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
            bot_token: Slack Bot Token for Web API mode. If not provided,
                reads from SLACK_BOT_TOKEN environment variable.
            default_channel: Default channel ID for Web API posts. If not
                provided, reads from SLACK_CHANNEL_ID environment variable.
        """
        self._webhook_url: str = (
            webhook_url
            if webhook_url is not None
            else os.getenv("SLACK_WEBHOOK_URL", "")
        )
        self._bot_token: str = (
            bot_token if bot_token is not None else os.getenv("SLACK_BOT_TOKEN", "")
        )
        self._default_channel: str = (
            default_channel
            if default_channel is not None
            else os.getenv("SLACK_CHANNEL_ID", "")
        )
        self._http_session = http_session
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._timeout = timeout

    def __repr__(self) -> str:
        """Mask credentials to prevent accidental exposure in logs/tracebacks."""
        mode = "web_api" if self._bot_token else "webhook"
        return f"<{type(self).__name__} mode={mode}>"

    @property
    def uses_web_api(self) -> bool:
        """Whether this handler uses the Slack Web API (vs webhook)."""
        return bool(self._bot_token)

    async def handle(
        self,
        alert: ModelSlackAlert,
    ) -> ModelSlackAlertResult:
        """Execute Slack alert delivery via Web API or webhook.

        Mode is resolved automatically: if ``bot_token`` is configured,
        uses Web API (chat.postMessage) which supports threading. Otherwise
        falls back to incoming webhook.

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
                - thread_ts: Slack ts of the posted message (Web API only)

        Note:
            This handler does not raise exceptions during normal operation.
            All errors are captured and returned in ModelSlackAlertResult
            to support graceful degradation in alerting scenarios.
        """
        start_time = time.perf_counter()

        if self._bot_token:
            return await self._handle_web_api(alert, start_time)
        return await self._handle_webhook(alert, start_time)

    async def _handle_webhook(
        self,
        alert: ModelSlackAlert,
        start_time: float,
    ) -> ModelSlackAlertResult:
        """Send alert via incoming webhook (original behavior)."""
        correlation_id = alert.correlation_id

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

        slack_payload = self._format_block_kit_message(alert)

        session_created = False
        session = self._http_session
        if session is None:
            session = aiohttp.ClientSession()
            session_created = True

        try:
            return await self._send_webhook_with_retry(
                session=session,
                slack_payload=slack_payload,
                correlation_id=correlation_id,
                start_time=start_time,
            )
        finally:
            if session_created and session is not None:
                await session.close()

    async def _handle_web_api(
        self,
        alert: ModelSlackAlert,
        start_time: float,
    ) -> ModelSlackAlertResult:
        """Send alert via Slack Web API (chat.postMessage)."""
        correlation_id = alert.correlation_id
        channel = alert.channel or self._default_channel

        if not channel:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ModelSlackAlertResult(
                success=False,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                error="SLACK_CHANNEL_ID not configured (required for Web API mode)",
                error_code="SLACK_NOT_CONFIGURED",
                retry_count=0,
            )

        blocks = self._format_block_kit_message(alert)["blocks"]
        emoji = _SEVERITY_EMOJI.get(alert.severity, ":white_circle:")
        title = alert.title or _SEVERITY_TITLES.get(alert.severity, "Alert")
        # Strip markdown formatting for clean notification previews
        plain_message = (
            alert.message[:200].replace("*", "").replace("_", "").replace("`", "")
        )
        fallback_text = f"{emoji} {title}: {plain_message}"

        api_payload: dict[str, object] = {
            "channel": channel,
            "blocks": blocks,
            "text": fallback_text,
        }
        if alert.thread_ts:
            api_payload["thread_ts"] = alert.thread_ts

        session_created = False
        session = self._http_session
        if session is None:
            session = aiohttp.ClientSession()
            session_created = True

        try:
            return await self._send_web_api_with_retry(
                session=session,
                api_payload=api_payload,
                correlation_id=correlation_id,
                start_time=start_time,
            )
        finally:
            if session_created and session is not None:
                await session.close()

    async def _send_webhook_with_retry(
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
                        # Slack webhooks return 200 with a text body.
                        # Only "ok" indicates true success; known error
                        # bodies and unrecognized responses are treated
                        # as failures to avoid masking new error strings.
                        body_text = (await response.text()).strip()
                        if body_text != "ok":
                            last_error = f"Webhook error: {body_text[:100]}"
                            last_error_code = "SLACK_WEBHOOK_ERROR"
                            logger.warning(
                                "Slack webhook returned non-ok body on 200",
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "body": body_text[:100],
                                    "attempt": attempt + 1,
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

                        logger.info(
                            "Slack alert delivered successfully",
                            extra={
                                "correlation_id": str(correlation_id),
                                "duration_ms": round(duration_ms, 2),
                                "retry_count": retry_count,
                                "mode": "webhook",
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
                        retry_after = response.headers.get("Retry-After")
                        last_error = "Slack rate limit (429)"
                        last_error_code = "SLACK_RATE_LIMITED"
                        logger.warning(
                            "Slack rate limited, will retry",
                            extra={
                                "correlation_id": str(correlation_id),
                                "attempt": attempt + 1,
                                "max_attempts": self._max_retries + 1,
                                "retry_after": retry_after,
                            },
                        )
                        # Respect Slack's Retry-After header if present
                        if retry_after and attempt < self._max_retries:
                            try:
                                await asyncio.sleep(float(retry_after))
                                retry_count += 1
                                continue
                            except (ValueError, TypeError):
                                pass  # Fall through to default backoff

                    elif 400 <= response.status < 500:
                        # 4xx client errors (except 429, handled above)
                        # are non-retryable -- fail fast.
                        response_text = await response.text()
                        last_error = f"HTTP {response.status}: {response_text[:100]}"
                        last_error_code = f"SLACK_HTTP_{response.status}"
                        logger.warning(
                            "Slack webhook client error (non-retryable)",
                            extra={
                                "correlation_id": str(correlation_id),
                                "status_code": response.status,
                                "attempt": attempt + 1,
                            },
                        )
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        return ModelSlackAlertResult(
                            success=False,
                            duration_ms=duration_ms,
                            correlation_id=correlation_id,
                            error=last_error,
                            error_code=last_error_code,
                            retry_count=retry_count,
                        )

                    elif response.status >= 500:
                        # 5xx server errors are retryable
                        response_text = await response.text()
                        last_error = f"HTTP {response.status}: {response_text[:100]}"
                        last_error_code = f"SLACK_HTTP_{response.status}"
                        logger.warning(
                            "Slack webhook server error (retryable)",
                            extra={
                                "correlation_id": str(correlation_id),
                                "status_code": response.status,
                                "attempt": attempt + 1,
                            },
                        )

                    else:
                        # Unexpected status (1xx, 2xx non-200, 3xx)
                        last_error = f"Unexpected HTTP {response.status}"
                        last_error_code = f"SLACK_HTTP_{response.status}"
                        logger.warning(
                            "Slack webhook unexpected status code",
                            extra={
                                "correlation_id": str(correlation_id),
                                "status_code": response.status,
                                "attempt": attempt + 1,
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
                "mode": "webhook",
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

    async def _send_web_api_with_retry(
        self,
        session: aiohttp.ClientSession,
        api_payload: dict[str, object],
        correlation_id: UUID,
        start_time: float,
    ) -> ModelSlackAlertResult:
        """Send via Slack Web API with retry logic.

        The Web API always returns HTTP 200 with a JSON body containing
        ``ok: true/false``. On success, the response includes a ``ts``
        field that can be used for threading.

        Args:
            session: aiohttp ClientSession for HTTP requests
            api_payload: chat.postMessage payload
            correlation_id: UUID for distributed tracing
            start_time: Performance timer start for duration calculation

        Returns:
            ModelSlackAlertResult with operation outcome and thread_ts
        """
        retry_count = 0
        last_error: str | None = None
        last_error_code: str | None = None

        # Auth header constructed per-call (not stored on self) to limit
        # exposure surface.  The token is intentionally never logged.
        headers = {
            "Authorization": f"Bearer {self._bot_token}",
        }

        for attempt in range(self._max_retries + 1):
            try:
                async with session.post(
                    _SLACK_WEB_API_URL,
                    json=api_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        last_error = "Slack rate limit (429)"
                        last_error_code = "SLACK_RATE_LIMITED"
                        logger.warning(
                            "Slack Web API rate limited, will retry",
                            extra={
                                "correlation_id": str(correlation_id),
                                "attempt": attempt + 1,
                                "max_attempts": self._max_retries + 1,
                                "retry_after": retry_after,
                            },
                        )
                        # Respect Slack's Retry-After header if present
                        if retry_after and attempt < self._max_retries:
                            try:
                                await asyncio.sleep(float(retry_after))
                                retry_count += 1
                                continue
                            except (ValueError, TypeError):
                                pass  # Fall through to default backoff
                    elif 400 <= response.status < 500:
                        # 4xx client errors (except 429, handled above)
                        # are non-retryable -- fail fast.
                        response_text = await response.text()
                        last_error = f"HTTP {response.status}: {response_text[:100]}"
                        last_error_code = f"SLACK_HTTP_{response.status}"
                        logger.warning(
                            "Slack Web API client error (non-retryable)",
                            extra={
                                "correlation_id": str(correlation_id),
                                "status_code": response.status,
                                "attempt": attempt + 1,
                            },
                        )
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        return ModelSlackAlertResult(
                            success=False,
                            duration_ms=duration_ms,
                            correlation_id=correlation_id,
                            error=last_error,
                            error_code=last_error_code,
                            retry_count=retry_count,
                        )

                    elif response.status >= 500:
                        # 5xx server errors are retryable
                        response_text = await response.text()
                        last_error = f"HTTP {response.status}: {response_text[:100]}"
                        last_error_code = f"SLACK_HTTP_{response.status}"
                        logger.warning(
                            "Slack Web API server error (retryable)",
                            extra={
                                "correlation_id": str(correlation_id),
                                "status_code": response.status,
                                "attempt": attempt + 1,
                            },
                        )

                    elif response.status == 200:
                        # HTTP 200 - parse JSON response
                        try:
                            body = await response.json()
                        except (aiohttp.ContentTypeError, ValueError):
                            last_error = "Slack Web API returned non-JSON response"
                            last_error_code = "SLACK_API_ERROR"
                            logger.warning(
                                "Slack Web API returned non-JSON body for HTTP 200",
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "attempt": attempt + 1,
                                },
                            )
                            if attempt < self._max_retries:
                                backoff_index = min(
                                    attempt, len(self._retry_backoff) - 1
                                )
                                await asyncio.sleep(self._retry_backoff[backoff_index])
                                retry_count += 1
                                continue
                            break
                        if body.get("ok"):
                            thread_ts = body.get("ts")
                            logger.info(
                                "Slack alert delivered successfully via Web API",
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "duration_ms": round(duration_ms, 2),
                                    "retry_count": retry_count,
                                    "thread_ts": thread_ts,
                                    "mode": "web_api",
                                },
                            )
                            return ModelSlackAlertResult(
                                success=True,
                                duration_ms=duration_ms,
                                correlation_id=correlation_id,
                                retry_count=retry_count,
                                thread_ts=thread_ts,
                            )
                        else:
                            # Slack API returned ok=false
                            slack_error = body.get("error", "unknown_error")
                            if slack_error == "ratelimited":
                                last_error = "Slack rate limit (API)"
                                last_error_code = "SLACK_RATE_LIMITED"
                                logger.warning(
                                    "Slack Web API rate limited (ok=false)",
                                    extra={
                                        "correlation_id": str(correlation_id),
                                        "attempt": attempt + 1,
                                        "max_attempts": self._max_retries + 1,
                                    },
                                )
                            else:
                                last_error = f"Slack API error: {slack_error}"
                                last_error_code = "SLACK_API_ERROR"
                                logger.warning(
                                    "Slack Web API returned error",
                                    extra={
                                        "correlation_id": str(correlation_id),
                                        "slack_error": slack_error,
                                        "attempt": attempt + 1,
                                    },
                                )
                                # Non-retryable API errors (invalid_auth, channel_not_found, etc.)
                                if slack_error in {
                                    "invalid_auth",
                                    "not_authed",
                                    "account_inactive",
                                    "token_revoked",
                                    "channel_not_found",
                                    "not_in_channel",
                                    "is_archived",
                                    "msg_too_long",
                                    "no_text",
                                    "ekm_access_denied",
                                    "team_access_not_granted",
                                }:
                                    duration_ms = (
                                        time.perf_counter() - start_time
                                    ) * 1000
                                    return ModelSlackAlertResult(
                                        success=False,
                                        duration_ms=duration_ms,
                                        correlation_id=correlation_id,
                                        error=last_error,
                                        error_code=last_error_code,
                                        retry_count=retry_count,
                                    )

                    else:
                        # Unexpected status (1xx, 2xx non-200, 3xx)
                        last_error = f"Unexpected HTTP {response.status}"
                        last_error_code = f"SLACK_HTTP_{response.status}"
                        logger.warning(
                            "Slack Web API unexpected status code",
                            extra={
                                "correlation_id": str(correlation_id),
                                "status_code": response.status,
                                "attempt": attempt + 1,
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

            except TimeoutError:
                last_error = "Request timeout"
                last_error_code = "SLACK_TIMEOUT"
                logger.warning(
                    "Slack Web API timeout",
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
                    "Slack Web API connection error",
                    extra={
                        "correlation_id": str(correlation_id),
                        "attempt": attempt + 1,
                    },
                )

            except aiohttp.ClientError as e:
                last_error = sanitize_error_message(e)
                last_error_code = "SLACK_CLIENT_ERROR"
                logger.warning(
                    "Slack Web API client error",
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
                    "Retrying Slack Web API",
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
            "Slack Web API delivery failed after retries",
            extra={
                "correlation_id": str(correlation_id),
                "duration_ms": round(duration_ms, 2),
                "retry_count": retry_count,
                "error_code": last_error_code,
                "mode": "web_api",
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
