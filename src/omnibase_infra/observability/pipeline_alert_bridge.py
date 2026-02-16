# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pipeline Alert Bridge - Wires failure detection to human notification.

This module bridges the gap between the intelligence pipeline's failure
detection mechanisms and human-visible notifications via Slack. It connects:

1. DLQ events -> Slack alerts (via register_dlq_callback)
2. Wiring health degradation -> Slack alerts (via WiringHealthChecker)
3. Cold-start blocked -> Slack alerts (after configurable timeout)
4. Pipeline recovery -> Slack resolution notifications

All delivery infrastructure already exists (HandlerSlackWebhook, ModelSlackAlert,
register_dlq_callback, ModelWiringHealthAlert.to_slack_message). This module
wires these components together with rate limiting to prevent alert storms.

Architecture:
    +-----------------------+
    | DLQ Callback Hook     |---+
    +-----------------------+   |
    | Wiring Health Checker |---+---> PipelineAlertBridge ---> HandlerSlackWebhook
    +-----------------------+   |        (rate-limited)                   (Slack)
    | Cold-Start Monitor    |---+
    +-----------------------+   |
    | Recovery Detector     |---+
    +-----------------------+

Related Tickets:
    - OMN-2291: Intelligence pipeline resilience testing
    - OMN-1905: Slack webhook handler
    - OMN-1895: Wiring health monitoring

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from uuid import UUID, uuid4

from omnibase_infra.event_bus.models import ModelDlqEvent
from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
)
from omnibase_infra.observability.wiring_health.model_wiring_health_alert import (
    ModelWiringHealthAlert,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

logger = logging.getLogger(__name__)

# Rate limiting defaults
_DEFAULT_RATE_LIMIT_WINDOW_SECONDS: float = 300.0  # 5 minutes
_DEFAULT_MAX_ALERTS_PER_WINDOW: int = 5
_DEFAULT_COLD_START_TIMEOUT_SECONDS: float = 300.0  # 5 minutes


class PipelineAlertBridge:
    """Bridges pipeline failure detection to Slack notifications.

    Provides automatic alerting for:
    - DLQ events (messages that failed processing)
    - Wiring health degradation (topic mismatch threshold exceeded)
    - Cold-start blocked conditions (dependencies unavailable)
    - Recovery events (previously degraded pipeline returns healthy)

    Rate limiting prevents alert storms: at most ``max_alerts_per_window``
    alerts per ``rate_limit_window_seconds`` per alert category.

    .. note::

        **Threading constraint**: The ``asyncio.Lock`` instances used for rate
        limiting, health state, and cold-start tracking are bound to a single
        event loop. This class must be created and used within the same event
        loop. Do not share instances across threads running separate loops.

    Attributes:
        _handler: Slack webhook handler for delivery.
        _rate_limit_window: Window duration for rate limiting.
        _max_alerts_per_window: Max alerts allowed per window per category.
        _alert_timestamps: Per-category timestamps for rate limiting.
        _previous_health_state: Tracks last known health for recovery detection.
        _cold_start_alerted: Whether cold-start alert has been sent.

    Example:
        >>> bridge = PipelineAlertBridge(
        ...     slack_handler=HandlerSlackWebhook(webhook_url="https://..."),
        ...     environment="prod",
        ... )
        >>> # Register DLQ callback
        >>> unregister = await event_bus.register_dlq_callback(bridge.on_dlq_event)
        >>> # Check wiring health periodically
        >>> await bridge.on_wiring_health_check(metrics, alert)
    """

    def __init__(
        self,
        slack_handler: HandlerSlackWebhook,
        environment: str,
        rate_limit_window_seconds: float = _DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        max_alerts_per_window: int = _DEFAULT_MAX_ALERTS_PER_WINDOW,
        cold_start_timeout_seconds: float = _DEFAULT_COLD_START_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the pipeline alert bridge.

        Args:
            slack_handler: Configured HandlerSlackWebhook for Slack delivery.
            environment: Environment identifier (e.g., "prod", "dev").
            rate_limit_window_seconds: Window for rate limiting (default 300s).
            max_alerts_per_window: Max alerts per window per category (default 5).
            cold_start_timeout_seconds: How long to wait before alerting on
                blocked cold-start (default 300s / 5 minutes).
        """
        self._handler = slack_handler
        self._environment = environment
        self._rate_limit_window = rate_limit_window_seconds
        self._max_alerts_per_window = max_alerts_per_window
        self._cold_start_timeout = cold_start_timeout_seconds

        # Rate limiting: track timestamps per alert category
        self._alert_timestamps: dict[str, list[float]] = defaultdict(list)
        self._rate_limit_lock = asyncio.Lock()

        # Health state tracking for recovery detection
        self._previous_health_state: bool | None = None
        self._health_lock = asyncio.Lock()

        # Cold-start tracking
        self._cold_start_alerted = False
        self._cold_start_lock = asyncio.Lock()

        logger.info(
            "PipelineAlertBridge initialized",
            extra={
                "environment": environment,
                "rate_limit_window": rate_limit_window_seconds,
                "max_alerts_per_window": max_alerts_per_window,
                "cold_start_timeout": cold_start_timeout_seconds,
            },
        )

    async def on_dlq_event(self, event: ModelDlqEvent) -> None:
        """Handle DLQ callback by sending a Slack alert.

        Designed for use with ``event_bus.register_dlq_callback(bridge.on_dlq_event)``.
        Sends CRITICAL severity when the DLQ publish itself failed, WARNING otherwise.

        Args:
            event: The DLQ event containing failure context.
        """
        if not await self._check_rate_limit("dlq"):
            logger.debug(
                "DLQ alert suppressed by rate limit",
                extra={"correlation_id": str(event.correlation_id)},
            )
            return

        severity = (
            EnumAlertSeverity.CRITICAL
            if event.is_critical
            else EnumAlertSeverity.WARNING
        )

        if event.is_critical:
            safe_dlq_error = sanitize_error_string(event.dlq_error_message or "")
            title = "DLQ Publish Failed - Message May Be Lost"
            message = (
                f"*DLQ publish failed* for topic `{event.original_topic}`.\n"
                f"The original message could not be preserved in the DLQ "
                f"and may be permanently lost.\n\n"
                f"*Error*: {event.dlq_error_type}: {safe_dlq_error}"
            )
        else:
            safe_error = sanitize_error_string(event.error_message or "")
            title = "Message Routed to DLQ"
            message = (
                f"A message from topic `{event.original_topic}` failed "
                f"processing and was routed to DLQ `{event.dlq_topic}`.\n\n"
                f"*Error*: {event.error_type}: {safe_error}\n"
                f"*Retries exhausted*: {event.retry_count}"
            )

        alert = ModelSlackAlert(
            severity=severity,
            message=message,
            title=title,
            details={
                "Environment": self._environment,
                "Original Topic": event.original_topic,
                "Error Type": event.error_type,
                "Consumer Group": event.consumer_group,
                "Retry Count": str(event.retry_count),
            },
            correlation_id=event.correlation_id,
        )

        try:
            result = await self._handler.handle(alert)
        except Exception:
            logger.exception(
                "Alert delivery failed for category=dlq, suppressing to protect caller",
                extra={"correlation_id": str(event.correlation_id)},
            )
            return

        if result.success:
            logger.info(
                "DLQ Slack alert delivered",
                extra={
                    "correlation_id": str(event.correlation_id),
                    "severity": severity.value,
                },
            )
        else:
            logger.warning(
                "DLQ Slack alert delivery failed",
                extra={
                    "correlation_id": str(event.correlation_id),
                    "error": result.error,
                    "error_code": result.error_code,
                },
            )

    async def on_wiring_health_check(
        self,
        is_healthy: bool,
        alert: ModelWiringHealthAlert | None,
        correlation_id: UUID | None = None,
    ) -> None:
        """Handle wiring health check result and send alerts as needed.

        Sends a WARNING alert when health degrades, and an INFO resolution
        alert when health recovers.

        Args:
            is_healthy: Current overall health status.
            alert: ModelWiringHealthAlert if unhealthy, None if healthy.
            correlation_id: Optional correlation ID for tracing.
        """
        correlation_id = correlation_id or uuid4()

        async with self._health_lock:
            previous = self._previous_health_state
            self._previous_health_state = is_healthy

        # Case 1: Degraded (was healthy or unknown, now unhealthy)
        if not is_healthy and alert is not None:
            if not await self._check_rate_limit("wiring_health"):
                logger.debug(
                    "Wiring health alert suppressed by rate limit",
                    extra={"correlation_id": str(correlation_id)},
                )
                return

            slack_alert = ModelSlackAlert(
                severity=EnumAlertSeverity.WARNING,
                message=alert.summary,
                title=f"Wiring Health Degraded - {self._environment}",
                details={
                    "Environment": self._environment,
                    "Unhealthy Topics": ", ".join(alert.unhealthy_topics),
                    "Threshold": f"{alert.threshold:.1%}",
                },
                correlation_id=correlation_id,
            )

            try:
                result = await self._handler.handle(slack_alert)
            except Exception:
                logger.exception(
                    "Alert delivery failed for category=wiring_health, "
                    "suppressing to protect caller",
                    extra={"correlation_id": str(correlation_id)},
                )
                return

            if result.success:
                logger.info(
                    "Wiring health degradation alert delivered",
                    extra={"correlation_id": str(correlation_id)},
                )
            else:
                logger.warning(
                    "Wiring health alert delivery failed",
                    extra={
                        "correlation_id": str(correlation_id),
                        "error": result.error,
                    },
                )

        # Case 2: Recovery (was unhealthy, now healthy)
        elif is_healthy and previous is False:
            await self._send_recovery_alert(
                title="Wiring Health Recovered",
                message=(
                    f"Pipeline wiring health has returned to normal "
                    f"in *{self._environment}*. All monitored topics are "
                    f"within acceptable mismatch thresholds."
                ),
                correlation_id=correlation_id,
            )

    async def on_cold_start_blocked(
        self,
        dependency_name: str,
        elapsed_seconds: float,
        correlation_id: UUID | None = None,
    ) -> None:
        """Alert when cold-start bootstrap is blocked for too long.

        Should be called periodically during bootstrap when a required
        dependency is unavailable. Sends alert once after the configured
        timeout threshold is exceeded.

        Args:
            dependency_name: Name of the unavailable dependency (e.g., "PostgreSQL").
            elapsed_seconds: How long the dependency has been unavailable.
            correlation_id: Optional correlation ID for tracing.
        """
        correlation_id = correlation_id or uuid4()

        if elapsed_seconds < self._cold_start_timeout:
            return

        async with self._cold_start_lock:
            if self._cold_start_alerted:
                return

            if not await self._check_rate_limit("cold_start"):
                return

            alert = ModelSlackAlert(
                severity=EnumAlertSeverity.WARNING,
                message=(
                    f"Pipeline cold-start has been blocked for "
                    f"*{elapsed_seconds:.0f} seconds* waiting for "
                    f"`{dependency_name}` in *{self._environment}*.\n\n"
                    f"The pipeline cannot start until this dependency "
                    f"becomes available. Check service health and network "
                    f"connectivity."
                ),
                title=f"Pipeline Cold-Start Blocked - {self._environment}",
                details={
                    "Environment": self._environment,
                    "Blocked Dependency": dependency_name,
                    "Elapsed Seconds": f"{elapsed_seconds:.0f}",
                    "Threshold Seconds": f"{self._cold_start_timeout:.0f}",
                },
                correlation_id=correlation_id,
            )

            try:
                result = await self._handler.handle(alert)
            except Exception:
                logger.exception(
                    "Alert delivery failed for category=cold_start, "
                    "suppressing to protect caller",
                    extra={"correlation_id": str(correlation_id)},
                )
                return

            if result.success:
                self._cold_start_alerted = True
                logger.info(
                    "Cold-start blocked alert delivered",
                    extra={
                        "correlation_id": str(correlation_id),
                        "dependency": dependency_name,
                        "elapsed_seconds": elapsed_seconds,
                    },
                )
            else:
                logger.warning(
                    "Cold-start blocked alert delivery failed",
                    extra={
                        "correlation_id": str(correlation_id),
                        "error": result.error,
                    },
                )

    async def on_cold_start_resolved(
        self,
        dependency_name: str,
        correlation_id: UUID | None = None,
    ) -> None:
        """Alert when a previously blocked cold-start dependency becomes available.

        Only sends a recovery alert if a cold-start blocked alert was previously
        sent. Resets the cold-start alerted flag so future blocks can be detected.

        Args:
            dependency_name: Name of the now-available dependency.
            correlation_id: Optional correlation ID for tracing.
        """
        async with self._cold_start_lock:
            if not self._cold_start_alerted:
                return
            self._cold_start_alerted = False

        await self._send_recovery_alert(
            title=f"Cold-Start Dependency Resolved - {self._environment}",
            message=(
                f"Dependency `{dependency_name}` is now available in "
                f"*{self._environment}*. Pipeline bootstrap can proceed."
            ),
            correlation_id=correlation_id or uuid4(),
        )

    async def _send_recovery_alert(
        self,
        title: str,
        message: str,
        correlation_id: UUID,
    ) -> None:
        """Send an INFO-level recovery notification.

        Args:
            title: Alert title.
            message: Alert message body.
            correlation_id: Correlation ID for tracing.
        """
        if not await self._check_rate_limit("recovery"):
            logger.debug(
                "Recovery alert suppressed by rate limit",
                extra={"correlation_id": str(correlation_id)},
            )
            return

        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.INFO,
            message=message,
            title=title,
            details={
                "Environment": self._environment,
                "Status": "Recovered",
            },
            correlation_id=correlation_id,
        )

        try:
            result = await self._handler.handle(alert)
        except Exception:
            logger.exception(
                "Alert delivery failed for category=recovery, "
                "suppressing to protect caller",
                extra={"correlation_id": str(correlation_id)},
            )
            return

        if result.success:
            logger.info(
                "Recovery alert delivered",
                extra={
                    "correlation_id": str(correlation_id),
                    "title": title,
                },
            )
        else:
            logger.warning(
                "Recovery alert delivery failed",
                extra={
                    "correlation_id": str(correlation_id),
                    "error": result.error,
                },
            )

    async def _check_rate_limit(self, category: str) -> bool:
        """Check if an alert for the given category is allowed by rate limiting.

        Uses a sliding window approach: timestamps older than the window are
        pruned, and a new alert is allowed only if the count within the window
        is below the configured maximum.

        Args:
            category: Alert category (e.g., "dlq", "wiring_health", "cold_start").

        Returns:
            True if the alert is allowed, False if rate-limited.
        """
        now = time.monotonic()

        async with self._rate_limit_lock:
            timestamps = self._alert_timestamps[category]

            # Remove timestamps outside the window
            cutoff = now - self._rate_limit_window
            self._alert_timestamps[category] = [ts for ts in timestamps if ts > cutoff]
            timestamps = self._alert_timestamps[category]

            # Check if we're at the limit
            if len(timestamps) >= self._max_alerts_per_window:
                return False

            # Record this alert
            timestamps.append(now)
            return True


__all__: list[str] = ["PipelineAlertBridge"]
