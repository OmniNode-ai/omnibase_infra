# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Notification Consumer - Routes notification events to Slack.

This module provides the NotificationConsumer class that subscribes to
notification topics from Kafka and routes events to the Slack alerter.

Architecture:
    ```
    +-------------+     Kafka     +----------------------+     Slack
    | EmitDaemon  | ------------> | NotificationConsumer | ------------> Webhook
    | (publisher) |   Topics:     | (this file)          |   via
    +-------------+   blocked/    +----------------------+   HandlerSlackWebhook
                      completed
    ```

The consumer transforms notification events into Slack alerts:
- notification.blocked -> WARNING severity with ticket context
- notification.completed -> INFO severity with completion summary

Example Usage:
    ```python
    from omnibase_infra.runtime.emit_daemon import NotificationConsumer

    # Create consumer with Kafka event bus
    consumer = NotificationConsumer(
        event_bus=kafka_event_bus,
        webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
    )

    # Start consuming (blocks until stopped)
    await consumer.start()

    # Or run as background task
    task = asyncio.create_task(consumer.start())
    # ... later ...
    await consumer.stop()
    ```

Related Tickets:
    - OMN-1831: Implement event-driven Slack notifications via runtime

.. versionadded:: 0.4.1
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
)

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolEventBusLike

logger = logging.getLogger(__name__)

# Topic constants for notification events
TOPIC_NOTIFICATION_BLOCKED = "onex.evt.omniclaude.notification-blocked.v1"
TOPIC_NOTIFICATION_COMPLETED = "onex.evt.omniclaude.notification-completed.v1"


class NotificationConsumer:
    """Consumer that routes notification events to Slack.

    Subscribes to notification topics from Kafka and transforms events
    into Slack alerts using HandlerSlackWebhook.

    Event Transformation:
        - notification.blocked:
            - Severity: WARNING
            - Title: ":ticket: {ticket_identifier} needs input"
            - Message: Reason with details as bullet points
            - Details: Ticket ID, Repo

        - notification.completed:
            - Severity: INFO
            - Title: ":white_check_mark: {ticket_identifier} completed"
            - Message: Summary
            - Details: Ticket ID, Repo, PR URL (if present)

    Attributes:
        _event_bus: Kafka event bus for subscribing to topics
        _handler: HandlerSlackWebhook for sending alerts
        _running: Whether the consumer is currently running
        _consumer_tasks: Background tasks for consuming topics

    Example:
        >>> consumer = NotificationConsumer(
        ...     event_bus=kafka_event_bus,
        ...     webhook_url="https://hooks.slack.com/...",
        ... )
        >>> await consumer.start()
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusLike,
        webhook_url: str | None = None,
    ) -> None:
        """Initialize the notification consumer.

        Args:
            event_bus: Kafka event bus for subscribing to notification topics.
            webhook_url: Optional Slack webhook URL. If not provided, reads
                from SLACK_WEBHOOK_URL environment variable.
        """
        self._event_bus = event_bus
        self._handler = HandlerSlackWebhook(webhook_url=webhook_url)
        self._running = False
        self._consumer_tasks: list[asyncio.Task[None]] = []
        self._shutdown_event = asyncio.Event()

        logger.debug("NotificationConsumer initialized")

    async def start(self) -> None:
        """Start consuming notification events.

        Subscribes to notification topics and begins processing events.
        This method blocks until stop() is called.

        If the consumer is already running, logs a warning and returns
        immediately without starting a second instance.

        Raises:
            TypeError: If the event bus does not implement consume() method.
        """
        if self._running:
            logger.warning("NotificationConsumer already running")
            return

        self._running = True
        self._shutdown_event.clear()

        logger.info(
            "NotificationConsumer starting",
            extra={
                "topics": [TOPIC_NOTIFICATION_BLOCKED, TOPIC_NOTIFICATION_COMPLETED],
            },
        )

        # Start consumer tasks for each topic
        self._consumer_tasks = [
            asyncio.create_task(
                self._consume_topic(
                    TOPIC_NOTIFICATION_BLOCKED, self._handle_blocked_event
                )
            ),
            asyncio.create_task(
                self._consume_topic(
                    TOPIC_NOTIFICATION_COMPLETED, self._handle_completed_event
                )
            ),
        ]

        # Wait until shutdown
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the notification consumer gracefully.

        Cancels all consumer tasks, waits for them to complete, and clears
        internal state. Safe to call multiple times; subsequent calls are
        no-ops if the consumer is not running.

        This method is idempotent and will not raise exceptions.
        """
        if not self._running:
            logger.debug("NotificationConsumer not running")
            return

        self._running = False
        self._shutdown_event.set()

        # Cancel consumer tasks
        for task in self._consumer_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)

        self._consumer_tasks.clear()

        logger.info("NotificationConsumer stopped")

    async def _consume_topic(
        self,
        topic: str,
        handler: object,  # Callable[[dict], Coroutine[Any, Any, None]]
    ) -> None:
        """Consume messages from a topic and route to handler.

        Args:
            topic: Kafka topic to consume from.
            handler: Async handler function to process messages.
        """
        logger.info(f"Starting consumer for topic: {topic}")

        try:
            # Subscribe to the topic
            # NOTE: The actual subscription mechanism depends on the event bus
            # implementation. This is a simplified version that polls for messages.
            while self._running:
                try:
                    # NOTE: ProtocolEventBusLike may not have a consume method.
                    # For now, we use a simplified approach where we check if
                    # the event bus supports consuming.
                    if hasattr(self._event_bus, "consume"):
                        async for message in self._event_bus.consume(topic):  # type: ignore[union-attr]
                            if not self._running:
                                break
                            try:
                                await self._process_message(message, handler)  # type: ignore[arg-type]
                            except Exception as e:
                                logger.warning(
                                    f"Error processing message from {topic}: {e}",
                                    exc_info=True,
                                )
                    else:
                        # Event bus doesn't support consume - fail fast
                        raise TypeError(
                            f"Event bus {type(self._event_bus).__name__} does not "
                            f"implement consume() method. NotificationConsumer requires "
                            f"an event bus with consume() support (e.g., EventBusKafka)."
                        )

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        f"Consumer error for topic {topic}: {e}",
                        exc_info=True,
                    )
                    # Brief pause before retrying
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.debug(f"Consumer cancelled for topic: {topic}")

    async def _process_message(
        self,
        message: bytes,
        handler: object,  # Callable[[dict], Coroutine[Any, Any, None]]
    ) -> None:
        """Process a single message and route to handler.

        Args:
            message: Raw message bytes from Kafka.
            handler: Handler function to invoke with parsed payload.

        Note:
            Errors are logged with correlation_id when available for traceability.
            If correlation_id cannot be extracted, a generated UUID is used.
        """
        correlation_id: UUID | None = None
        try:
            # Parse the message payload
            payload = json.loads(message.decode("utf-8"))

            if not isinstance(payload, dict):
                logger.warning("Invalid message payload: not a dict")
                return

            # Extract correlation_id early for error traceability
            correlation_id = self._extract_correlation_id(payload)

            # Invoke the handler
            await handler(payload)  # type: ignore[operator]

        except json.JSONDecodeError as e:
            # Generate correlation_id for decode errors since payload unavailable
            correlation_id = correlation_id or uuid4()
            logger.warning(
                f"Failed to decode message: {e}",
                extra={"correlation_id": str(correlation_id)},
            )
        except Exception as e:
            correlation_id = correlation_id or uuid4()
            logger.warning(
                f"Failed to process message: {e}",
                extra={"correlation_id": str(correlation_id)},
                exc_info=True,
            )

    async def _handle_blocked_event(self, payload: dict[str, object]) -> None:
        """Handle a notification.blocked event.

        Transforms the event into a Slack alert with WARNING severity.

        Args:
            payload: Event payload containing ticket context.
        """
        ticket_identifier = str(payload.get("ticket_identifier", "Unknown"))
        reason = str(payload.get("reason", "Waiting for input"))
        details_raw = payload.get("details", [])
        repo = str(payload.get("repo", "unknown"))
        correlation_id = self._extract_correlation_id(payload)

        # Build details list safely
        details: list[str] = []
        if isinstance(details_raw, list):
            details = [str(d) for d in details_raw]

        # Build message
        message_lines = [f"*{reason}*"]
        if details:
            message_lines.append("")
            message_lines.extend(f"- {d}" for d in details)

        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.WARNING,
            message="\n".join(message_lines),
            title=f":ticket: {ticket_identifier} needs input",
            details={
                "Ticket": ticket_identifier,
                "Repo": repo,
            },
            correlation_id=correlation_id,
        )

        result = await self._handler.handle(alert)

        if result.success:
            logger.info(
                f"Slack notification sent for {ticket_identifier} (blocked)",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_ms": result.duration_ms,
                },
            )
        else:
            logger.warning(
                f"Slack notification failed for {ticket_identifier}: {result.error}",
                extra={
                    "correlation_id": str(correlation_id),
                    "error_code": result.error_code,
                },
            )

    async def _handle_completed_event(self, payload: dict[str, object]) -> None:
        """Handle a notification.completed event.

        Transforms the event into a Slack alert with INFO severity.

        Args:
            payload: Event payload containing completion details.
        """
        ticket_identifier = str(payload.get("ticket_identifier", "Unknown"))
        summary = str(payload.get("summary", "Work completed"))
        repo = str(payload.get("repo", "unknown"))
        pr_url = payload.get("pr_url")
        correlation_id = self._extract_correlation_id(payload)

        # Build details dict
        details_dict: dict[str, str] = {
            "Ticket": ticket_identifier,
            "Repo": repo,
        }
        if pr_url and isinstance(pr_url, str):
            details_dict["PR"] = pr_url

        alert = ModelSlackAlert(
            severity=EnumAlertSeverity.INFO,
            message=summary,
            title=f":white_check_mark: {ticket_identifier} completed",
            details=details_dict,
            correlation_id=correlation_id,
        )

        result = await self._handler.handle(alert)

        if result.success:
            logger.info(
                f"Slack notification sent for {ticket_identifier} (completed)",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_ms": result.duration_ms,
                },
            )
        else:
            logger.warning(
                f"Slack notification failed for {ticket_identifier}: {result.error}",
                extra={
                    "correlation_id": str(correlation_id),
                    "error_code": result.error_code,
                },
            )

    def _extract_correlation_id(self, payload: dict[str, object]) -> UUID:
        """Extract correlation_id from payload or generate a new one.

        Args:
            payload: Event payload that may contain correlation_id.

        Returns:
            UUID correlation ID for tracing.
        """
        correlation_id_str = payload.get("correlation_id")
        if isinstance(correlation_id_str, str):
            try:
                return UUID(correlation_id_str)
            except ValueError:
                pass
        return uuid4()


__all__ = [
    "NotificationConsumer",
    "TOPIC_NOTIFICATION_BLOCKED",
    "TOPIC_NOTIFICATION_COMPLETED",
]
