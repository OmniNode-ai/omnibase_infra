# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RuntimeLogEventBridge - Python logging.Handler that emits structured Kafka events.

Best-effort capture of ERROR/WARNING log records from allowlisted loggers.
Decoupled via asyncio.Queue to avoid blocking the logging thread.

Features:
    - 3-layer circular logging prevention (bridge logger excluded)
    - Queue-based decoupling (sync logging -> async emission)
    - Allowlist-only attachment (NOT root logger)
    - Rate limited: max 5 events per fingerprint per 5 minutes
    - Self-metrics: events_emitted, events_dropped, events_rate_limited

Feature flag: ENABLE_RUNTIME_LOG_BRIDGE (default off)

Related Tickets:
    - OMN-5521: Create RuntimeLogEventBridge
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.topic_constants import TOPIC_RUNTIME_ERROR
from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer

# Bridge's own logger -- must never be captured by the bridge itself
_bridge_logger = logging.getLogger(f"{__name__}._bridge")

# Rate limit: max 5 events per fingerprint per 5 minutes
_RATE_LIMIT_WINDOW_SECONDS = 300.0
_RATE_LIMIT_MAX_PER_WINDOW = 5

# Queue size for decoupling sync logging from async emission
_DEFAULT_QUEUE_SIZE = 1000

# Pattern to templatize log messages (replace numbers, UUIDs, IPs, etc.)
_TEMPLATIZE_PATTERNS = [
    (
        re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
        ),
        "{}",
    ),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "{}"),
    (re.compile(r"\b\d+\.\d+\b"), "{}"),
    (re.compile(r"\b\d+\b"), "{}"),
]

# Logger name prefix -> error category mapping
_CATEGORY_MAP: dict[str, EnumRuntimeErrorCategory] = {
    "aiokafka.consumer": EnumRuntimeErrorCategory.KAFKA_CONSUMER,
    "aiokafka.producer": EnumRuntimeErrorCategory.KAFKA_PRODUCER,
    "aiokafka": EnumRuntimeErrorCategory.KAFKA_CONSUMER,
    "asyncpg": EnumRuntimeErrorCategory.DATABASE,
    "aiohttp.client": EnumRuntimeErrorCategory.HTTP_CLIENT,
    "aiohttp.server": EnumRuntimeErrorCategory.HTTP_SERVER,
    "aiohttp": EnumRuntimeErrorCategory.HTTP_CLIENT,
    "uvicorn": EnumRuntimeErrorCategory.HTTP_SERVER,
}


def _templatize_message(msg: str) -> str:
    """Replace variable parts of a log message with placeholders.

    Args:
        msg: Raw log message.

    Returns:
        Templatized message suitable for fingerprinting.
    """
    result = msg
    for pattern, replacement in _TEMPLATIZE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _categorize_logger(logger_family: str) -> EnumRuntimeErrorCategory:
    """Map a logger name to an error category.

    Args:
        logger_family: Python logger name.

    Returns:
        Best-match error category.
    """
    for prefix, category in _CATEGORY_MAP.items():
        if logger_family.startswith(prefix):
            return category
    return EnumRuntimeErrorCategory.UNKNOWN


def _log_level_to_severity(levelno: int) -> EnumRuntimeErrorSeverity:
    """Map Python log level to error severity.

    Args:
        levelno: Python logging level number.

    Returns:
        Corresponding severity enum.
    """
    if levelno >= logging.CRITICAL:
        return EnumRuntimeErrorSeverity.CRITICAL
    if levelno >= logging.ERROR:
        return EnumRuntimeErrorSeverity.ERROR
    return EnumRuntimeErrorSeverity.WARNING


class RuntimeLogEventBridge(logging.Handler):
    """Python logging.Handler that captures log records as structured Kafka events.

    Attach to specific named loggers only (NOT root logger).
    Uses an asyncio.Queue for sync-to-async decoupling.

    Attributes:
        events_emitted: Count of successfully emitted events.
        events_dropped: Count of events that failed to emit or were queue-full.
        events_rate_limited: Count of events suppressed by rate limiter.
    """

    def __init__(
        self,
        producer: AIOKafkaProducer,
        *,
        topic: str = TOPIC_RUNTIME_ERROR,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        hostname: str = "",
        service_label: str = "",
    ) -> None:
        """Initialize the bridge.

        Args:
            producer: An already-started AIOKafkaProducer.
            topic: Topic to emit to.
            queue_size: Max queue size for decoupling.
            hostname: Machine hostname.
            service_label: Service display label.
        """
        super().__init__(level=logging.WARNING)
        self._producer = producer
        self._topic = topic
        self._hostname = hostname
        self._service_label = service_label
        self._queue: asyncio.Queue[ModelRuntimeErrorEvent] = asyncio.Queue(
            maxsize=queue_size
        )
        self._rate_limit_cache: dict[str, list[float]] = {}
        self._running = False
        self._drain_task: asyncio.Task[None] | None = None

        # Self-metrics
        self.events_emitted: int = 0
        self.events_dropped: int = 0
        self.events_rate_limited: int = 0

    @staticmethod
    def is_enabled() -> bool:
        """Check if runtime log bridge is enabled via feature flag."""
        return (
            os.environ.get(  # ONEX_FLAG_EXEMPT: declared in service-level contract (contracts/services/runtime.contract.yaml)
                "ENABLE_RUNTIME_LOG_BRIDGE", ""
            )
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Handle a log record (called by Python logging framework).

        Converts to ModelRuntimeErrorEvent and enqueues for async emission.
        Non-blocking -- drops if queue is full.

        Args:
            record: The log record to process.
        """
        if not self.is_enabled():
            return

        # Circular prevention: skip records from our own bridge logger
        if record.name.startswith(__name__):
            return

        try:
            raw_message = self.format(record) if record.msg else str(record.msg)
            message_template = _templatize_message(raw_message)
            logger_family = record.name
            error_category = _categorize_logger(logger_family)
            severity = _log_level_to_severity(record.levelno)

            # Rate limiting
            fingerprint = hashlib.sha256(
                f"{logger_family}:{error_category}:{message_template}".encode()
            ).hexdigest()[:16]

            if self._is_rate_limited(fingerprint):
                self.events_rate_limited += 1
                return

            # Extract exception info
            exception_type = ""
            exception_message = ""
            stack_trace = ""
            if record.exc_info and record.exc_info[1]:
                exc = record.exc_info[1]
                exception_type = type(exc).__name__
                exception_message = str(exc)[:500]
                if record.exc_text:
                    stack_trace = record.exc_text[:2000]

            event = ModelRuntimeErrorEvent.create(
                logger_family=logger_family,
                log_level=record.levelname,
                message_template=message_template,
                raw_message=raw_message[:2000],
                error_category=error_category,
                severity=severity,
                exception_type=exception_type,
                exception_message=exception_message,
                stack_trace=stack_trace,
                hostname=self._hostname,
                service_label=self._service_label,
            )

            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self.events_dropped += 1

        except Exception:  # noqa: BLE001 - must never raise from logging handler
            self.events_dropped += 1

    def _is_rate_limited(self, fingerprint: str) -> bool:
        """Check if an event with this fingerprint is rate-limited."""
        now = time.monotonic()
        timestamps = self._rate_limit_cache.get(fingerprint, [])

        # Remove expired entries
        timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW_SECONDS]

        if len(timestamps) >= _RATE_LIMIT_MAX_PER_WINDOW:
            return True

        timestamps.append(now)
        self._rate_limit_cache[fingerprint] = timestamps
        return False

    async def start(self) -> None:
        """Start the async drain loop."""
        self._running = True
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def stop(self) -> None:
        """Stop the drain loop and flush remaining events."""
        self._running = False
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

    async def _drain_loop(self) -> None:
        """Background task: drain queue and emit to Kafka."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._emit_to_kafka(event)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                # Flush remaining
                while not self._queue.empty():
                    try:
                        event = self._queue.get_nowait()
                        await self._emit_to_kafka(event)
                    except asyncio.QueueEmpty:
                        break
                raise

    async def _emit_to_kafka(self, event: ModelRuntimeErrorEvent) -> None:
        """Emit a single event to Kafka."""
        try:
            payload = json.dumps(
                event.model_dump(mode="json"),
            ).encode("utf-8")
            await self._producer.send(self._topic, value=payload)
            self.events_emitted += 1
        except Exception:  # noqa: BLE001 - best-effort emission
            self.events_dropped += 1
            _bridge_logger.debug(
                "Failed to emit runtime error event (fingerprint=%s)",
                event.fingerprint,
                exc_info=True,
            )

    def attach_to_loggers(self, logger_names: list[str]) -> None:
        """Attach this handler to the specified named loggers.

        Args:
            logger_names: List of logger names to attach to.
        """
        for name in logger_names:
            target_logger = logging.getLogger(name)
            target_logger.addHandler(self)
            _bridge_logger.debug("Attached RuntimeLogEventBridge to logger: %s", name)

    def detach_from_loggers(self, logger_names: list[str]) -> None:
        """Detach this handler from the specified named loggers.

        Args:
            logger_names: List of logger names to detach from.
        """
        for name in logger_names:
            target_logger = logging.getLogger(name)
            target_logger.removeHandler(self)


__all__ = ["RuntimeLogEventBridge"]
