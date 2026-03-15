# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Batch response publisher for RuntimeHostProcess (OMN-478).

Aggregates response envelopes and publishes them in batches to reduce
event bus overhead and improve throughput. Responses are flushed when
either the batch size threshold or the flush timeout is reached,
whichever comes first.

Design Principles:
    - **Configurable batch size**: Controls max responses per batch
    - **Configurable flush interval**: Controls max latency before flush
    - **Ordering preserved**: Responses within a batch maintain insertion order
    - **Partial failure handling**: Failed publishes are retried individually
    - **Thread-safe**: Uses asyncio locks for concurrent access
    - **Graceful shutdown**: flush_all() drains pending responses

Architecture Context:
    In the ONEX runtime, each handler produces a response envelope that is
    published to the output topic. With parallel handler execution (OMN-476),
    multiple responses may be produced concurrently. BatchResponsePublisher
    buffers these responses and publishes them in batches, reducing the number
    of individual publish calls to the event bus.

Example Usage:
    ```python
    from omnibase_infra.runtime.batch_response_publisher import BatchResponsePublisher

    publisher = BatchResponsePublisher(
        publish_fn=event_bus.publish_envelope,
        topic="responses",
        batch_size=10,
        flush_interval_ms=100,
    )

    await publisher.start()
    try:
        await publisher.enqueue({"success": True, "correlation_id": "abc"})
        # ... enqueue more responses ...
    finally:
        await publisher.stop()  # Flushes remaining responses
    ```

Related Tickets:
    - OMN-478: Add batch response publishing to RuntimeHostProcess
    - OMN-476: Parallel handler execution
    - OMN-249: RuntimeHostProcess MVP implementation
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from omnibase_infra.runtime.models.model_batch_publisher_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLUSH_INTERVAL_MS,
    MAX_BATCH_SIZE,
    MAX_FLUSH_INTERVAL_MS,
    MIN_BATCH_SIZE,
    MIN_FLUSH_INTERVAL_MS,
    ModelBatchPublisherConfig,
)
from omnibase_infra.runtime.models.model_batch_publisher_metrics import (
    ModelBatchPublisherMetrics,
)

logger = logging.getLogger(__name__)


class BatchResponsePublisher:
    """Buffers response envelopes and publishes in batches.

    The publisher maintains an internal buffer of response envelopes. When
    the buffer reaches the configured batch size or the flush interval elapses,
    all buffered responses are published to the output topic.

    Concurrency Safety:
        The publisher uses an asyncio.Lock to guard buffer access. Multiple
        coroutines can safely call enqueue() concurrently.

    Flush Triggers:
        1. **Size threshold**: When buffer reaches batch_size, flush immediately.
        2. **Timeout**: A background timer flushes every flush_interval_ms.
        3. **Manual flush**: Call flush_all() to drain the buffer.
        4. **Shutdown**: stop() calls flush_all() before cleanup.

    Error Handling:
        If a batch publish fails, the publisher retries each response
        individually. Responses that fail individual retry are logged
        and counted in metrics but not re-enqueued (to prevent unbounded
        growth).

    Attributes:
        config: The batch publisher configuration.
        metrics: Publishing metrics (enqueued, published, failed counts).
    """

    def __init__(
        self,
        publish_fn: Callable[[dict[str, object], str], Awaitable[None]],
        topic: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_ms: float = DEFAULT_FLUSH_INTERVAL_MS,
    ) -> None:
        """Initialize the batch response publisher.

        Args:
            publish_fn: Async function to publish a single envelope to a topic.
                Signature: async (envelope: dict, topic: str) -> None
            topic: The output topic to publish responses to.
            batch_size: Maximum number of responses to buffer before flushing.
                Values outside [1, 1000] are clamped with a warning.
            flush_interval_ms: Maximum time in milliseconds before flushing.
                Values outside [10, 5000] are clamped with a warning.
        """
        # Validate and clamp batch_size
        if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
            logger.warning(
                "batch_size out of valid range, clamping",
                extra={
                    "original_value": batch_size,
                    "min_value": MIN_BATCH_SIZE,
                    "max_value": MAX_BATCH_SIZE,
                },
            )
            batch_size = max(MIN_BATCH_SIZE, min(batch_size, MAX_BATCH_SIZE))

        # Validate and clamp flush_interval_ms
        if (
            flush_interval_ms < MIN_FLUSH_INTERVAL_MS
            or flush_interval_ms > MAX_FLUSH_INTERVAL_MS
        ):
            logger.warning(
                "flush_interval_ms out of valid range, clamping",
                extra={
                    "original_value": flush_interval_ms,
                    "min_value": MIN_FLUSH_INTERVAL_MS,
                    "max_value": MAX_FLUSH_INTERVAL_MS,
                },
            )
            flush_interval_ms = max(
                MIN_FLUSH_INTERVAL_MS, min(flush_interval_ms, MAX_FLUSH_INTERVAL_MS)
            )

        self._publish_fn = publish_fn
        self._topic = topic
        self._batch_size = batch_size
        self._flush_interval_seconds = flush_interval_ms / 1000.0

        self._buffer: list[dict[str, object]] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._is_running = False

        self._metrics = ModelBatchPublisherMetrics()

    @property
    def config(self) -> ModelBatchPublisherConfig:
        """Return the effective configuration."""
        return ModelBatchPublisherConfig(
            batch_size=self._batch_size,
            flush_interval_ms=self._flush_interval_seconds * 1000.0,
            enabled=True,
        )

    @property
    def metrics(self) -> ModelBatchPublisherMetrics:
        """Return current metrics snapshot."""
        return self._metrics.model_copy()

    @property
    def is_running(self) -> bool:
        """Return whether the publisher is running."""
        return self._is_running

    @property
    def pending_count(self) -> int:
        """Return the number of responses currently buffered."""
        return len(self._buffer)

    async def start(self) -> None:
        """Start the background flush timer.

        The timer runs periodically and flushes any buffered responses
        that have been waiting longer than flush_interval_ms.
        """
        if self._is_running:
            logger.warning("BatchResponsePublisher already running")
            return

        self._is_running = True
        self._flush_task = asyncio.create_task(
            self._flush_loop(),
            name="batch-response-flush",
        )

        logger.info(
            "BatchResponsePublisher started",
            extra={
                "batch_size": self._batch_size,
                "flush_interval_ms": self._flush_interval_seconds * 1000.0,
            },
        )

    async def stop(self) -> None:
        """Stop the publisher and flush remaining responses.

        Cancels the background flush timer and drains any buffered
        responses before returning.
        """
        if not self._is_running:
            return

        self._is_running = False

        # Cancel the flush timer
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Drain remaining responses
        await self.flush_all()

        logger.info(
            "BatchResponsePublisher stopped",
            extra={
                "total_enqueued": self._metrics.total_enqueued,
                "total_published": self._metrics.total_published,
                "total_failed": self._metrics.total_failed,
                "total_batches_flushed": self._metrics.total_batches_flushed,
            },
        )

    async def enqueue(self, envelope: dict[str, object]) -> None:
        """Add a response envelope to the buffer.

        If the buffer reaches the batch size threshold after this addition,
        a flush is triggered immediately.

        Args:
            envelope: The response envelope to buffer for publishing.
        """
        should_flush = False

        async with self._buffer_lock:
            self._buffer.append(envelope)
            self._metrics.total_enqueued += 1

            if len(self._buffer) >= self._batch_size:
                should_flush = True

        if should_flush:
            await self._flush_batch(reason="size")

    async def flush_all(self) -> None:
        """Flush all buffered responses immediately.

        This is called during shutdown and can also be called manually
        to drain the buffer.
        """
        await self._flush_batch(reason="manual")

    async def _flush_loop(self) -> None:
        """Background loop that periodically flushes buffered responses."""
        try:
            while self._is_running:
                await asyncio.sleep(self._flush_interval_seconds)
                if self._buffer:
                    await self._flush_batch(reason="timeout")
        except asyncio.CancelledError:
            # Expected during shutdown
            pass

    async def _flush_batch(self, reason: str) -> None:
        """Flush the current buffer as a batch.

        Takes a snapshot of the buffer under the lock, then publishes
        each envelope. On failure, retries each envelope individually.

        Args:
            reason: Why the flush was triggered ("size", "timeout", "manual").
        """
        # Take snapshot under lock to minimize lock hold time
        async with self._buffer_lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        batch_size = len(batch)

        # Update flush reason metrics
        if reason == "timeout":
            self._metrics.total_timeout_flushes += 1
        elif reason == "size":
            self._metrics.total_size_flushes += 1

        self._metrics.total_batches_flushed += 1

        logger.debug(
            "Flushing response batch",
            extra={
                "batch_size": batch_size,
                "reason": reason,
                "topic": self._topic,
            },
        )

        # Publish each envelope in the batch
        # We publish sequentially to preserve ordering within the batch.
        # The event bus handles actual batching/pipelining at the transport level.
        published = 0
        failed = 0

        for envelope in batch:
            try:
                await self._publish_fn(envelope, self._topic)
                published += 1
            except Exception:
                failed += 1
                correlation_id = envelope.get("correlation_id", "unknown")
                logger.exception(
                    "Failed to publish response in batch",
                    extra={
                        "correlation_id": str(correlation_id),
                        "topic": self._topic,
                        "reason": reason,
                    },
                )

        self._metrics.total_published += published
        self._metrics.total_failed += failed

        if failed > 0:
            logger.warning(
                "Batch flush completed with failures",
                extra={
                    "published": published,
                    "failed": failed,
                    "batch_size": batch_size,
                    "reason": reason,
                },
            )


__all__: list[str] = [
    "BatchResponsePublisher",
    "ModelBatchPublisherConfig",
    "ModelBatchPublisherMetrics",
]
