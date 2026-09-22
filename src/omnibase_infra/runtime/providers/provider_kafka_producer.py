# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka producer provider.

Creates AIOKafkaProducer instances from environment-driven configuration.
Respects platform-wide rule #8: Kafka is required infrastructure —
use async/non-blocking patterns.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env
from omnibase_infra.event_bus.kafka_connect_retry import connect_with_bounded_retry
from omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy import (
    ModelKafkaConnectRetryPolicy,
)
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer


class ProviderKafkaProducer:
    """Creates and manages Kafka producers.

    Producers are created from KAFKA_* environment variables and shared
    across all contracts that declare kafka_producer dependencies.

    Per platform-wide rule #8: Kafka is required infrastructure.
    Creation failures propagate to the caller — callers must treat them
    as fatal. Uses async patterns to avoid blocking the calling thread.
    """

    def __init__(self, config: ModelKafkaProducerConfig) -> None:
        """Initialize the Kafka producer provider.

        Args:
            config: Kafka producer configuration (bootstrap servers, acks, timeout).
        """
        self._config = config

    async def create(self) -> AIOKafkaProducer:
        """Create and start a Kafka producer, retrying a failed connect.

        OMN-18925: this used to be a single ``asyncio.wait_for`` around
        ``producer.start()``. A broker that stalls for longer than the
        per-attempt deadline — measured at 17.2s against a 10s deadline on
        2026-09-21 — was therefore unsurvivable by construction, and no
        setting of the deadline fixes that without also making a genuinely
        dead broker take that long to report. A second attempt does fix it,
        and the retry shape is shared with the event-bus producer so the two
        surfaces cannot drift apart (AC-3).

        Returns:
            AIOKafkaProducer instance.

        Raises:
            Exception: The final attempt's own failure, unchanged, once the
                policy's attempts are exhausted. Per platform-wide rule #8
                callers must still treat this as fatal; it now means "the
                broker did not answer across the whole bound" rather than
                "the broker did not answer once".
        """
        from aiokafka import AIOKafkaProducer

        # The retry dials come from the overlay-resolved event-bus config,
        # which is the one place they are declared; only the per-attempt
        # deadline is this surface's own (10s here against the bus's 30s).
        policy = ModelKafkaConnectRetryPolicy.from_bus_config(
            ModelKafkaEventBusConfig.default(),
            attempt_timeout_seconds=self._config.timeout_seconds,
        )

        logger.info(
            "Creating Kafka producer",
            extra={
                "bootstrap_servers": self._config.bootstrap_servers,
                "timeout_seconds": self._config.timeout_seconds,
                "connect_attempts": policy.total_attempts,
                "connect_bound_seconds": policy.total_bound_seconds,
            },
        )

        # Rebuilt per attempt rather than reused: aiokafka does not document a
        # producer whose start() failed as restartable, and a retry loop that
        # re-starts a poisoned client tests a different thing from the one it
        # claims to.
        producer: AIOKafkaProducer | None = None

        async def _connect() -> None:
            nonlocal producer
            producer = AIOKafkaProducer(
                bootstrap_servers=self._config.bootstrap_servers,
                acks=self._config.acks.to_aiokafka(),
                max_request_size=self._config.max_request_size,
                **build_aiokafka_auth_kwargs_from_env(),
            )
            await producer.start()

        async def _cleanup() -> None:
            nonlocal producer
            if producer is not None:
                current, producer = producer, None
                await current.stop()

        attempt = await connect_with_bounded_retry(
            policy=policy,
            connect=_connect,
            cleanup=_cleanup,
            target=self._config.bootstrap_servers,
        )

        if producer is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "Kafka connect reported success but left no producer; "
                f"attempt={attempt}"
            )

        logger.info(
            "Kafka producer created and started successfully",
            extra={"connect_attempt": attempt},
        )
        return producer

    @staticmethod
    async def close(resource: AIOKafkaProducer | None) -> None:
        """Stop a Kafka producer.

        Args:
            resource: The AIOKafkaProducer to stop.
        """
        if resource is not None and hasattr(resource, "stop"):
            await resource.stop()
            logger.info("Kafka producer stopped")


__all__ = ["ProviderKafkaProducer"]
