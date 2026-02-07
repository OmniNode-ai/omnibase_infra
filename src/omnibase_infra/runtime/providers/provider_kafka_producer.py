# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka producer provider.

Creates AIOKafkaProducer instances from environment-driven configuration.
Respects architecture constraint #7: Never block on Kafka.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)

logger = logging.getLogger(__name__)


class ProviderKafkaProducer:
    """Creates and manages Kafka producers.

    Producers are created from KAFKA_* environment variables and shared
    across all contracts that declare kafka_producer dependencies.

    Per architecture constraint #7: Kafka is optional. Creation failures
    are logged but do not block startup.
    """

    def __init__(self, config: ModelKafkaProducerConfig) -> None:
        """Initialize the Kafka producer provider.

        Args:
            config: Kafka producer configuration (bootstrap servers, acks, timeout).
        """
        self._config = config

    # ONEX_EXCLUDE: any_type - returns AIOKafkaProducer which varies by runtime
    async def create(self) -> Any:
        """Create and start a Kafka producer.

        Returns:
            AIOKafkaProducer instance.

        Raises:
            Exception: If producer creation or start fails.
        """
        from aiokafka import AIOKafkaProducer

        logger.info(
            "Creating Kafka producer",
            extra={
                "bootstrap_servers": self._config.bootstrap_servers,
                "timeout_seconds": self._config.timeout_seconds,
            },
        )

        producer = AIOKafkaProducer(
            bootstrap_servers=self._config.bootstrap_servers,
            acks=self._config.acks.to_aiokafka(),
        )

        try:
            await asyncio.wait_for(
                producer.start(),
                timeout=self._config.timeout_seconds,
            )
        except TimeoutError:
            # Best-effort cleanup to prevent resource leak
            try:
                await producer.stop()
            except Exception:
                pass
            raise

        logger.info("Kafka producer created and started successfully")
        return producer

    @staticmethod
    # ONEX_EXCLUDE: any_type - resource is AIOKafkaProducer, typed as Any for provider interface
    async def close(resource: Any) -> None:
        """Stop a Kafka producer.

        Args:
            resource: The AIOKafkaProducer to stop.
        """
        if resource is not None and hasattr(resource, "stop"):
            await resource.stop()
            logger.info("Kafka producer stopped")


__all__ = ["ProviderKafkaProducer"]
