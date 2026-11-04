# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka resource acquisition within the event-bus wiring effect."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer


class AdapterRegistrationSnapshotProducer:
    """Acquire the authenticated producer used by registration snapshots."""

    @staticmethod
    def create(bootstrap_servers: str | None) -> AIOKafkaProducer:
        """Construct the producer with the existing Kafka auth boundary."""
        from aiokafka import AIOKafkaProducer

        from omnibase_infra.event_bus.kafka_auth import (
            build_aiokafka_auth_kwargs_from_env,
        )

        return AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            **build_aiokafka_auth_kwargs_from_env(),
        )
