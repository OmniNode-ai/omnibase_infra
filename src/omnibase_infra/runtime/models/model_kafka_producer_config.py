# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka producer configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_kafka_acks import EnumKafkaAcks


class ModelKafkaProducerConfig(BaseModel):
    """Kafka producer configuration.

    Sources configuration from KAFKA_* environment variables.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers",
    )
    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Connection timeout in seconds",
    )
    acks: EnumKafkaAcks = Field(
        default=EnumKafkaAcks.ALL,
        description="Producer acknowledgment policy",
    )

    @classmethod
    def from_env(cls) -> ModelKafkaProducerConfig:
        """Create config from KAFKA_* environment variables.

        Raises:
            ValueError: If numeric env vars contain non-numeric values
                or KAFKA_ACKS contains an invalid acks value.
        """
        try:
            return cls(
                bootstrap_servers=os.getenv(
                    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
                ),
                timeout_seconds=float(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "10000"))
                / 1000.0,
                acks=EnumKafkaAcks(os.getenv("KAFKA_ACKS", "all")),
            )
        except (ValueError, TypeError) as e:
            msg = f"Invalid Kafka producer configuration: {e}"
            raise ValueError(msg) from e


__all__ = ["ModelKafkaProducerConfig"]
