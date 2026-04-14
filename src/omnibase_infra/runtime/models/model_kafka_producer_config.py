# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
        default_factory=lambda: os.environ["KAFKA_BOOTSTRAP_SERVERS"],
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
    max_request_size: int = Field(
        default=4 * 1024 * 1024,  # 4 MB
        ge=1024,
        le=52428800,
        description=(
            "Maximum size in bytes for a Kafka produce request. "
            "Passed to AIOKafkaProducer(max_request_size=...). "
            "Override via KAFKA_MAX_REQUEST_SIZE env var."
        ),
    )

    @classmethod
    def from_env(cls) -> ModelKafkaProducerConfig:
        """Create config from KAFKA_* environment variables.

        Raises:
            KeyError: If KAFKA_BOOTSTRAP_SERVERS is not set in the environment.
                Containers must always have this injected via compose overlay —
                no silent localhost fallback (OMN-8783).
            ValueError: If numeric env vars contain non-numeric values
                or KAFKA_ACKS contains an invalid acks value.
        """
        # OMN-8783: Hard-fail if KAFKA_BOOTSTRAP_SERVERS is absent. Containers
        # receive this via hardcoded_env in the catalog manifest (redpanda:9092).
        # A missing env var means the overlay was not applied — fail loudly.
        bootstrap = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
        try:
            timeout_ms = os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "10000")
            acks_raw = os.getenv("KAFKA_ACKS", "all")
            max_request_size_raw = os.getenv(
                "KAFKA_MAX_REQUEST_SIZE", str(4 * 1024 * 1024)
            )
            return cls(
                bootstrap_servers=bootstrap,
                timeout_seconds=float(timeout_ms) / 1000.0,
                acks=EnumKafkaAcks(acks_raw),
                max_request_size=int(max_request_size_raw),
            )
        except (ValueError, TypeError) as e:
            msg = f"Invalid Kafka producer configuration: {e}"
            raise ValueError(msg) from e


__all__ = ["ModelKafkaProducerConfig"]
