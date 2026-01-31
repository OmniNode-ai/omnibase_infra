# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Idempotency configuration model for event bus message consumption.

This module provides the configuration model for idempotency behavior in
event bus consumers. When enabled, consumers deduplicate messages based on
the `envelope_id` field from the event envelope using an INSERT ON CONFLICT
DO NOTHING pattern.

Idempotency Overview:
    Idempotency ensures that processing the same message multiple times
    produces the same result as processing it once. This is critical in
    distributed systems where message delivery can be at-least-once.

    The idempotency store tracks processed `envelope_id` values:
    - On first encounter: Record envelope_id, process message
    - On duplicate: Skip processing (already recorded)
    - After retention period: Prune old records to limit storage

Store Types:
    - postgres: Production-grade persistent storage using INSERT ON CONFLICT
    - memory: In-memory store for testing only (data lost on restart)

See Also:
    - EventBusSubcontractWiring: Uses this configuration for consumer setup
    - docs/patterns/idempotency_patterns.md: Implementation details
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelIdempotencyConfig(BaseModel):
    """Idempotency configuration for message consumption.

    When enabled, the consumer deduplicates messages based on the `envelope_id`
    field from the event envelope. The deduplication key is always `envelope_id`
    (not configurable) to ensure consistent behavior across all consumers.

    Attributes:
        enabled: Whether to enable idempotency checking. When False, all
            messages are processed regardless of prior processing.
            Default: False.
        store_type: Backend for storing processed envelope IDs.
            - "postgres": Production-grade, uses INSERT ON CONFLICT DO NOTHING
            - "memory": In-memory store for testing only
            Default: "postgres".
        retention_days: Number of days to retain processed envelope IDs before
            cleanup. Longer retention uses more storage but provides stronger
            deduplication guarantees for delayed retries.
            Must be between 1 and 90 days. Default: 7.

    Example:
        ```python
        from omnibase_infra.models.event_bus import ModelIdempotencyConfig

        # Production configuration
        config = ModelIdempotencyConfig(
            enabled=True,
            store_type="postgres",
            retention_days=14,
        )

        # Testing configuration
        test_config = ModelIdempotencyConfig(
            enabled=True,
            store_type="memory",
            retention_days=1,
        )

        # Disabled (default behavior)
        disabled = ModelIdempotencyConfig()  # enabled=False
        ```

    Configuration Guidelines:
        - Enable idempotency for all consumers processing side-effecting events
        - Use "postgres" store_type in production for durability
        - Set retention_days based on maximum expected retry window
        - For high-throughput topics, consider shorter retention to reduce storage

    Note:
        The deduplication key is always the `envelope_id` field from the event
        envelope. This is intentionally not configurable to ensure consistent
        behavior and prevent misconfiguration.

    See Also:
        EventBusSubcontractWiring: Consumer configuration that uses this model.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "enabled": True,
                    "store_type": "postgres",
                    "retention_days": 7,
                },
                {
                    "enabled": True,
                    "store_type": "memory",
                    "retention_days": 1,
                },
            ]
        },
    )

    enabled: bool = Field(
        default=False,
        description="Enable idempotency checking for message deduplication",
    )

    store_type: Literal["postgres", "memory"] = Field(
        default="postgres",
        description="Idempotency store backend. 'memory' is for testing only.",
    )

    retention_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Days to retain processed envelope IDs before cleanup",
    )


__all__ = ["ModelIdempotencyConfig"]
