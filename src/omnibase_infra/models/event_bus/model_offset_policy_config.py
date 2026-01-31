# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kafka offset commit policy configuration model.

This module defines the configuration for Kafka consumer offset commit strategies,
controlling when offsets are committed relative to handler execution.

Delivery Semantics:
    - At-least-once (default): Offsets committed AFTER successful handler execution.
      Messages may be redelivered on failure, requiring idempotent handlers.
    - At-most-once: Offsets committed BEFORE handler execution.
      Messages may be lost on failure, but never processed twice.
    - Manual: Explicit offset control for complex transaction scenarios.

Design Decision:
    The default is `commit_after_handler` (at-least-once) because:
    1. Message loss is typically worse than duplicate processing
    2. Idempotency can be enforced at the handler level (via idempotency keys)
    3. This aligns with Kafka best practices for reliable message processing

See Also:
    - ModelIdempotencyConfig: Pairs with at-least-once for duplicate detection
    - docs/patterns/kafka_delivery_semantics.md: Full delivery guarantee documentation
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelOffsetPolicyConfig(BaseModel):
    """Kafka offset commit policy configuration.

    Controls when consumer offsets are committed relative to handler execution.
    Default is 'commit_after_handler' for at-least-once delivery semantics.

    Attributes:
        commit_strategy: When to commit Kafka offsets relative to handler execution.
            - "commit_after_handler": At-least-once delivery (default, safe).
              Offsets committed only after successful handler completion.
              Messages may be redelivered on failure - handlers must be idempotent.
            - "commit_before_handler": At-most-once delivery (may lose messages).
              Offsets committed before handler execution begins.
              Suitable only when message loss is acceptable.
            - "manual": Explicit offset control via handler callback.
              For complex transactional scenarios requiring precise control.

    Example:
        ```python
        from omnibase_infra.models.event_bus import ModelOffsetPolicyConfig

        # Default: at-least-once (recommended)
        config = ModelOffsetPolicyConfig()
        assert config.commit_strategy == "commit_after_handler"

        # Explicit at-most-once (use with caution)
        config = ModelOffsetPolicyConfig(commit_strategy="commit_before_handler")

        # Manual control for transactions
        config = ModelOffsetPolicyConfig(commit_strategy="manual")
        ```

    Warning:
        Using "commit_before_handler" may result in message loss if the handler
        fails after offset commit. Only use when message loss is acceptable
        (e.g., metrics, non-critical logs).

    See Also:
        ModelIdempotencyConfig: For duplicate detection with at-least-once delivery.
    """

    commit_strategy: Literal[
        "commit_after_handler",
        "commit_before_handler",
        "manual",
    ] = Field(
        default="commit_after_handler",
        description=(
            "When to commit Kafka offsets relative to handler execution. "
            "'commit_after_handler' provides at-least-once delivery (default, safe). "
            "'commit_before_handler' provides at-most-once delivery (may lose messages). "
            "'manual' provides explicit control for transactional scenarios."
        ),
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commit_strategy": "commit_after_handler",
                },
                {
                    "commit_strategy": "commit_before_handler",
                },
                {
                    "commit_strategy": "manual",
                },
            ]
        },
    )


__all__ = ["ModelOffsetPolicyConfig"]
