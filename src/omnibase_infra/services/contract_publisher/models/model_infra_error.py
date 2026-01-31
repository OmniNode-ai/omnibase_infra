# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure Error Model.

Fatal error for infrastructure failures (Kafka, network, etc.).

.. versionadded:: 0.3.0
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelInfraError(BaseModel):
    """Infrastructure error during contract publishing.

    Represents errors that occur due to infrastructure failures (Kafka,
    network, etc.). These errors are potentially fatal depending on
    fail_fast configuration.

    Error Types:
        publisher_unavailable: Event bus publisher not available in container
        kafka_timeout: Kafka operation timed out
        broker_down: Kafka broker unreachable
        publish_failed: Generic publish failure
        serialization_failed: Failed to serialize event envelope

    Attributes:
        error_type: Category of the error
        message: Human-readable error description
        retriable: Whether the operation might succeed on retry

    Example:
        >>> error = ModelInfraError(
        ...     error_type="kafka_timeout",
        ...     message="Publish timed out after 30s",
        ...     retriable=True,
        ... )

    .. versionadded:: 0.3.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: Literal[
        "publisher_unavailable",
        "kafka_timeout",
        "broker_down",
        "publish_failed",
        "serialization_failed",
    ] = Field(description="Category of the error")
    message: str = Field(description="Human-readable error description")
    retriable: bool = Field(
        default=False,
        description="Whether the operation might succeed on retry",
    )


__all__ = ["ModelInfraError"]
