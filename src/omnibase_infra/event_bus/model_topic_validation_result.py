# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Topic Validation Result Model.

Structured result for startup topic existence validation, following the
``ModelIdempotencyStoreHealthCheckResult`` pattern. Used by
``TopicStartupValidator`` to report which platform topics are present
or missing on the Kafka/Redpanda broker.

Related Tickets:
    - OMN-3769: Registry-First Startup Assertions
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.enum_topic_validation_status import (
    EnumTopicValidationStatus,
)


class ModelTopicValidationResult(BaseModel):
    """Result of startup topic existence validation.

    Attributes:
        required_topics: All topic suffixes that the platform expects.
        present_topics: Topics confirmed present on the broker.
        missing_topics: Topics not found on the broker.
        is_valid: True when no topics are missing (or validation was skipped/unavailable).
        status: Overall validation outcome.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    required_topics: tuple[str, ...] = Field(default_factory=tuple)
    present_topics: tuple[str, ...] = Field(default_factory=tuple)
    missing_topics: tuple[str, ...] = Field(default_factory=tuple)
    is_valid: bool = Field(default=True)
    status: EnumTopicValidationStatus = Field(
        default=EnumTopicValidationStatus.SUCCESS,
    )


__all__: list[str] = [
    "ModelTopicValidationResult",
]
