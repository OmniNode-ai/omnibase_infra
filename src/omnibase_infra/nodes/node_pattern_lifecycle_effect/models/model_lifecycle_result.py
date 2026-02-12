# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for pattern lifecycle update operations."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumLifecycleTier, EnumValidationVerdict


class ModelLifecycleResult(BaseModel):
    """Result of a lifecycle tier update operation.

    Attributes:
        pattern_id: Identifier of the pattern that was evaluated.
        previous_tier: Tier before the verdict was applied.
        new_tier: Tier after the verdict was applied.
        tier_changed: Whether the tier changed as a result of the verdict.
        verdict_applied: The validation verdict that was applied.
        correlation_id: Correlation ID for distributed tracing.
        error: Error message (empty if success).
        error_code: Machine-readable error code (empty if success).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pattern_id: UUID = Field(
        ...,
        description="Identifier of the pattern that was evaluated.",
    )
    previous_tier: EnumLifecycleTier = Field(
        ...,
        description="Tier before the verdict was applied.",
    )
    new_tier: EnumLifecycleTier = Field(
        ...,
        description="Tier after the verdict was applied.",
    )
    tier_changed: bool = Field(
        ...,
        description="Whether the tier changed as a result of the verdict.",
    )
    verdict_applied: EnumValidationVerdict = Field(
        ...,
        description="The validation verdict that was applied.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    error: str = Field(
        default="",
        description="Error message (empty if success).",
    )
    error_code: str = Field(
        default="",
        description="Machine-readable error code (empty if success).",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            no error is present. Differs from typical Pydantic behavior.
        """
        return not self.error


__all__: list[str] = ["ModelLifecycleResult"]
