# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Configuration for an A/B baseline comparison run.

Defines the scenario, environment, and pattern toggle for a
baseline vs candidate comparison.  Baseline runs are optional
and only triggered for Tier 2+ (SHADOW_APPLY and above) promotion
decisions.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumLifecycleTier


class ModelBaselineRunConfig(BaseModel):
    """Configuration for an A/B baseline comparison run.

    Same scenario is run twice: once without the pattern (baseline)
    and once with the pattern (candidate).  The ``pattern_enabled``
    field is toggled by the run infrastructure -- the config itself
    describes what to run.

    Attributes:
        pattern_id: Identifier of the pattern being evaluated.
        scenario_id: Identifier of the test scenario to execute.
        correlation_id: Correlation ID for distributed tracing.
        current_tier: Current lifecycle tier of the pattern.
        target_tier: Target tier for the promotion decision.
        environment_snapshot: Opaque environment description to ensure
            baseline and candidate run under identical conditions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pattern_id: UUID = Field(
        ...,
        description="Identifier of the pattern being evaluated.",
    )
    scenario_id: UUID = Field(
        ...,
        description="Identifier of the test scenario to execute.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    current_tier: EnumLifecycleTier = Field(
        ...,
        description="Current lifecycle tier of the pattern.",
    )
    target_tier: EnumLifecycleTier = Field(
        ...,
        description="Target tier for the promotion decision.",
    )
    environment_snapshot: str = Field(
        default="",
        description=(
            "Opaque environment description to ensure baseline and "
            "candidate run under identical conditions."
        ),
    )

    def requires_baseline(self) -> bool:
        """Return True if this promotion decision requires a baseline run.

        Baseline runs are only required for Tier 2+ promotions
        (SHADOW_APPLY and above).  Tier 0->1 (OBSERVED->SUGGESTED)
        promotions do not require baseline comparison.

        Returns:
            True if current_tier is SUGGESTED or higher (i.e. the
            promotion is from SUGGESTED->SHADOW_APPLY or above).
        """
        tiers_requiring_baseline = {
            EnumLifecycleTier.SUGGESTED,
            EnumLifecycleTier.SHADOW_APPLY,
            EnumLifecycleTier.PROMOTED,
        }
        return self.current_tier in tiers_requiring_baseline


__all__: list[str] = ["ModelBaselineRunConfig"]
