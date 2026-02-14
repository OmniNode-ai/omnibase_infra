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
    correlation_id: UUID | None = Field(
        default=None,
        description=(
            "Correlation ID for distributed tracing. "
            "Auto-generated via uuid4() if not provided."
        ),
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

        Baseline runs are only required for **upward** promotions where
        ``current_tier`` is SUGGESTED, SHADOW_APPLY, or PROMOTED **and**
        ``target_tier`` is strictly above ``current_tier`` in the
        promotion ladder.

        Returns ``False`` for:

        - Tier 0->1 (OBSERVED->SUGGESTED) promotions
        - No-op transitions (target == current)
        - Demotions (target tier is lower than current)
        - DEFAULT as current tier (already at top, cannot promote further)
        - SUPPRESSED tier (excluded from promotion ladder)

        Returns:
            True only when ``current_tier`` is in {SUGGESTED, SHADOW_APPLY,
            PROMOTED} **and** ``target_tier`` is strictly above
            ``current_tier`` in the promotion rank ordering.
        """
        tiers_requiring_baseline = {
            EnumLifecycleTier.SUGGESTED,
            EnumLifecycleTier.SHADOW_APPLY,
            EnumLifecycleTier.PROMOTED,
        }
        if self.current_tier not in tiers_requiring_baseline:
            return False

        # Rank ordering: higher rank = further along the lifecycle.
        # SUPPRESSED is excluded (cannot promote).
        tier_rank: dict[EnumLifecycleTier, int] = {
            EnumLifecycleTier.OBSERVED: 0,
            EnumLifecycleTier.SUGGESTED: 1,
            EnumLifecycleTier.SHADOW_APPLY: 2,
            EnumLifecycleTier.PROMOTED: 3,
            EnumLifecycleTier.DEFAULT: 4,
        }

        current_rank = tier_rank.get(self.current_tier)
        target_rank = tier_rank.get(self.target_tier)

        # Unknown tiers (e.g. SUPPRESSED target) never require baseline.
        if current_rank is None or target_rank is None:
            return False

        # Only upward promotions require a baseline comparison.
        return target_rank > current_rank


__all__: list[str] = ["ModelBaselineRunConfig"]
