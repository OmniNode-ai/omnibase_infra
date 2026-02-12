# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pattern lifecycle promotion tiers for the lifecycle effect node."""

from __future__ import annotations

from enum import Enum


class EnumLifecycleTier(str, Enum):
    """Promotion tier for a pattern's lifecycle.

    Promotion order: OBSERVED -> SUGGESTED -> SHADOW_APPLY -> PROMOTED -> DEFAULT.
    Demotion: On repeated FAIL (2 -> demote one tier, 3 -> suppress).

    Values:
        OBSERVED: Pattern detected but not yet recommended.
        SUGGESTED: Pattern recommended for adoption.
        SHADOW_APPLY: Pattern applied in shadow mode (dry-run).
        PROMOTED: Pattern actively applied.
        DEFAULT: Pattern is the default standard.
        SUPPRESSED: Pattern suppressed after repeated failures.
    """

    OBSERVED = "observed"
    """Pattern detected but not yet recommended."""

    SUGGESTED = "suggested"
    """Pattern recommended for adoption."""

    SHADOW_APPLY = "shadow_apply"
    """Pattern applied in shadow mode (dry-run)."""

    PROMOTED = "promoted"
    """Pattern actively applied."""

    DEFAULT = "default"
    """Pattern is the default standard."""

    SUPPRESSED = "suppressed"
    """Pattern suppressed after repeated failures."""

    def can_promote(self) -> bool:
        """Return True if this tier can be promoted to a higher tier."""
        return self in (
            EnumLifecycleTier.OBSERVED,
            EnumLifecycleTier.SUGGESTED,
            EnumLifecycleTier.SHADOW_APPLY,
            EnumLifecycleTier.PROMOTED,
        )

    def promoted(self) -> EnumLifecycleTier:
        """Return the next promotion tier.

        Returns:
            The next higher tier, or DEFAULT if already PROMOTED.

        Raises:
            ValueError: If the tier cannot be promoted (DEFAULT or SUPPRESSED).
        """
        promotion_map: dict[EnumLifecycleTier, EnumLifecycleTier] = {
            EnumLifecycleTier.OBSERVED: EnumLifecycleTier.SUGGESTED,
            EnumLifecycleTier.SUGGESTED: EnumLifecycleTier.SHADOW_APPLY,
            EnumLifecycleTier.SHADOW_APPLY: EnumLifecycleTier.PROMOTED,
            EnumLifecycleTier.PROMOTED: EnumLifecycleTier.DEFAULT,
        }
        if self not in promotion_map:
            msg = f"Cannot promote from tier {self.value}"
            raise ValueError(msg)
        return promotion_map[self]

    def demoted(self) -> EnumLifecycleTier:
        """Return the next demotion tier.

        Returns:
            The next lower tier, OBSERVED if already at the bottom, or
            SUPPRESSED if already suppressed (demotion floor).
        """
        demotion_map: dict[EnumLifecycleTier, EnumLifecycleTier] = {
            EnumLifecycleTier.DEFAULT: EnumLifecycleTier.PROMOTED,
            EnumLifecycleTier.PROMOTED: EnumLifecycleTier.SHADOW_APPLY,
            EnumLifecycleTier.SHADOW_APPLY: EnumLifecycleTier.SUGGESTED,
            EnumLifecycleTier.SUGGESTED: EnumLifecycleTier.OBSERVED,
            EnumLifecycleTier.OBSERVED: EnumLifecycleTier.OBSERVED,
            EnumLifecycleTier.SUPPRESSED: EnumLifecycleTier.SUPPRESSED,
        }
        if self not in demotion_map:
            msg = f"Cannot demote from tier {self.value}"
            raise ValueError(msg)
        return demotion_map[self]

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumLifecycleTier"]
