# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Model for the aggregate demo reset report.

.. versionadded:: 0.9.1
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.cli.enum_reset_action import EnumResetAction
from omnibase_infra.cli.model_reset_action_result import ModelResetActionResult

__all__: list[str] = [
    "ModelDemoResetReport",
]


@dataclass
class ModelDemoResetReport:
    """Aggregate report of all demo reset actions.

    Attributes:
        actions: List of individual action results.
        dry_run: Whether this was a dry-run (no changes made).
    """

    actions: list[ModelResetActionResult] = field(default_factory=list)
    dry_run: bool = False

    @property
    def reset_count(self) -> int:
        """Number of resources that were reset."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.RESET)

    @property
    def preserved_count(self) -> int:
        """Number of resources explicitly preserved."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.PRESERVED)

    @property
    def error_count(self) -> int:
        """Number of actions that failed."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.ERROR)

    @property
    def skipped_count(self) -> int:
        """Number of actions skipped (e.g., already clean)."""
        return sum(1 for a in self.actions if a.action == EnumResetAction.SKIPPED)

    def format_summary(self) -> str:
        """Format the report as a human-readable summary.

        Returns:
            Multi-line string suitable for CLI output.
        """
        lines: list[str] = []
        mode = "DRY RUN" if self.dry_run else "EXECUTED"
        lines.append(f"Demo Reset Report ({mode})")
        lines.append("=" * 60)

        # Group by action type
        for action_type in EnumResetAction:
            group = [a for a in self.actions if a.action == action_type]
            if not group:
                continue

            label = action_type.value.upper()
            lines.append(f"\n  [{label}]")
            for item in group:
                lines.append(f"    {item.resource}: {item.detail}")

        lines.append("")
        lines.append(
            f"Summary: {self.reset_count} reset, "
            f"{self.preserved_count} preserved, "
            f"{self.skipped_count} skipped, "
            f"{self.error_count} errors"
        )

        return "\n".join(lines)
