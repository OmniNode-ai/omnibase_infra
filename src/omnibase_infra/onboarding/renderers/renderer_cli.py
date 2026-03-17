# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI renderer for onboarding plans (OMN-5269)."""

from __future__ import annotations

from omnibase_infra.onboarding.model_onboarding_step import ModelOnboardingStep

# ANSI color codes
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_DIM = "\033[2m"


class RendererOnboardingCli:
    """Renders a resolved onboarding plan as colorized terminal output."""

    def render(
        self,
        steps: list[ModelOnboardingStep],
        title: str = "Onboarding",
    ) -> str:
        """Render steps as colorized CLI output.

        Args:
            steps: Resolved steps in execution order.
            title: Section title.

        Returns:
            Colorized string for terminal display.
        """
        lines: list[str] = [
            f"{_BOLD}{_CYAN}=== {title} ==={_RESET}",
            "",
        ]

        total_seconds = sum(s.estimated_duration_seconds or 0 for s in steps)
        total_minutes = total_seconds // 60
        lines.append(
            f"{_DIM}{len(steps)} steps, ~{total_minutes} min estimated{_RESET}"
        )
        lines.append("")

        for i, step in enumerate(steps, 1):
            prefix = f"{_YELLOW}[{i}/{len(steps)}]{_RESET}"
            lines.append(f"{prefix} {_BOLD}{step.name}{_RESET}")

            if step.description:
                lines.append(f"      {_DIM}{step.description}{_RESET}")

            if step.verification:
                lines.append(
                    f"      {_GREEN}verify:{_RESET} {step.verification.target}"
                )

            lines.append("")

        return "\n".join(lines)


__all__ = ["RendererOnboardingCli"]
