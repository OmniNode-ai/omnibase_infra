# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Markdown renderer for onboarding plans (OMN-5269)."""

from __future__ import annotations

from omnibase_infra.onboarding.model_onboarding_step import ModelOnboardingStep


class RendererOnboardingMarkdown:
    """Renders a resolved onboarding plan as a markdown checklist."""

    def render(
        self,
        steps: list[ModelOnboardingStep],
        title: str = "Onboarding Checklist",
    ) -> str:
        """Render steps as a markdown checklist.

        Args:
            steps: Resolved steps in execution order.
            title: Document title.

        Returns:
            Markdown string with checklist.
        """
        lines: list[str] = [
            "<!-- GENERATED FROM canonical.yaml -- DO NOT EDIT MANUALLY -->",
            "",
            f"# {title}",
            "",
        ]

        for i, step in enumerate(steps, 1):
            lines.append(f"## {step.name}")
            lines.append("")
            if step.description:
                lines.append(step.description)
                lines.append("")

            lines.append(f"- [ ] **{step.name}**")

            if step.verification:
                lines.append(
                    f"  - Verify: `{step.verification.target}` "
                    f"({step.verification.check_type})"
                )

            if step.estimated_duration_seconds:
                minutes = step.estimated_duration_seconds // 60
                seconds = step.estimated_duration_seconds % 60
                if minutes > 0:
                    lines.append(f"  - Estimated time: {minutes}m {seconds}s")
                else:
                    lines.append(f"  - Estimated time: {seconds}s")

            lines.append("")

        return "\n".join(lines)


__all__ = ["RendererOnboardingMarkdown"]
