# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Markdown renderer for onboarding plans (OMN-5269)."""

from __future__ import annotations

from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_step_result import (
    ModelStepResult,
)
from omnibase_infra.onboarding.model_onboarding_step import ModelOnboardingStep
from omnibase_infra.utils import sanitize_error_string


class RendererOnboardingMarkdown:
    """Renders a resolved onboarding plan as a markdown checklist."""

    def render(
        self,
        steps: list[ModelOnboardingStep],
        step_results: list[ModelStepResult],
        title: str = "Onboarding Checklist",
    ) -> str:
        """Render all steps with pass/fail indicators as a markdown checklist.

        Args:
            steps: All resolved steps in execution order.
            step_results: Per-step execution results (same length and order as steps).
            title: Document title.

        Returns:
            Markdown string with checklist.
        """
        result_by_key = {r.step_key: r for r in step_results}

        lines: list[str] = [
            f"# {title}",
            "",
        ]

        passed_count = sum(
            1
            for s in steps
            if result_by_key.get(s.step_key) and result_by_key[s.step_key].passed
        )
        lines.append(f"{passed_count}/{len(steps)} steps passed")
        lines.append("")

        for step in steps:
            result = result_by_key.get(step.step_key)
            if result is None:
                indicator = "[?]"
                suffix = " — Missing execution result"
            elif result.passed:
                indicator = "[x]"
                suffix = ""
            else:
                indicator = "[!]"
                suffix = (
                    f" — {sanitize_error_string(result.message)}"
                    if result.message
                    else ""
                )

            lines.append(f"## {step.name}")
            lines.append("")
            if step.description:
                lines.append(step.description)
                lines.append("")

            lines.append(f"- {indicator} **{step.name}**{suffix}")

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
