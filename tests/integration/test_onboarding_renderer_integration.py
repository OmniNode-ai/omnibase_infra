# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for the onboarding renderer pipeline (OMN-8810).

Validates the full renderer pipeline: load canonical graph -> resolve policy
-> construct step_results -> render markdown output. Tests run against real
canonical.yaml data without external service dependencies.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_step_result import (
    ModelStepResult,
)
from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import resolve_policy
from omnibase_infra.onboarding.renderers.renderer_markdown import (
    RendererOnboardingMarkdown,
)

pytestmark = pytest.mark.integration


class TestOnboardingRendererIntegration:
    """Integration: renderer pipeline with real canonical graph data."""

    def _load_steps(self) -> list:
        graph = load_canonical_graph()
        return resolve_policy(graph, target_capabilities=["first_node_running"])

    def _all_passing(self, steps: list) -> list[ModelStepResult]:
        return [
            ModelStepResult(step_key=s.step_key, passed=True, message="Passed")
            for s in steps
        ]

    def test_all_steps_present_in_rendered_output(self) -> None:
        steps = self._load_steps()
        results = self._all_passing(steps)
        output = RendererOnboardingMarkdown().render(steps, results)
        for step in steps:
            assert step.name in output, (
                f"Step '{step.name}' missing from rendered output"
            )

    def test_no_html_comment_in_rendered_output(self) -> None:
        steps = self._load_steps()
        results = self._all_passing(steps)
        output = RendererOnboardingMarkdown().render(steps, results)
        assert "GENERATED FROM canonical.yaml" not in output
        assert "<!--" not in output

    def test_pass_fail_indicators_correct(self) -> None:
        steps = self._load_steps()
        assert len(steps) >= 2, "Need at least 2 steps to test indicators"
        results = [
            ModelStepResult(step_key=steps[0].step_key, passed=True, message="OK"),
            ModelStepResult(
                step_key=steps[1].step_key, passed=False, message="intentional failure"
            ),
        ] + [
            ModelStepResult(step_key=s.step_key, passed=False, message="Skipped")
            for s in steps[2:]
        ]
        output = RendererOnboardingMarkdown().render(steps, results)
        assert "[x]" in output
        assert "[!]" in output
        assert "intentional failure" in output

    def test_summary_line_reflects_pass_count(self) -> None:
        steps = self._load_steps()
        results = self._all_passing(steps)
        output = RendererOnboardingMarkdown().render(steps, results)
        expected = f"{len(steps)}/{len(steps)} steps passed"
        assert expected in output
