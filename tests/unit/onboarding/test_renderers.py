# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for onboarding renderers (OMN-5269)."""

from __future__ import annotations

from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_step_result import (
    ModelStepResult,
)
from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import resolve_policy
from omnibase_infra.onboarding.renderers.renderer_cli import RendererOnboardingCli
from omnibase_infra.onboarding.renderers.renderer_markdown import (
    RendererOnboardingMarkdown,
)


def _get_standalone_steps():
    graph = load_canonical_graph()
    return resolve_policy(graph, target_capabilities=["first_node_running"])


def _all_passing_results(steps):
    return [
        ModelStepResult(step_key=s.step_key, passed=True, message="Passed")
        for s in steps
    ]


class TestRendererOnboardingMarkdown:
    """Tests for the markdown renderer."""

    def test_produces_valid_checklist(self) -> None:
        steps = _get_standalone_steps()
        results = _all_passing_results(steps)
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, results, title="Standalone Quickstart")
        assert "# Standalone Quickstart" in output
        assert "[x]" in output

    def test_no_html_comment_in_output(self) -> None:
        steps = _get_standalone_steps()
        results = _all_passing_results(steps)
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, results)
        assert "GENERATED FROM canonical.yaml" not in output
        assert "<!--" not in output

    def test_contains_step_names(self) -> None:
        steps = _get_standalone_steps()
        results = _all_passing_results(steps)
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, results)
        assert "Check Python Installation" in output
        assert "Install uv Package Manager" in output

    def test_contains_verification_commands(self) -> None:
        steps = _get_standalone_steps()
        results = _all_passing_results(steps)
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, results)
        assert "python3 --version" in output
        assert "uv --version" in output

    def test_failed_step_shows_exclamation_indicator(self) -> None:
        steps = _get_standalone_steps()
        results = [
            ModelStepResult(step_key=steps[0].step_key, passed=True, message="OK"),
            ModelStepResult(
                step_key=steps[1].step_key, passed=False, message="command not found"
            ),
        ] + [
            ModelStepResult(
                step_key=s.step_key,
                passed=False,
                message="Skipped due to previous failure",
            )
            for s in steps[2:]
        ]
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, results)
        assert "[!]" in output
        assert "command not found" in output

    def test_summary_line_shows_pass_count(self) -> None:
        steps = _get_standalone_steps()
        results = _all_passing_results(steps)
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, results)
        assert "steps passed" in output

    def test_all_steps_rendered_even_when_some_fail(self) -> None:
        steps = _get_standalone_steps()
        results = [
            ModelStepResult(step_key=steps[0].step_key, passed=True, message="OK"),
            ModelStepResult(
                step_key=steps[1].step_key, passed=False, message="failed"
            ),
        ] + [
            ModelStepResult(
                step_key=s.step_key,
                passed=False,
                message="Skipped due to previous failure",
            )
            for s in steps[2:]
        ]
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, results)
        # All 5 step names must appear, not just the passed ones
        for step in steps:
            assert step.name in output


class TestRendererOnboardingCli:
    """Tests for the CLI renderer."""

    def test_produces_colorized_output(self) -> None:
        steps = _get_standalone_steps()
        renderer = RendererOnboardingCli()
        output = renderer.render(steps, title="Standalone")
        assert "Standalone" in output
        assert "[1/5]" in output

    def test_contains_step_names(self) -> None:
        steps = _get_standalone_steps()
        renderer = RendererOnboardingCli()
        output = renderer.render(steps)
        assert "Check Python Installation" in output

    def test_contains_verify_targets(self) -> None:
        steps = _get_standalone_steps()
        renderer = RendererOnboardingCli()
        output = renderer.render(steps)
        assert "python3 --version" in output
