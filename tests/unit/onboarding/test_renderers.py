# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for onboarding renderers (OMN-5269)."""

from __future__ import annotations

from omnibase_infra.onboarding.loader import load_canonical_graph
from omnibase_infra.onboarding.policy_resolver import resolve_policy
from omnibase_infra.onboarding.renderers.renderer_cli import RendererOnboardingCli
from omnibase_infra.onboarding.renderers.renderer_markdown import (
    RendererOnboardingMarkdown,
)


def _get_standalone_steps():
    graph = load_canonical_graph()
    return resolve_policy(graph, target_capabilities=["first_node_running"])


class TestRendererOnboardingMarkdown:
    """Tests for the markdown renderer."""

    def test_produces_valid_checklist(self) -> None:
        steps = _get_standalone_steps()
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps, title="Standalone Quickstart")
        assert "# Standalone Quickstart" in output
        assert "- [ ]" in output
        assert "GENERATED FROM canonical.yaml" in output

    def test_contains_step_names(self) -> None:
        steps = _get_standalone_steps()
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps)
        assert "Check Python Installation" in output
        assert "Install uv Package Manager" in output

    def test_contains_verification_commands(self) -> None:
        steps = _get_standalone_steps()
        renderer = RendererOnboardingMarkdown()
        output = renderer.render(steps)
        assert "python3 --version" in output
        assert "uv --version" in output


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
