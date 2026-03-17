# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the onboarding orchestrator handler (OMN-5270)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    handle_onboarding,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.probes.model_verification_result import ModelVerificationResult


def _make_passing_result(
    check_type: str = "command_exit_0", target: str = "test"
) -> ModelVerificationResult:
    return ModelVerificationResult(
        passed=True,
        check_type=check_type,
        target=target,
        message="Passed",
        elapsed_ms=10,
    )


def _make_failing_result(
    check_type: str = "command_exit_0", target: str = "test"
) -> ModelVerificationResult:
    return ModelVerificationResult(
        passed=False,
        check_type=check_type,
        target=target,
        message="Failed",
        elapsed_ms=10,
    )


class TestHandleOnboarding:
    """Tests for handle_onboarding()."""

    @pytest.mark.asyncio
    async def test_happy_path_all_pass(self) -> None:
        """All verifications pass for standalone quickstart."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            new_callable=AsyncMock,
            return_value=_make_passing_result(),
        ):
            output = await handle_onboarding(input_model)

        assert output.success is True
        assert output.total_steps == 5
        assert output.completed_steps == 5
        assert len(output.step_results) == 5
        assert all(r.passed for r in output.step_results)
        assert "Onboarding Progress" in output.rendered_output

    @pytest.mark.asyncio
    async def test_verification_failure_stops_execution(self) -> None:
        """A failing step stops subsequent steps."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
        )
        # First call passes, second fails
        mock = AsyncMock(
            side_effect=[
                _make_passing_result(),
                _make_failing_result(),
                _make_passing_result(),
                _make_passing_result(),
                _make_passing_result(),
            ]
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            mock,
        ):
            output = await handle_onboarding(input_model)

        assert output.success is False
        assert output.completed_steps == 1  # Only first passed
        # Steps after failure should be skipped
        skipped = [r for r in output.step_results if "Skipped" in r.message]
        assert len(skipped) == 3  # 3 steps skipped after the failure

    @pytest.mark.asyncio
    async def test_partial_progress_rendering(self) -> None:
        """Rendered output includes only completed steps."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
        )
        mock = AsyncMock(
            side_effect=[
                _make_passing_result(),
                _make_failing_result(),
                _make_passing_result(),
                _make_passing_result(),
                _make_passing_result(),
            ]
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            mock,
        ):
            output = await handle_onboarding(input_model)

        # Only first step (Check Python) should be in rendered output
        assert "Check Python" in output.rendered_output
        # Third step (Install omnibase_core) should NOT be rendered
        assert "Install omnibase_core" not in output.rendered_output
