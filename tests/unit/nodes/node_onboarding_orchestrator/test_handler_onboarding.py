# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the onboarding orchestrator handler (OMN-5270)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    handle_onboarding,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.enum_onboarding_status import (
    EnumOnboardingStatus,
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
        """Rendered output includes all steps with correct pass/fail indicators."""
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

        # All 5 steps appear in rendered output (not just completed ones)
        assert "Check Python" in output.rendered_output
        # Passed step shows [x] indicator
        assert "[x]" in output.rendered_output
        # Failed step shows [!] indicator
        assert "[!]" in output.rendered_output
        # No leaked HTML comment
        assert "<!-- GENERATED FROM" not in output.rendered_output
        # Summary line present
        assert "steps passed" in output.rendered_output


class TestEnumOnboardingStatusFields:
    """Tests for EnumOnboardingStatus and capability fields (OMN-11056)."""

    @pytest.mark.asyncio
    async def test_all_pass_yields_passed_status(self) -> None:
        """All steps passing sets status=PASSED."""
        input_model = ModelOnboardingInput(target_capabilities=["first_node_running"])
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            new_callable=AsyncMock,
            return_value=_make_passing_result(),
        ):
            output = await handle_onboarding(input_model)

        assert output.status == EnumOnboardingStatus.PASSED

    @pytest.mark.asyncio
    async def test_failure_without_skip_yields_failed_status(self) -> None:
        """A failing step (no skips) sets status=FAILED."""
        input_model = ModelOnboardingInput(
            target_capabilities=["first_node_running"],
            continue_on_failure=True,
        )
        mock = AsyncMock(
            side_effect=[
                _make_failing_result(),
                _make_failing_result(),
                _make_failing_result(),
                _make_failing_result(),
                _make_failing_result(),
            ]
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            mock,
        ):
            output = await handle_onboarding(input_model)

        assert output.status == EnumOnboardingStatus.FAILED

    @pytest.mark.asyncio
    async def test_blocked_when_steps_skipped(self) -> None:
        """Steps skipped due to previous failure sets status=BLOCKED."""
        input_model = ModelOnboardingInput(target_capabilities=["first_node_running"])
        mock = AsyncMock(
            side_effect=[
                _make_passing_result(),
                _make_failing_result(),
            ]
        )
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            mock,
        ):
            output = await handle_onboarding(input_model)

        assert output.status == EnumOnboardingStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_verified_capabilities_populated_on_pass(self) -> None:
        """Capabilities from passing steps appear in verified_capabilities."""
        input_model = ModelOnboardingInput(target_capabilities=["first_node_running"])
        with patch(
            "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
            new_callable=AsyncMock,
            return_value=_make_passing_result(),
        ):
            output = await handle_onboarding(input_model)

        # first_node_running graph steps produce capabilities — at least one should land
        assert isinstance(output.verified_capabilities, list)
        assert isinstance(output.unmet_capabilities, list)
        # All passed so nothing should be in unmet
        assert output.unmet_capabilities == []
