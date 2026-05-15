# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for onboarding verification semantics (OMN-11050)."""

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
from omnibase_infra.onboarding.model_onboarding_step_verification import (
    ModelOnboardingStepVerification,
)
from omnibase_infra.probes.model_verification_result import ModelVerificationResult

pytestmark = pytest.mark.integration


def _passing_result(target: str) -> ModelVerificationResult:
    return ModelVerificationResult(
        passed=True,
        check_type="integration_stub",
        target=target,
        message="Passed",
        elapsed_ms=1,
    )


@pytest.mark.asyncio
async def test_verification_named_infra_steps_resolve_through_handler() -> None:
    """The real DAG policy resolves renamed check_* steps through the handler."""
    calls: list[str] = []

    async def fake_execute(
        verification: ModelOnboardingStepVerification,
    ) -> ModelVerificationResult:
        target = verification.target
        calls.append(target)
        return _passing_result(target)

    input_model = ModelOnboardingInput(target_capabilities=["event_bus_connected"])
    with patch(
        "omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding.execute_verification",
        AsyncMock(side_effect=fake_execute),
    ):
        output = await handle_onboarding(input_model)

    assert output.success is True
    assert output.status == EnumOnboardingStatus.PASSED
    assert output.total_steps == 6
    assert output.completed_steps == 6
    assert "docker_infra_running" in output.verified_capabilities
    assert "event_bus_running" in output.verified_capabilities
    assert "event_bus_connected" in output.verified_capabilities
    assert output.unmet_capabilities == []
    assert calls == [
        "python3 --version",
        "uv --version",
        "omnibase_core",
        "localhost:5436",
        "localhost:19092",
        "uv run python -c \"from omnibase_infra.event_bus import get_bus; print('ok')\"",
    ]
