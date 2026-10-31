# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed onboarding request and constructor adapter regression tests."""

import inspect

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    HandlerOnboarding,
    OnboardingHandlerError,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput


async def test_handle_interactive_request() -> None:
    adapter = AdapterFakeInput(
        responses={
            "choose_deployment_mode": "local",
            "configure_local_services": ["kafka", "postgres"],
        }
    )
    handler = HandlerOnboarding(input_adapter=adapter)
    request = ModelOnboardingInput(policy_name="interactive_onboarding")

    output = await handler.handle(request)

    assert output.success is True
    assert output.policy_name == "interactive_onboarding"
    assert output.terminal_step == "write_config_local"
    assert output.provenance is not None
    assert output.provenance.env_dict["ONEX_DEPLOYMENT_MODE"] == "local"
    assert handler.input_adapter is adapter


async def test_interactive_request_requires_constructor_adapter() -> None:
    request = ModelOnboardingInput(policy_name="interactive_onboarding")
    with pytest.raises(OnboardingHandlerError, match="input_adapter is required"):
        await HandlerOnboarding().handle(request)


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelOnboardingInput.model_validate({"unknown_field": True})


def test_handle_has_one_request_parameter() -> None:
    assert list(inspect.signature(HandlerOnboarding().handle).parameters) == ["request"]
