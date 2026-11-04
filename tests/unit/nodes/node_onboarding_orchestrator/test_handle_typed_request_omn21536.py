# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed onboarding request and constructor adapter regression tests."""

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from omnibase_core.enums import EnumInjectionScope
from omnibase_core.errors.error_service_resolution import ServiceResolutionError
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    HandlerOnboarding,
    OnboardingHandlerError,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.protocol_input_adapter import ProtocolInputAdapter
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine


async def test_handle_interactive_request() -> None:
    adapter = AdapterFakeInput(
        responses={
            "choose_deployment_mode": "local",
            "configure_local_services": ["kafka", "postgres"],
        }
    )
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        interface=ProtocolInputAdapter,
        instance=adapter,
        scope=EnumInjectionScope.GLOBAL,
    )
    handler = HandlerOnboarding(container)
    request = ModelOnboardingInput(policy_name="interactive_onboarding")

    output = await handler.handle(request)

    assert output.success is True
    assert output.policy_name == "interactive_onboarding"
    assert output.terminal_step == "write_config_local"
    assert output.provenance is not None
    assert output.provenance.env_dict["ONEX_DEPLOYMENT_MODE"] == "local"


async def test_interactive_request_requires_registered_adapter() -> None:
    request = ModelOnboardingInput(policy_name="interactive_onboarding")
    with pytest.raises(
        OnboardingHandlerError, match="input_adapter is required"
    ) as exc_info:
        await HandlerOnboarding(ModelONEXContainer()).handle(request)
    assert isinstance(exc_info.value.__cause__, ServiceResolutionError)


async def test_interactive_request_preserves_unexpected_container_error() -> None:
    container = MagicMock(spec=ModelONEXContainer)
    error = RuntimeError("adapter initialization failed")
    container.get_service_async = AsyncMock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        await HandlerOnboarding(container).handle(
            ModelOnboardingInput(policy_name="interactive_onboarding")
        )
    assert caught.value is error


async def test_packaged_contract_constructs_and_dispatches_onboarding() -> None:
    contract = (
        Path(__file__).resolve().parents[4]
        / "src/omnibase_infra/nodes/node_onboarding_orchestrator/contract.yaml"
    )
    manifest = discover_contracts_from_paths([contract])
    assert manifest.errors == ()
    assert manifest.total_discovered == 1
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        interface=ProtocolInputAdapter,
        instance=AdapterFakeInput(
            responses={
                "choose_deployment_mode": "local",
                "configure_local_services": ["kafka", "postgres"],
            }
        ),
        scope=EnumInjectionScope.GLOBAL,
    )
    engine = MessageDispatchEngine()
    event_bus = EventBusInmemory()
    await event_bus.start()
    try:
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=engine,
            event_bus=event_bus,
            environment="test",
            container=container,
            subscribe_immediately=False,
        )
        assert report.total_failed == 0, report
        assert report.total_wired == 1
        engine.freeze()
        request = ModelOnboardingInput(policy_name="interactive_onboarding")
        topic = manifest.contracts[0].event_bus.subscribe_topics[0]
        result = await engine.dispatch(
            topic,
            ModelEventEnvelope[object](
                payload=request.model_dump(mode="json"),
                event_type="ModelOnboardingInput",
            ),
        )
        assert result.status is EnumDispatchStatus.SUCCESS, result
    finally:
        await event_bus.close()


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelOnboardingInput.model_validate({"unknown_field": True})


def test_handle_has_one_request_parameter() -> None:
    assert list(
        inspect.signature(HandlerOnboarding(ModelONEXContainer()).handle).parameters
    ) == ["request"]
