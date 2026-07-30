# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumer-handler compatibility for the runtime-owned delegation port."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.integration


def _delegation_route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        event_type="omnibase-infra.delegation-request",
        terminal_event="onex.evt.omnibase-infra.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator/contract.yaml",
        package_name="omnimarket",
    )


def test_runtime_port_exposes_consumer_handler_optional_parameters() -> None:
    """The injected implementation and its protocol evolve as one boundary."""
    for dispatch_method in (
        ProtocolDelegationDispatchPort.dispatch,
        RuntimeDelegationDispatchPort.dispatch,
    ):
        parameters = inspect.signature(dispatch_method).parameters
        assert parameters["max_tokens"].annotation in {"int | None", int | None}
        assert parameters["backend_id"].default is None
        assert parameters["response_contract"].default is None


@pytest.mark.asyncio
async def test_absent_consumer_features_dispatch_through_runtime_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Always-supplied None kwargs must reach the existing Pattern-B route."""
    route = _delegation_route()
    captured_commands: list[ModelDispatchBusCommand] = []

    class FakePatternBBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
            captured_commands.append(command)
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={"content": "workflow-ok"},
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakePatternBBroker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=object(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
    )

    result = await port.dispatch(
        prompt="workflow probe",
        task_type="reasoning",
        correlation_id=uuid4(),
        max_tokens=None,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        quality_contract_mode="extend_task_class",
        acceptance_criteria=(),
        tenant_id=None,
        backend_id=None,
        response_contract=None,
    )

    assert result["status"] == "completed"
    assert captured_commands[0].payload["prompt"] == "workflow probe"
    assert "max_tokens" not in captured_commands[0].payload
    assert "backend_id" not in captured_commands[0].payload
    assert "response_contract" not in captured_commands[0].payload


@pytest.mark.asyncio
async def test_metered_terminal_cost_crosses_the_runtime_consumer_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metered workflow terminal reaches the consumer as measured actual cost."""
    route = _delegation_route()

    class FakePatternBBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={
                    "model_used": "gemini-2.5-flash",
                    "prompt_tokens": 115,
                    "completion_tokens": 130,
                    "final_attempt_cost": 0.00137,
                    "cumulative_attempt_cost": 0.00182,
                },
                completed_at=datetime.now(UTC),
            )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakePatternBBroker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=object(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
    )

    result = await port.dispatch(
        prompt="metered workflow probe",
        task_type="reasoning",
        correlation_id=uuid4(),
        max_tokens=None,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        quality_contract_mode="extend_task_class",
        acceptance_criteria=(),
        tenant_id=None,
        backend_id=None,
        response_contract=None,
    )

    assert result["cost_usd"] == pytest.approx(0.00182)
