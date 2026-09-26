# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Routing pins coexist with the dogfood fault guard (OMN-19124)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

import omnibase_infra.runtime.dogfood_delegation_fault_routes as fault_routes
from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def fault_declaration(tmp_path: Path) -> Path:
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "dogfood": {
                        "broker": "dogfood-broker:9092",
                        "delegation_fault_routes": [
                            {
                                "backend_id": "dogfood-fault-429",
                                "endpoint_url": "http://dogfood-delegation-fault-429:8080/v1/chat/completions",
                                "expected_http_status": 429,
                                "requested_timeout_seconds": 240,
                                "max_attempts": 1,
                                "no_escalation": True,
                            }
                        ],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _request(backend_id: str, no_escalation: bool) -> SimpleNamespace:
    return SimpleNamespace(
        backend_id=backend_id,
        no_escalation=no_escalation,
        requested_timeout_seconds=240,
    )


def test_routing_pin_consumer_admits_dev_lane(
    fault_declaration: Path,
) -> None:
    fault_routes.validate_dogfood_delegation_fault_request(
        request=_request("cloud-glm", False),
        event_bus=SimpleNamespace(environment="dev", bootstrap_servers="redpanda:9092"),
        path_for_test=fault_declaration,
    )


def test_routing_pin_consumer_admits_without_bus_identity(
    fault_declaration: Path,
) -> None:
    fault_routes.validate_dogfood_delegation_fault_request(
        request=_request("cloud-glm", False),
        event_bus=None,
        path_for_test=fault_declaration,
    )


def test_routing_pin_consumer_refuses_no_escalation_on_dev(
    fault_declaration: Path,
) -> None:
    with pytest.raises(InfraUnavailableError, match="dogfood lane"):
        fault_routes.validate_dogfood_delegation_fault_request(
            request=_request("cloud-glm", True),
            event_bus=SimpleNamespace(
                environment="dev", bootstrap_servers="redpanda:9092"
            ),
            path_for_test=fault_declaration,
        )


def test_routing_pin_consumer_refuses_declared_fault_on_dev(
    fault_declaration: Path,
) -> None:
    with pytest.raises(InfraUnavailableError, match="dogfood lane"):
        fault_routes.validate_dogfood_delegation_fault_request(
            request=_request("dogfood-fault-429", False),
            event_bus=SimpleNamespace(
                environment="dev", bootstrap_servers="redpanda:9092"
            ),
            path_for_test=fault_declaration,
        )


def test_routing_pin_consumer_refuses_unloadable_fault_declaration(
    tmp_path: Path,
) -> None:
    with pytest.raises(InfraUnavailableError, match="cannot load"):
        fault_routes.validate_dogfood_delegation_fault_request(
            request=_request("cloud-glm", False),
            event_bus=None,
            path_for_test=tmp_path / "missing.yaml",
        )


@pytest.mark.asyncio
async def test_routing_pin_producer_publishes_on_dev_without_fault_resolution(
    monkeypatch: pytest.MonkeyPatch,
    fault_declaration: Path,
) -> None:
    route = ModelRuntimeLocalIngressRoute(
        node_name="node_delegation_orchestrator",
        contract_name="node_delegation_orchestrator",
        command_topic="onex.cmd.omnimarket.delegation-request.v1",
        event_type="omnimarket.delegation-request",
        terminal_event="onex.evt.omnimarket.delegation-completed.v1",
        terminal_events=(
            "onex.evt.omnimarket.delegation-completed.v1",
            "onex.evt.omnimarket.delegation-failed.v1",
        ),
        contract_path="/contracts/omnimarket/node_delegation_orchestrator.yaml",
        package_name="omnimarket",
    )
    published: list[dict[str, object]] = []

    class FakeBus:
        environment = "dev"
        bootstrap_servers = "redpanda:9092"

    class FakeBroker:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def dispatch_request(
            self, command: ModelDispatchBusCommand
        ) -> tuple[object, ModelDispatchBusTerminalResult]:
            published.append(dict(command.payload))
            return route, ModelDispatchBusTerminalResult(
                correlation_id=command.correlation_id,
                status="completed",
                payload={"content": "ok"},
                completed_at=datetime.now(UTC),
            )

    def refuse_fault_resolution(**_kwargs: object) -> None:
        raise AssertionError("ordinary routing pin must not resolve a fault route")

    declared_routes = fault_routes.load_dogfood_delegation_fault_routes(
        path_for_test=fault_declaration
    )
    monkeypatch.setattr(
        fault_routes,
        "load_dogfood_delegation_fault_routes",
        lambda **_kwargs: declared_routes,
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        FakeBroker,
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.resolve_bounded_delegation_route",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        fault_routes,
        "resolve_dogfood_delegation_fault_route",
        refuse_fault_resolution,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=FakeBus(),  # type: ignore[arg-type]
        routes={"delegation.orchestrate": route},
    )

    await port.dispatch(
        prompt="probe",
        task_type="document",
        correlation_id=uuid4(),
        max_tokens=512,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        backend_id="cloud-glm",
    )

    assert published[0]["backend_id"] == "cloud-glm"
    assert "no_escalation" not in published[0]
