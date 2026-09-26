# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19124 routing-pin admission through the real dispatch boundary.

The packaged declaration belongs to OmniMarket, which is intentionally absent
from this test environment.  These integration seams route its real loaders to
a temporary declaration while retaining the production classification and
consumer validation logic.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml

import omnibase_infra.runtime.dogfood_delegation_fault_routes as fault_routes
from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime import service_delegation_dispatch_port as port_module
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute

pytestmark = pytest.mark.integration

_INTERNAL = "redpanda:9092"


class _BrokerReachedError(Exception):
    """Raised by the stand-in broker with the command that reached it."""

    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__("dispatch reached the stand-in broker")
        self.payload = payload


class _StandInBroker:
    """Final dispatch seam: no Kafka client is started by these tests."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def dispatch_request(
        self, command: ModelDispatchBusCommand
    ) -> tuple[object, object]:
        payload = command.payload
        assert isinstance(payload, dict)
        raise _BrokerReachedError(dict(payload))


def _route() -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
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


def _fault_declaration(tmp_path: Path) -> Path:
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
                                "endpoint_url": (
                                    "http://dogfood-delegation-fault-429:8080/"
                                    "v1/chat/completions"
                                ),
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


def _route_loaders_to(monkeypatch: pytest.MonkeyPatch, declaration: Path) -> None:
    original_load = fault_routes.load_dogfood_delegation_fault_routes
    original_load_document = fault_routes._load_lane_document

    def load_for_test(*, path_for_test: Path | None = None) -> object:
        del path_for_test
        return original_load(path_for_test=declaration)

    def load_document_for_test(path_for_test: Path | None = None) -> object:
        del path_for_test
        return original_load_document(path_for_test=declaration)

    monkeypatch.setattr(
        fault_routes, "load_dogfood_delegation_fault_routes", load_for_test
    )
    monkeypatch.setattr(fault_routes, "_load_lane_document", load_document_for_test)


def _dev_bus() -> EventBusKafka:
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=_INTERNAL,
            environment="dev",
        )
    )


def _port(event_bus: EventBusKafka) -> port_module.RuntimeDelegationDispatchPort:
    return port_module.RuntimeDelegationDispatchPort(
        event_bus=event_bus,
        routes={
            "omnimarket.node_delegation_orchestrator.delegation.orchestrate": (_route())
        },
    )


async def _dispatch(
    port: port_module.RuntimeDelegationDispatchPort,
    *,
    backend_id: str,
    no_escalation: bool,
) -> None:
    await port.dispatch(
        prompt="OMN-19124 integration probe",
        task_type="document",
        correlation_id=uuid4(),
        max_tokens=16,
        source_file_path=None,
        source_session_id=None,
        wait=True,
        backend_id=backend_id,
        no_escalation=no_escalation,
        execution_timeout_seconds=240,
    )


@pytest.mark.asyncio
async def test_routing_pin_on_dev_reaches_broker_with_no_fault_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _route_loaders_to(monkeypatch, _fault_declaration(tmp_path))
    monkeypatch.setattr(port_module, "RuntimePatternBBroker", _StandInBroker)
    monkeypatch.setattr(
        port_module, "resolve_bounded_delegation_route", lambda **_kwargs: None
    )

    with pytest.raises(_BrokerReachedError) as reached:
        await _dispatch(_port(_dev_bus()), backend_id="cloud-glm", no_escalation=False)

    assert reached.value.payload["backend_id"] == "cloud-glm"
    assert "no_escalation" not in reached.value.payload


@pytest.mark.asyncio
async def test_declared_fault_pin_on_dev_refuses_before_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _route_loaders_to(monkeypatch, _fault_declaration(tmp_path))
    monkeypatch.setattr(port_module, "RuntimePatternBBroker", _StandInBroker)
    monkeypatch.setattr(
        port_module, "resolve_bounded_delegation_route", lambda **_kwargs: None
    )

    with pytest.raises(InfraUnavailableError, match="dogfood lane"):
        await _dispatch(
            _port(_dev_bus()), backend_id="dogfood-fault-429", no_escalation=True
        )


def test_consumer_guard_admits_routing_pin_and_refuses_fault_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _route_loaders_to(monkeypatch, _fault_declaration(tmp_path))
    event_bus = _dev_bus()

    fault_routes.validate_dogfood_delegation_fault_request(
        request=SimpleNamespace(
            backend_id="cloud-glm",
            no_escalation=False,
            requested_timeout_seconds=240,
        ),
        event_bus=event_bus,
    )

    for backend_id, no_escalation in (
        ("cloud-glm", True),
        ("dogfood-fault-429", False),
    ):
        with pytest.raises(InfraUnavailableError, match="dogfood lane"):
            fault_routes.validate_dogfood_delegation_fault_request(
                request=SimpleNamespace(
                    backend_id=backend_id,
                    no_escalation=no_escalation,
                    requested_timeout_seconds=240,
                ),
                event_bus=event_bus,
            )


@pytest.mark.asyncio
async def test_missing_declaration_fails_closed_before_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _route_loaders_to(monkeypatch, tmp_path / "missing-ci_bus_lanes.yaml")
    monkeypatch.setattr(port_module, "RuntimePatternBBroker", _StandInBroker)
    monkeypatch.setattr(
        port_module, "resolve_bounded_delegation_route", lambda **_kwargs: None
    )

    with pytest.raises(InfraUnavailableError, match="cannot load"):
        await _dispatch(_port(_dev_bus()), backend_id="cloud-glm", no_escalation=False)
