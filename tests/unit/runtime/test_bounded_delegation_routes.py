# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime.bounded_delegation_routes import (
    resolve_bounded_delegation_route,
)
from omnibase_infra.runtime.protocol_addressed_broker_transport import (
    ProtocolAddressedBrokerTransport,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.unit


class _AddressedBus:
    environment = "dogfood"
    bootstrap_servers = "192.168.86.105:47092"

    async def publish(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("publish must not be reached")

    async def subscribe(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("subscribe must not be reached")


class _UnaddressedBus:
    async def publish(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("an unidentified runtime bus must not publish")

    async def subscribe(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("an unidentified runtime bus must not subscribe")


def _route(**changes: object) -> ModelRuntimeLocalIngressRoute:
    fields: dict[str, object] = {
        "node_name": "node_delegation_orchestrator",
        "contract_name": "node_delegation_orchestrator",
        "command_topic": "onex.cmd.omnibase-infra.delegation-request.v1",
        "event_type": "omnimarket.delegation-request",
        "terminal_event": "onex.evt.omnibase-infra.delegation-completed.v1",
        "terminal_events": (
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
        "contract_path": "/contracts/node_delegation_orchestrator/contract.yaml",
        "package_name": "omnimarket",
    }
    fields.update(changes)
    return ModelRuntimeLocalIngressRoute(**fields)  # type: ignore[arg-type]


def _overlay(tmp_path: Path, mutate: tuple[str, str] | None = None) -> Path:
    """Write the minimal declared lane fixture needed by this unit seam.

    Resource-byte binding is covered independently; this suite must not depend
    on a particular adjacent worktree layout to exercise Infra's fail-closed
    route validation.
    """
    route: dict[str, object] = {
        "consumer": "omnimarket.nodes.node_delegation_orchestrator",
        "terminal_route": "terminal_events",
        "repository_owner": "omnimarket",
    }
    if mutate is not None:
        route[mutate[0]] = mutate[1]
    raw: dict[str, object] = {
        "lanes": {
            "dogfood": {
                "broker": "192.168.86.105:47092",
                "security_protocol": "PLAINTEXT",
                "delegation_routes": [route],
            }
        }
    }
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_kafka_bus_exposes_public_addressed_transport_identity() -> None:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="192.168.86.105:47092",
            environment="dogfood",
        )
    )

    assert isinstance(bus, ProtocolAddressedBrokerTransport)
    assert bus.bootstrap_servers == "192.168.86.105:47092"
    assert bus.environment == "dogfood"


def test_bounded_route_resolves_broker_from_lane_and_route_from_contract(
    tmp_path: Path,
) -> None:
    resolved = resolve_bounded_delegation_route(
        transport=_AddressedBus(),
        selected_route=_route(),
        overlay_path_for_test=_overlay(tmp_path),
    )

    assert resolved is not None
    assert resolved.lane == "dogfood"
    assert resolved.broker == _AddressedBus.bootstrap_servers
    assert resolved.consumer == "omnimarket.nodes.node_delegation_orchestrator"
    assert resolved.terminal_route == "terminal_events"
    assert resolved.repository_owner == "omnimarket"


@pytest.mark.parametrize("field", ["environment", "bootstrap_servers"])
def test_bounded_route_refuses_empty_transport_identity(
    tmp_path: Path, field: str
) -> None:
    bus = _AddressedBus()
    setattr(bus, field, "  ")

    with pytest.raises(InfraUnavailableError, match="non-empty runtime environment"):
        resolve_bounded_delegation_route(
            transport=bus,
            selected_route=_route(),
            overlay_path_for_test=_overlay(tmp_path),
        )


@pytest.mark.parametrize(
    ("mutation", "route_changes", "message"),
    [
        (("repository_owner", "omnibase_infra"), {}, "repository owner mismatch"),
        (("consumer", "omnimarket.nodes.other"), {}, "consumer mismatch"),
        (("terminal_route", "wrong"), {}, "terminal route does not match"),
        (None, {"terminal_events": ("only-one",)}, "terminal route does not match"),
        (None, {"command_topic": ""}, "no command topic"),
    ],
)
def test_route_contract_mutations_fail_closed(
    tmp_path: Path,
    mutation: tuple[str, str] | None,
    route_changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(InfraUnavailableError, match=message):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(**route_changes),
            overlay_path_for_test=_overlay(tmp_path, mutation),
        )


def test_missing_route_row_and_broker_mismatch_fail_closed(tmp_path: Path) -> None:
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["lanes"]["dogfood"]["delegation_routes"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(InfraUnavailableError, match="exactly one"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(),
            overlay_path_for_test=path,
        )

    mismatch = _overlay(tmp_path, ("repository_owner", "omnimarket"))
    raw = yaml.safe_load(mismatch.read_text(encoding="utf-8"))
    raw["lanes"]["dogfood"]["broker"] = "other-broker:9092"
    mismatch.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(InfraUnavailableError, match="broker mismatch"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(),
            overlay_path_for_test=mismatch,
        )


@pytest.mark.parametrize(
    "missing", [("consumer",), ("terminal_route",), ("repository_owner",)]
)
def test_each_required_row_field_is_required(
    tmp_path: Path, missing: tuple[str]
) -> None:
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["lanes"]["dogfood"]["delegation_routes"][0][missing[0]]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(InfraUnavailableError, match="incomplete or invalid"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(),
            overlay_path_for_test=path,
        )


def test_targeted_broker_with_unexpected_environment_fails_closed(
    tmp_path: Path,
) -> None:
    bus = _AddressedBus()
    bus.environment = "prod"
    with pytest.raises(InfraUnavailableError, match="unexpected runtime environment"):
        resolve_bounded_delegation_route(
            transport=bus,
            selected_route=_route(),
            overlay_path_for_test=_overlay(tmp_path),
        )


def test_targeted_internal_broker_with_unexpected_environment_fails_closed(
    tmp_path: Path,
) -> None:
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["lanes"]["dogfood"]["broker_topology"] = {
        "external_bootstrap_servers": "192.168.86.105:47092",
        "internal_bootstrap_servers": "redpanda:9092",
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    bus = _AddressedBus()
    bus.environment = "prod"
    bus.bootstrap_servers = "redpanda:9092"

    with pytest.raises(InfraUnavailableError, match="unexpected runtime environment"):
        resolve_bounded_delegation_route(
            transport=bus,
            selected_route=_route(),
            overlay_path_for_test=path,
        )


@pytest.mark.asyncio
async def test_dispatch_refuses_mismatch_before_broker_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_broker(*args: object, **kwargs: object) -> None:
        raise AssertionError("broker must not be constructed after route refusal")

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        unexpected_broker,
    )
    route = _route()
    overlay = _overlay(tmp_path, ("consumer", "omnimarket.nodes.wrong"))
    real_resolver = resolve_bounded_delegation_route

    def resolve_with_test_overlay(**kwargs: object) -> object:
        return real_resolver(
            **kwargs,  # type: ignore[arg-type]
            overlay_path_for_test=overlay,
        )

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.resolve_bounded_delegation_route",
        resolve_with_test_overlay,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=_AddressedBus(),  # type: ignore[arg-type]
        routes={
            "omnimarket.node_delegation_orchestrator.delegation.orchestrate": route
        },
    )

    with pytest.raises(InfraUnavailableError, match="consumer mismatch"):
        await port.dispatch(
            prompt="bounded test",
            task_type="document",
            correlation_id=uuid4(),
            max_tokens=128,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=10,
            terminal_delivery_margin_seconds=2,
        )


@pytest.mark.asyncio
async def test_dispatch_refuses_unaddressed_bus_before_broker_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_broker(*args: object, **kwargs: object) -> None:
        raise AssertionError("broker must not be constructed without bus identity")

    monkeypatch.setattr(
        "omnibase_infra.runtime.service_delegation_dispatch_port.RuntimePatternBBroker",
        unexpected_broker,
    )
    port = RuntimeDelegationDispatchPort(
        event_bus=_UnaddressedBus(),  # type: ignore[arg-type]
        routes={
            "omnimarket.node_delegation_orchestrator.delegation.orchestrate": _route()
        },
    )

    with pytest.raises(
        InfraUnavailableError, match="does not expose its configured broker"
    ):
        await port.dispatch(
            prompt="unaddressed bus test",
            task_type="document",
            correlation_id=uuid4(),
            max_tokens=128,
            source_file_path=None,
            source_session_id=None,
            wait=True,
            execution_timeout_seconds=10,
            terminal_delivery_margin_seconds=2,
        )


def test_explicit_internal_topology_identity_is_accepted_only_for_declared_pair(
    tmp_path: Path,
) -> None:
    path = _overlay(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["lanes"]["dogfood"]["broker_topology"] = {
        "external_bootstrap_servers": "192.168.86.105:47092",
        "internal_bootstrap_servers": "redpanda:9092",
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    bus = _AddressedBus()
    bus.bootstrap_servers = "redpanda:9092"

    resolved = resolve_bounded_delegation_route(
        transport=bus,
        selected_route=_route(),
        overlay_path_for_test=path,
    )

    assert resolved is not None
    assert resolved.broker == "192.168.86.105:47092"


def test_undeclared_internal_topology_identity_is_refused(tmp_path: Path) -> None:
    path = _overlay(tmp_path)
    bus = _AddressedBus()
    bus.bootstrap_servers = "redpanda:9092"

    with pytest.raises(InfraUnavailableError, match="broker mismatch"):
        resolve_bounded_delegation_route(
            transport=bus,
            selected_route=_route(),
            overlay_path_for_test=path,
        )
