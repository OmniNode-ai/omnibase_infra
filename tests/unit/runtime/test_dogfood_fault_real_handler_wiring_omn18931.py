# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exercise the dogfood guard through the real state_io registration arm."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import omnimarket
import pytest
import yaml

import omnibase_infra.runtime.dogfood_delegation_fault_routes as fault_routes
from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelDiscoveredContract,
    ModelHandlerRoutingEntry,
)

pytestmark = pytest.mark.unit


class _RuntimeBus:
    def __init__(self, *, environment: str, bootstrap_servers: str) -> None:
        self.environment = environment
        self.bootstrap_servers = bootstrap_servers
        self.publish_calls = 0

    async def publish(self, *args: object, **kwargs: object) -> None:
        self.publish_calls += 1
        raise AssertionError("the pre-dispatch refusal must prevent publish")

    async def subscribe(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("the pre-dispatch refusal must prevent subscribe")


class _FakeStateStore:
    async def load(self, _key: str) -> None:
        return None

    async def select_recoverable_batches(self) -> list[dict[str, object]]:
        return []

    async def recover_stale_rows(self) -> None:
        return None


class _FakeCodec:
    def flush(self, _correlation_id: str) -> None:
        return None


class _RecordingWorkflow:
    instances: list[_RecordingWorkflow] = []

    def __init__(
        self,
        event_bus: object,
        dispatch_port: object | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.dispatch_port = dispatch_port
        self.handle_calls = 0
        self.__class__.instances.append(self)

    async def handle(self, request: ModelDelegationRequest) -> list[object]:
        self.handle_calls += 1
        return []


def _contract_and_request_entry() -> tuple[
    ModelDiscoveredContract,
    ModelHandlerRoutingEntry,
]:
    market_package_root = Path(omnimarket.__file__).resolve().parent
    contract_path = (
        market_package_root / "nodes" / "node_delegation_orchestrator" / "contract.yaml"
    )
    contract = _parse_contract(
        contract_path=contract_path,
        entry_point_name="node_delegation_orchestrator",
        package_name="omnimarket",
        package_version="0.0.0-test",
    )
    entry = next(
        candidate
        for candidate in contract.handler_routing.handlers
        if candidate.event_model is not None
        and candidate.event_model.name == "ModelDelegationRequest"
    )
    return contract, entry


@pytest.fixture(autouse=True)
def _isolated_state_io_and_lane_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    market_root = Path(omnimarket.__file__).resolve().parents[2]
    lane_path = market_root / "config" / "ci_bus_lanes.yaml"

    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL",
        "postgresql://test-only:unused@127.0.0.1:5432/unused",
    )
    monkeypatch.setattr(
        handler_wiring, "StateStoreAdapter", lambda *args, **kwargs: _FakeStateStore()
    )
    monkeypatch.setattr(handler_wiring, "_read_completion_bound", lambda _path: None)
    monkeypatch.setattr(
        fault_routes,
        "_load_lane_document",
        lambda _path=None: yaml.safe_load(lane_path.read_text(encoding="utf-8")),
    )

    import_handler_class = handler_wiring._import_handler_class

    def _resolve_test_classes(module: str, name: str) -> type:
        if name == "HandlerDelegationWorkflow":
            return _RecordingWorkflow
        if name == "StateIoCodec":
            return _FakeCodec
        return import_handler_class(module, name)

    monkeypatch.setattr(handler_wiring, "_import_handler_class", _resolve_test_classes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment", "backend_id", "message"),
    [
        ("dogfood", "undeclared-fault-backend", "backend pin is not a declared"),
        ("dev", "dogfood-fault-429", "require dogfood lane and broker identity"),
    ],
)
async def test_prepared_state_io_dispatcher_guards_raw_request_before_workflow(
    environment: str,
    backend_id: str,
    message: str,
) -> None:
    _RecordingWorkflow.instances.clear()
    contract, entry = _contract_and_request_entry()
    bus = _RuntimeBus(
        environment=environment,
        bootstrap_servers="192.168.86.105:47092",
    )
    prepared = _prepare_handler_wiring(
        contract=contract,
        entry=entry,
        dispatch_engine=None,
        resolver=ServiceHandlerResolver(),
        ownership_query=ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        ),
        event_bus=bus,
        container=None,
    )
    request = ModelDelegationRequest(
        prompt="raw undeclared route must stop before inference",
        task_type="document",
        correlation_id=uuid4(),
        emitted_at=datetime.now(UTC),
        backend_id=backend_id,
        no_escalation=True,
        requested_timeout_seconds=240,
    )
    envelope = ModelEventEnvelope[object](
        payload=request.model_dump(mode="json"),
        correlation_id=request.correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="omnibase-infra.delegation-request",
        payload_type="ModelDelegationRequest",
        source_tool="omn18931-wiring-test",
    )

    with pytest.raises(InfraUnavailableError, match=message):
        await prepared.dispatcher(envelope)

    assert len(_RecordingWorkflow.instances) == 1
    handler = _RecordingWorkflow.instances[0]
    assert handler.event_bus is bus
    assert handler.handle_calls == 0
    assert bus.publish_calls == 0
