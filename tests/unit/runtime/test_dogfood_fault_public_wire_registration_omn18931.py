# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Exercise public delegate-skill wire parsing through actual auto-wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import omnimarket
import pytest
from omnimarket.nodes.node_delegate_skill_orchestrator.models.model_delegate_skill_request import (
    ModelDelegateSkillRequest,
)

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelDiscoveredContract,
    ModelHandlerRoutingEntry,
)

pytestmark = pytest.mark.unit


class _RuntimeBus:
    environment = "dogfood"
    bootstrap_servers = "192.168.86.105:47092"


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


class _RecordingDelegateHandler:
    instances: list[_RecordingDelegateHandler] = []

    def __init__(self, event_bus: object, dispatch_port: object | None = None) -> None:
        self.event_bus = event_bus
        self.dispatch_port = dispatch_port
        self.requests: list[ModelDelegateSkillRequest] = []
        self.__class__.instances.append(self)

    async def handle(self, request: ModelDelegateSkillRequest) -> list[object]:
        self.requests.append(request)
        return []


def _contract_and_entry() -> tuple[ModelDiscoveredContract, ModelHandlerRoutingEntry]:
    market_package_root = Path(omnimarket.__file__).resolve().parent
    contract = _parse_contract(
        contract_path=(
            market_package_root
            / "nodes"
            / "node_delegate_skill_orchestrator"
            / "contract.yaml"
        ),
        entry_point_name="node_delegate_skill_orchestrator",
        package_name="omnimarket",
        package_version="0.0.0-test",
    )
    entry = next(
        candidate
        for candidate in contract.handler_routing.handlers
        if candidate.event_model is not None
        and candidate.event_model.name == "ModelDelegateSkillRequest"
    )
    return contract, entry


@pytest.fixture(autouse=True)
def _isolated_state_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL",
        "postgresql://test-only:unused@127.0.0.1:5432/unused",
    )
    monkeypatch.setattr(
        handler_wiring, "StateStoreAdapter", lambda *args, **kwargs: _FakeStateStore()
    )
    monkeypatch.setattr(handler_wiring, "_read_completion_bound", lambda _path: None)
    import_handler_class = handler_wiring._import_handler_class

    def _resolve_test_handler(module: str, name: str) -> type:
        if name == "HandlerDelegateSkill":
            return _RecordingDelegateHandler
        if name == "StateIoCodec":
            return _FakeCodec
        return import_handler_class(module, name)

    monkeypatch.setattr(handler_wiring, "_import_handler_class", _resolve_test_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_id", ["dogfood-fault-429", "dogfood-fault-503"])
async def test_registered_public_consumer_deserializes_fault_wire(
    backend_id: str,
) -> None:
    """The declared public event model reaches the registered handler unchanged."""
    _RecordingDelegateHandler.instances.clear()
    contract, entry = _contract_and_entry()
    bus = _RuntimeBus()
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
    request = ModelDelegateSkillRequest(
        prompt="exercise declared isolated fault route",
        task_type="document",
        source="codex",
        correlation_id=uuid4(),
        backend_id=backend_id,
        no_escalation=True,
        requested_timeout_seconds=240,
    )
    envelope = ModelEventEnvelope[object](
        payload=request.model_dump(mode="json"),
        correlation_id=request.correlation_id,
        envelope_timestamp=datetime.now(UTC),
        event_type="omnimarket.delegate-skill",
        payload_type="ModelDelegateSkillRequest",
        source_tool="omn18931-public-wire-test",
    )

    await prepared.dispatcher(envelope)

    assert len(_RecordingDelegateHandler.instances) == 1
    received = _RecordingDelegateHandler.instances[0].requests
    assert len(received) == 1
    assert received[0].backend_id == backend_id
    assert received[0].no_escalation is True
    assert received[0].requested_timeout_seconds == 240
