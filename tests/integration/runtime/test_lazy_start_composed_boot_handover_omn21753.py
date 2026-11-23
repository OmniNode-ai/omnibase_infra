# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21753 -- a composed lazy-start boot wakes a handover-declaring contract.

The shape of the dev lane's start-only boot set: a small boot set is wired at
boot, every other contract is dormant. The boot wires through the runtime's own
pieces (boot split, ``wire_from_manifest``, the ingress handler's publish gate)
over the in-process bus. A dormant contract that declares
``consumer_group_handover`` is woken through the front door and attaches with the
handover applied; a second dormant contract whose wake fails leaves the boot and
the heartbeat's publish healthy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    boot_manifest,
)
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.handlers.lazy_start_publish_gate import _GATE_ATTRIBUTE
from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)
from omnibase_infra.models.lazy_start import ModelLazyStartRequest
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

INGRESS_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_lazy_start_ingress/contract.yaml"
)
INGRESS = "node_lazy_start_ingress"
REDUCER = "node_schedule_run_reducer"
BROKEN = "node_schedule_run_broken"
REDUCER_TOPIC = "onex.cmd.platform.schedule-run.v1"
BROKEN_TOPIC = "onex.cmd.platform.schedule-run-broken.v1"
HEARTBEAT_TOPIC = "onex.evt.platform.node-heartbeat.v1"

_HANDLED: list[str] = []


class _Handler:
    async def handle(self, request: Any) -> None:
        _HANDLED.append(type(request).__name__)


class _HandoverInmemoryBus(EventBusInmemory):
    """The in-process bus, with the handover recorded so a test can read it."""

    def __init__(self) -> None:
        super().__init__(environment="local")
        self.handovers: list[tuple[str, str, str]] = []

    async def hand_over_committed_offsets(
        self,
        *,
        topic: str,
        from_group: str,
        to_group: str,
        from_auto_offset_reset: Literal["earliest", "latest"],
        handover_rule: Literal[
            "copy_if_uncommitted", "last_ran"
        ] = "copy_if_uncommitted",
    ) -> ModelOffsetHandoverResult:
        self.handovers.append((topic, from_group, to_group))
        return ModelOffsetHandoverResult(
            topic=topic, from_group=from_group, to_group=to_group
        )


def _dormant(
    root: Path, name: str, topic: str, *, handover: bool, importable: bool
) -> ModelDiscoveredContract:
    directory = root / name
    directory.mkdir()
    path = directory / "contract.yaml"
    text = (
        f"name: {name}\nnode_type: REDUCER_GENERIC\ncontract_version: 1.0.0\n"
        f"description: OMN-21753 composed boot fixture\nevent_bus:\n"
        f"  subscribe_topics:\n    - {topic}\n"
    )
    if handover:
        text += (
            "consumer_group_handover:\n  from_service: omnimarket-projections\n"
            "  from_node: schedule-run-writer\n  from_version: v1\n"
            "  from_auto_offset_reset: earliest\n"
        )
    path.write_text(text, encoding="utf-8")
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=path,
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="_Handler" if importable else "_MissingHandler",
                        module=__name__,
                    ),
                    event_model=None,
                ),
            ),
        ),
    )


@pytest.fixture
def lane(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The lane's hot-load row: on_demand, a start-only boot set (the ingress)."""
    root = tmp_path / "overlays"
    (root / "runtime_contract_hotload").mkdir(parents=True)
    (root / "runtime_contract_hotload" / "overlay.yaml").write_text(
        "schema_version: 1\nlanes:\n  dogfood:\n    enabled: true\n"
        "    mode: on_demand\n    allowed_environments: [local]\n"
        "    lazy_start:\n"
        f"      ingress_contract: {INGRESS}\n"
        f"      boot_contracts: [{INGRESS}]\n"
        "      budget_seconds: 30\n      max_retries: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dogfood")
    monkeypatch.delenv("ONEX_BOOT_UNIVERSE_PROVISION", raising=False)
    StoreLazyStartDormant.clear()
    return root


@pytest.mark.asyncio
async def test_composed_boot_wakes_a_handover_contract_and_survives_a_failed_wake(
    lane: Path, tmp_path: Path
) -> None:
    ingress = discover_contracts_from_paths([INGRESS_CONTRACT]).contracts[0]
    reducer = _dormant(tmp_path, REDUCER, REDUCER_TOPIC, handover=True, importable=True)
    broken = _dormant(tmp_path, BROKEN, BROKEN_TOPIC, handover=False, importable=False)
    full = ModelAutoWiringManifest(contracts=(ingress, reducer, broken))

    # Boot: the boot set is wired, the rest is dormant.
    boot = boot_manifest(full)
    assert [c.name for c in boot.contracts] == [INGRESS]
    assert set(StoreLazyStartDormant.contracts()) == {REDUCER, BROKEN}

    bus = _HandoverInmemoryBus()
    await bus.start()
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, bus)
    report = await wire_from_manifest(
        boot,
        MessageDispatchEngine(),
        environment="local",
        subscribe_immediately=False,
        container=container,
        event_bus=bus,
    )
    assert report.total_failed == 0
    # Wiring the ingress contract built its handler, which put the wake-on-publish
    # gate on the bus; the gate's owner is this boot's front door.
    gate = getattr(bus.publish, _GATE_ATTRIBUTE)
    front_door = gate.__self__
    assert isinstance(front_door, LazyStartFrontDoor)

    # The dormant reducer declares a handover: its wake attaches, handover applied.
    envelope = ModelEventEnvelope[object](
        payload={"ask": "first"}, correlation_id=uuid4(), event_type="Fixture"
    )
    answer = await front_door.handle(
        ModelLazyStartRequest(
            target_contract=REDUCER,
            topic=REDUCER_TOPIC,
            correlation_id=uuid4(),
            envelope=envelope,
        )
    )
    assert answer.outcome is EnumLazyStartOutcome.DELIVERED, answer.detail
    assert [h[0] for h in bus.handovers] == [REDUCER_TOPIC]
    assert bus.handovers[0][1] == (
        "local.omnimarket-projections.schedule-run-writer.consume.v1"
    )

    # A deliberately failing second wake, reached by a publish as the heartbeat's is:
    # the publish returns, the boot's bus keeps serving, the reducer stays attached.
    await bus.publish(BROKEN_TOPIC, None, b"{}")
    await bus.publish(HEARTBEAT_TOPIC, None, b"{}")
    assert BROKEN in front_door.wake_failures
    assert BROKEN not in front_door.serving
    assert REDUCER in front_door.serving
    assert (
        await front_door.handle(
            ModelLazyStartRequest(
                target_contract=BROKEN,
                topic=BROKEN_TOPIC,
                correlation_id=uuid4(),
                envelope=envelope,
            )
        )
    ).outcome is EnumLazyStartOutcome.START_FAILED
    await bus.close()
