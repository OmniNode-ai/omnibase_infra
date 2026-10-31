# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21753 (under OMN-21288): a folded writer resumes its retired container's offsets.

The one-runtime fold stops a projection writer container and lets the main
runtime consume the writer's contract. The runtime's group is new, and a new
group starts at the bus default ``latest``, so every event published between
the container stopping and the runtime joining was never projected.

These tests run the real path -- a contract YAML on disk, the real
``wire_from_manifest``, the real per-topic callback and the real dispatch
engine -- on ``EventBusSemanticFake``, the in-process substrate that models
consumer-group joins, committed offsets and ``auto_offset_reset`` (OMN-15789).
``tests/integration/event_bus/test_consumer_group_handover_redpanda_omn21288.py``
runs the same scenario against a Redpanda broker.

The handover on this substrate is implemented by ``_HandoverSemanticFake``
below through ``plan_offset_handover``, the same rule the Kafka path of
``hand_over_committed_offsets`` applies; that function asks a bus that
implements ``ProtocolConsumerGroupOffsetHandover`` directly.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest

from omnibase_core.enums.enum_consumer_group_purpose import EnumConsumerGroupPurpose
from omnibase_core.event_bus.event_bus_semantic_fake import EventBusSemanticFake
from omnibase_core.event_bus.protocol_node_identity_like import (
    ProtocolNodeIdentityLike,
)
from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    hand_over_committed_offsets,
    load_consumer_group_handover,
    plan_offset_handover,
    supports_offset_handover,
)
from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)
from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)
from omnibase_infra.protocols.protocol_consumer_group_offset_handover import (
    ProtocolConsumerGroupOffsetHandover,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import ModelHandlerRef
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit

ENVIRONMENT = "local"
TOPIC = "onex.evt.platform.node-heartbeat.v1"
# The retired live-events writer's group on the dev lane, byte for byte as
# docker/docker-compose.dev-lane.yml sets KAFKA_CONSUMER_GROUP.
RETIRED_GROUP = "local.omnimarket-projections.live-events-writer.consume.v1"

DECLARATION = """
consumer_group_handover:
  from_service: omnimarket-projections
  from_node: live-events-writer
  from_version: v1
  from_auto_offset_reset: earliest
"""

_CONTRACT_BASE = f"""
name: projection_live_events
node_type: REDUCER_GENERIC
contract_version: 1.3.0
description: OMN-21753 folded-writer fixture
event_bus:
  subscribe_topics:
    - {TOPIC}
"""

_projected: list[str] = []


class _ProjectionWriter:
    """Records each event id it projects, as the live-events writer upserts it."""

    async def handle(self, request: dict[str, Any]) -> None:
        _projected.append(str(request["payload"]["event_id"]))


class _HandoverSemanticFake(EventBusSemanticFake):
    """The group-offset substrate, given the handover through the shared rule.

    ``EventBusSemanticFake`` replays a joining group's backlog inline, inside
    ``subscribe``. Auto-wiring subscribes before it freezes the dispatch engine,
    and dispatch waits for the freeze, so an inline replay would wait on
    itself. A broker delivers from a consume loop, after ``subscribe`` returns;
    each subscription here gets one such loop. Offsets and group membership
    stay the fake's own.
    """

    def __init__(self, environment: str) -> None:
        super().__init__(environment=environment)
        self._loops: list[
            tuple[asyncio.Task[None], asyncio.Queue[ModelEventMessage]]
        ] = []

    async def subscribe(
        self,
        topic: str,
        node_identity: ProtocolNodeIdentityLike | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
        auto_offset_reset: str = "latest",
    ) -> Callable[[], Awaitable[None]]:
        assert on_message is not None
        handler = on_message
        queue: asyncio.Queue[ModelEventMessage] = asyncio.Queue()

        async def consume_loop() -> None:
            while True:
                message = await queue.get()
                try:
                    await handler(message)
                finally:
                    queue.task_done()

        async def fetch(message: ModelEventMessage) -> None:
            queue.put_nowait(message)

        self._loops.append((asyncio.create_task(consume_loop()), queue))
        return await super().subscribe(
            topic,
            node_identity,
            fetch,
            group_id=group_id,
            purpose=purpose,
            required_for_readiness=required_for_readiness,
            auto_offset_reset=auto_offset_reset,
        )

    async def settle(self) -> None:
        """Wait until every consume loop has handled everything it fetched."""
        for _, queue in self._loops:
            await queue.join()

    async def stop_loops(self) -> None:
        for task, _ in self._loops:
            task.cancel()
        await asyncio.gather(*(task for task, _ in self._loops), return_exceptions=True)

    async def hand_over_committed_offsets(
        self,
        *,
        topic: str,
        from_group: str,
        to_group: str,
        from_auto_offset_reset: Literal["earliest", "latest"],
    ) -> ModelOffsetHandoverResult:
        async with self._lock:
            successor = self._group_committed_offset.get((topic, to_group))
            plan = plan_offset_handover(
                partitions=(0,),
                successor_committed={0: successor},
                retired_committed={
                    0: self._group_committed_offset.get((topic, from_group))
                },
                first_offsets={0: self._history_base_offset.get(topic, 0)},
                end_offsets={0: self._topic_offsets.get(topic, 0)},
                retired_auto_offset_reset=from_auto_offset_reset,
            )
            if 0 in plan:
                self._group_committed_offset[(topic, to_group)] = plan[0]
        return ModelOffsetHandoverResult(
            topic=topic,
            from_group=from_group,
            to_group=to_group,
            committed=plan,
            kept=(0,) if successor is not None else (),
        )


class _BusWithoutHandover:
    """A group-offset bus with no handover capability."""

    def __init__(self) -> None:
        self._bus = EventBusSemanticFake(environment=ENVIRONMENT)
        self.subscribed: list[str] = []

    async def subscribe(self, *args: Any, **kwargs: Any) -> Any:
        self.subscribed.append(str(kwargs.get("topic")))
        return await self._bus.subscribe(*args, **kwargs)

    async def publish(self, *args: Any, **kwargs: Any) -> Any:
        return await self._bus.publish(*args, **kwargs)


def _contract(tmp_path: Path, *, declaration: str) -> ModelDiscoveredContract:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CONTRACT_BASE + declaration, encoding="utf-8")
    return ModelDiscoveredContract(
        name="projection_live_events",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=3, patch=0),
        contract_path=contract_path,
        entry_point_name="projection_live_events",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(TOPIC,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="_ProjectionWriter", module=__name__),
                    event_model=None,
                ),
            ),
        ),
    )


async def _wire(contract: ModelDiscoveredContract, bus: object) -> None:
    """Wire through the real manifest path, then open dispatch."""
    engine = MessageDispatchEngine()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_ProjectionWriter,
    ):
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(contract,)),
            engine,
            event_bus=bus,
            environment=ENVIRONMENT,
        )
    engine.freeze()


async def _publish(bus: EventBusSemanticFake, event_ids: list[str]) -> None:
    for event_id in event_ids:
        envelope = {
            "event_type": TOPIC,
            "payload": {"event_id": event_id},
            "correlation_id": "6f1c3b0e-9a43-4a7e-8c55-2b0b2a8c1d11",
        }
        await bus.publish(TOPIC, event_id.encode(), json.dumps(envelope).encode())


async def _run_the_fold(
    tmp_path: Path, *, declaration: str
) -> tuple[_HandoverSemanticFake, list[str], list[str]]:
    """Container consumes, retires, the window passes, the runtime joins.

    Returns the bus, what the retired container projected and what the
    runtime projected.
    """
    _projected.clear()
    bus = _HandoverSemanticFake(environment=ENVIRONMENT)
    await bus.start()

    container_projected: list[str] = []

    async def container(message: Any) -> None:
        container_projected.append(json.loads(message.value)["payload"]["event_id"])

    leave = await bus.subscribe(
        TOPIC,
        group_id=RETIRED_GROUP,
        on_message=container,
        auto_offset_reset="earliest",
    )
    await _publish(bus, ["before-0", "before-1", "before-2"])
    await bus.settle()
    await leave()  # the fold retires the writer container

    await _publish(bus, ["window-0", "window-1"])  # nobody is joined

    await _wire(_contract(tmp_path, declaration=declaration), bus)
    await _publish(bus, ["after-0", "after-1"])
    await bus.settle()
    await bus.stop_loops()
    return bus, container_projected, list(_projected)


@pytest.mark.asyncio
async def test_window_events_are_projected_exactly_once(tmp_path: Path) -> None:
    """AC1: every event, before, during and after the handover, exactly once."""
    _, container, runtime = await _run_the_fold(tmp_path, declaration=DECLARATION)

    assert container == ["before-0", "before-1", "before-2"]
    assert runtime == ["window-0", "window-1", "after-0", "after-1"], (
        "the runtime must start at the retired group's committed offset: the "
        "window events once, and nothing the container already projected"
    )
    every = Counter(container + runtime)
    assert set(every.values()) == {1}
    assert len(every) == 7


@pytest.mark.asyncio
async def test_without_declaration_the_window_is_lost(tmp_path: Path) -> None:
    """AC2: the defect, reproduced. A new group at ``latest`` skips the window."""
    _, container, runtime = await _run_the_fold(tmp_path, declaration="")

    assert container == ["before-0", "before-1", "before-2"]
    assert runtime == ["after-0", "after-1"]


@pytest.mark.asyncio
async def test_handover_is_idempotent_across_a_runtime_restart(tmp_path: Path) -> None:
    """AC4: the second start keeps the runtime's own position, never rewinds."""
    bus, _, runtime = await _run_the_fold(tmp_path, declaration=DECLARATION)
    successor = next(
        group for (topic, group) in list(bus._group_members) if group != RETIRED_GROUP
    )
    own_offset = await bus.get_committed_offset(TOPIC, successor)
    assert own_offset == 7

    second = await bus.hand_over_committed_offsets(
        topic=TOPIC,
        from_group=RETIRED_GROUP,
        to_group=successor,
        from_auto_offset_reset="earliest",
    )

    assert second.committed == {}
    assert second.kept == (0,)
    assert await bus.get_committed_offset(TOPIC, successor) == own_offset
    assert runtime == ["window-0", "window-1", "after-0", "after-1"]


@pytest.mark.asyncio
async def test_wiring_refuses_a_declared_handover_on_a_bus_without_it(
    tmp_path: Path,
) -> None:
    """AC3: refused, not joined at ``latest`` as if nothing were declared."""
    bus = _BusWithoutHandover()
    await bus._bus.start()
    assert not isinstance(bus, ProtocolConsumerGroupOffsetHandover)

    with pytest.raises(TypeError, match="consumer_group_handover"):
        await _wire(_contract(tmp_path, declaration=DECLARATION), bus)

    assert bus.subscribed == []


@pytest.mark.parametrize(
    "declaration",
    [
        "consumer_group_handover: live-events-writer\n",
        (
            "consumer_group_handover:\n  from_service: omnimarket-projections\n"
            "  from_node: live-events-writer\n  from_version: v1\n"
        ),
        (
            "consumer_group_handover:\n  from_service: omnimarket-projections\n"
            "  from_node: live-events-writer\n  from_version: v1\n"
            "  from_auto_offset_reset: newest\n"
        ),
    ],
    ids=["not-a-mapping", "no-reset-policy", "unknown-reset-policy"],
)
@pytest.mark.asyncio
async def test_wiring_refuses_a_malformed_declaration(
    tmp_path: Path, declaration: str
) -> None:
    """AC3: a declaration that cannot be read never wires as if absent."""
    bus = _HandoverSemanticFake(environment=ENVIRONMENT)
    await bus.start()

    with pytest.raises(ValueError):
        await _wire(_contract(tmp_path, declaration=declaration), bus)

    assert bus._group_members == {}


def test_source_group_matches_the_retired_container_group(tmp_path: Path) -> None:
    """The declaration names the exact group the dev-lane container used."""
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT_BASE + DECLARATION, encoding="utf-8")

    handover = load_consumer_group_handover(path)

    assert handover == ModelConsumerGroupHandover(
        from_service="omnimarket-projections",
        from_node="live-events-writer",
        from_version="v1",
        from_auto_offset_reset="earliest",
    )
    assert handover.source_group(ENVIRONMENT) == RETIRED_GROUP
    assert handover.source_group("other-lane") == (
        "other-lane.omnimarket-projections.live-events-writer.consume.v1"
    )


def test_an_undeclared_or_missing_contract_has_no_handover(tmp_path: Path) -> None:
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT_BASE, encoding="utf-8")
    assert load_consumer_group_handover(path) is None
    assert load_consumer_group_handover(tmp_path / "absent.yaml") is None


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("earliest", {0: 40, 1: 12, 2: 5}), ("latest", {0: 40, 1: 30, 2: 9})],
)
def test_plan_covers_every_partition_rule(
    policy: Literal["earliest", "latest"], expected: dict[int, int]
) -> None:
    """Kept, copied, never committed, and committed below retention."""
    plan = plan_offset_handover(
        partitions=(0, 1, 2, 3),
        successor_committed={3: 77},
        retired_committed={0: 40, 1: None, 2: 3},
        first_offsets={0: 0, 1: 12, 2: 5, 3: 0},
        end_offsets={0: 50, 1: 30, 2: 9, 3: 80},
        retired_auto_offset_reset=policy,
    )
    assert plan == expected


@pytest.mark.asyncio
async def test_inmemory_bus_has_nothing_to_hand_over() -> None:
    """No group on the inline bus holds a position, so nothing is committed."""
    bus = EventBusInmemory(environment=ENVIRONMENT)
    assert supports_offset_handover(bus)

    result = await hand_over_committed_offsets(
        bus,
        topic=TOPIC,
        from_group=RETIRED_GROUP,
        to_group="local.omnimarket.projection_live_events.consume.1.3.0",
        from_auto_offset_reset="earliest",
    )

    assert result.committed == {}
    assert result.kept == ()


@pytest.mark.asyncio
async def test_handover_refuses_a_bus_it_cannot_serve() -> None:
    bus = _BusWithoutHandover()
    assert not supports_offset_handover(bus)
    with pytest.raises(TypeError, match="cannot hand a retired consumer group"):
        await hand_over_committed_offsets(
            bus,
            topic=TOPIC,
            from_group=RETIRED_GROUP,
            to_group="successor",
            from_auto_offset_reset="earliest",
        )
