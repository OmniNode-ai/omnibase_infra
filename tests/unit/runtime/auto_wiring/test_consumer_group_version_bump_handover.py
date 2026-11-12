# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: a contract version bump loses no rows.

Auto-wiring names a projection's consumer group after its contract version, so a
bump (``1.1.2`` to ``1.2.0`` on the work ledger projection, 2026-11-11) joins a
NEW group. That group has no committed offset, starts at the bus default
``latest`` and never reads what was published between the old group stopping and
the new one joining: 39 rows at the 1.1.1 to 1.1.2 bump, 16 at 1.1.2 to 1.2.0.

These tests run the real path as ``test_consumer_group_handover_omn21288.py``
does (a contract YAML on disk, the real ``wire_from_manifest``, the real
per-topic callback and dispatch engine) on ``EventBusSemanticFake``, the
in-process substrate that models group joins, committed offsets and
``auto_offset_reset``. ``_VersionedFake`` gives it the two things the
version handover asks of a bus: the handover itself, through the shared
``plan_offset_handover`` rule, and the listing of the groups it holds.
``test_consumer_group_version_bump_redpanda_omn19998.py`` runs the scenario
against a Redpanda broker.

The failing control is ``test_without_version_handover_the_gap_is_lost``: the
switch off is the behaviour before this change, and the same scenario loses the
gap.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    ConsumerGroupFenceError,
    plan_offset_handover,
)
from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)
from omnibase_infra.models.event_bus.model_version_handover_policy import (
    VERSION_HANDOVER_ENV_VAR,
    load_version_handover_policy,
    select_version_predecessors,
)
from omnibase_infra.protocols.protocol_consumer_group_lister import (
    ProtocolConsumerGroupLister,
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
TOPIC = "onex.evt.omnimarket.work-ledger-status.v1"
NODE = "projection_work_ledger"
GROUP_PREFIX = f"{ENVIRONMENT}.omnimarket.{NODE}.consume"

_CONTRACT_BASE = f"""
name: {NODE}
node_type: REDUCER_GENERIC
contract_version: {{version}}
description: OMN-19998 projection fixture
event_bus:
  subscribe_topics:
    - {TOPIC}
"""

_projected: list[str] = []


class _ProjectionWriter:
    """Records each event id it projects, as the live-events writer upserts it."""

    async def handle(self, request: dict[str, Any]) -> None:
        _projected.append(str(request["payload"]["event_id"]))


class _VersionedFake(EventBusSemanticFake):
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
        self._leavers: dict[str, Callable[[], Awaitable[None]]] = {}
        # Groups whose consumer is still running: the handover is fenced off them.
        self.groups_with_members: set[str] = set()

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
        before = set(self._group_members)
        unsubscribe = await super().subscribe(
            topic,
            node_identity,
            fetch,
            group_id=group_id,
            purpose=purpose,
            required_for_readiness=required_for_readiness,
            auto_offset_reset=auto_offset_reset,
        )
        for _joined_topic, joined_group in set(self._group_members) - before:
            self._leavers[joined_group] = unsubscribe
        return unsubscribe

    async def leave(self, group: str) -> None:
        """The consumer of ``group`` stops: no member, its commit stays."""
        await self._leavers.pop(group)()

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
        handover_rule: Literal[
            "copy_if_uncommitted", "last_ran"
        ] = "copy_if_uncommitted",
    ) -> ModelOffsetHandoverResult:
        if from_group in self.groups_with_members:
            raise ConsumerGroupFenceError(
                f"retired consumer group {from_group!r} still has members"
            )
        async with self._lock:
            successor = self._group_committed_offset.get((topic, to_group))
            plan = plan_offset_handover(
                partitions=(0,),  # the semantic fake models one partition per topic
                successor_committed={0: successor},
                retired_committed={
                    0: self._group_committed_offset.get((topic, from_group))
                },
                first_offsets={0: self._history_base_offset.get(topic, 0)},
                end_offsets={0: self._topic_offsets.get(topic, 0)},
                retired_auto_offset_reset=from_auto_offset_reset,
                handover_rule=handover_rule,
            )
            if 0 in plan:
                self._group_committed_offset[(topic, to_group)] = plan[0]
        return ModelOffsetHandoverResult(
            topic=topic,
            from_group=from_group,
            to_group=to_group,
            committed=plan,
            kept=(0,) if successor is not None and 0 not in plan else (),
        )

    async def list_consumer_groups(self) -> list[str]:
        """Every group the fake holds a member or a committed offset for."""
        return sorted(
            {group for (_t, group) in self._group_members}
            | {group for (_t, group) in self._group_committed_offset}
        )


def _contract(tmp_path: Path, version: tuple[int, int, int]) -> ModelDiscoveredContract:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        _CONTRACT_BASE.format(version=".".join(map(str, version))), encoding="utf-8"
    )
    major, minor, patch_ = version
    return ModelDiscoveredContract(
        name=NODE,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=major, minor=minor, patch=patch_),
        contract_path=contract_path,
        entry_point_name=NODE,
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


def _group(version: str) -> str:
    return f"{GROUP_PREFIX}.{version}"


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


async def _bump(
    tmp_path: Path,
) -> tuple[_VersionedFake, list[str], list[str]]:
    """1.1.1 consumes and stops, the deploy gap passes, 1.2.0 joins.

    Returns the bus, what 1.1.1 projected and what 1.2.0 projected.
    """
    _projected.clear()
    bus = _VersionedFake(environment=ENVIRONMENT)
    await bus.start()

    await _wire(_contract(tmp_path, (1, 1, 1)), bus)
    await _publish(bus, ["a-0", "a-1"])
    await bus.settle()
    await bus.leave(_group("1.1.1"))  # the rebuild stops the old runtime
    old = list(_projected)

    await _publish(bus, ["gap-0", "gap-1"])  # nobody is joined

    await _wire(_contract(tmp_path, (1, 2, 0)), bus)
    await _publish(bus, ["after-0"])
    await bus.settle()
    return bus, old, list(_projected)[len(old) :]


@pytest.mark.asyncio
async def test_version_bump_loses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: the new version projects the gap and what follows, nothing twice."""
    monkeypatch.delenv(VERSION_HANDOVER_ENV_VAR, raising=False)
    bus, old, new = await _bump(tmp_path)
    await bus.stop_loops()

    assert old == ["a-0", "a-1"]
    assert new == ["gap-0", "gap-1", "after-0"], (
        "the new version must start at the old group's committed offset"
    )
    every = Counter(old + new)
    assert set(every.values()) == {1}
    assert len(every) == 5


@pytest.mark.asyncio
async def test_without_version_handover_the_gap_is_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2, the failing control: switched off, the new group skips the gap."""
    monkeypatch.setenv(VERSION_HANDOVER_ENV_VAR, '{"enabled": false}')
    bus, old, new = await _bump(tmp_path)
    await bus.stop_loops()

    assert old == ["a-0", "a-1"]
    assert new == ["after-0"]


@pytest.mark.asyncio
async def test_rollback_never_rewinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: back to 1.1.1 takes the later commit of 1.2.0, replaying nothing."""
    monkeypatch.delenv(VERSION_HANDOVER_ENV_VAR, raising=False)
    bus, _old, _new = await _bump(tmp_path)
    await bus.leave(_group("1.2.0"))
    before = len(_projected)

    await _publish(bus, ["back-0"])
    await _wire(_contract(tmp_path, (1, 1, 1)), bus)
    await bus.settle()
    await bus.stop_loops()

    assert _projected[before:] == ["back-0"]


@pytest.mark.asyncio
async def test_a_predecessor_with_members_is_skipped_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """AC4, error chain: a live peer cannot stop the new version from joining."""
    monkeypatch.delenv(VERSION_HANDOVER_ENV_VAR, raising=False)
    _projected.clear()
    bus = _VersionedFake(environment=ENVIRONMENT)
    await bus.start()
    await _wire(_contract(tmp_path, (1, 1, 1)), bus)
    await _publish(bus, ["a-0"])
    await bus.settle()
    bus.groups_with_members.add(_group("1.1.1"))

    with caplog.at_level(logging.WARNING):
        await _wire(_contract(tmp_path, (1, 2, 0)), bus)

    await _publish(bus, ["after-0"])
    await bus.settle()
    await bus.stop_loops()
    assert _group("1.1.1") in caplog.text
    assert "still has members" in caplog.text
    assert _projected == ["a-0", "after-0", "after-0"], (
        "both versions are live peers on the one topic and each reads it"
    )


@pytest.mark.parametrize("value", ["not json", "[1]", '{"enabled": "maybe"}'])
@pytest.mark.asyncio
async def test_an_unreadable_policy_refuses_the_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """AC4, error chain: a policy that cannot be read never runs as the default."""
    monkeypatch.setenv(VERSION_HANDOVER_ENV_VAR, value)
    bus = _VersionedFake(environment=ENVIRONMENT)
    await bus.start()

    with pytest.raises(ValueError):
        await _wire(_contract(tmp_path, (1, 2, 0)), bus)

    assert bus._group_members == {}


@pytest.mark.asyncio
async def test_a_bus_that_cannot_list_groups_has_no_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bus with no group state wires as before: nothing to hand over."""
    monkeypatch.delenv(VERSION_HANDOVER_ENV_VAR, raising=False)
    plain = EventBusSemanticFake(environment=ENVIRONMENT)
    await plain.start()
    assert not isinstance(plain, ProtocolConsumerGroupLister)

    await _wire(_contract(tmp_path, (1, 2, 0)), plain)

    assert any(group == _group("1.2.0") for (_t, group) in plain._group_members)


_SUCCESSOR = f"{GROUP_PREFIX}.1.2.0.__t.onex.evt.x.v1"


@pytest.mark.parametrize(
    ("listed", "expected"),
    [
        (
            [
                _SUCCESSOR,
                f"{GROUP_PREFIX}.1.1.2.__t.onex.evt.x.v1",
                f"{GROUP_PREFIX}.1.1.1.__t.onex.evt.x.v1",
                f"{GROUP_PREFIX}.1.1.2.__t.onex.evt.OTHER.v1",
                f"{ENVIRONMENT}.omnimarket.other_node.consume.1.1.2.__t.onex.evt.x.v1",
                f"{GROUP_PREFIX}.1.1.2.__i.inst.__t.onex.evt.x.v1",
                "not-a-group",
            ],
            (
                f"{GROUP_PREFIX}.1.1.1.__t.onex.evt.x.v1",
                f"{GROUP_PREFIX}.1.1.2.__t.onex.evt.x.v1",
            ),
        ),
        ([_SUCCESSOR], ()),
        ([], ()),
    ],
    ids=["same-node-same-topic-other-versions", "only-itself", "nothing-listed"],
)
def test_select_version_predecessors(
    listed: list[str], expected: tuple[str, ...]
) -> None:
    assert select_version_predecessors(_SUCCESSOR, listed) == expected


def test_a_group_with_no_version_segment_has_no_predecessors() -> None:
    assert select_version_predecessors("some-adhoc-group", ["some-adhoc-group"]) == ()


def test_policy_defaults_and_overrides() -> None:
    default = load_version_handover_policy({})
    assert default.enabled is True
    assert default.from_auto_offset_reset == "latest"
    assert (
        load_version_handover_policy(
            {VERSION_HANDOVER_ENV_VAR: '{"enabled": false}'}
        ).enabled
        is False
    )
    assert (
        load_version_handover_policy(
            {VERSION_HANDOVER_ENV_VAR: '{"fence": {"wait_seconds": 5}}'}
        ).fence.wait_seconds
        == 5
    )
