# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21753: a lazy wake of a handover-declaring contract attaches, and a failed
lazy wake fails that contract only.

The dev runtime of 2026-11-22T16:59:29Z woke ``node_schedule_run_reducer`` on its
first event. The wake handed ``HandlerContractSubscribePreflight`` the
``RecordingBus`` wrapper of ``LazyStartRuntimeTransaction``, and the preflight
refused it: ``supports_offset_handover`` tests the bus with ``isinstance`` against
a ``runtime_checkable`` Protocol, and since Python 3.12 that test reads members
through ``inspect.getattr_static``, which never calls the wrapper's
``__getattr__``. The refusal then reached the heartbeat publish and the boot.
The recorded lines are in ``tests/fixtures/lazy_start_omn21753``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.handlers.lazy_start_publish_gate import install_publish_gate
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
)
from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)
from omnibase_infra.models.lazy_start import (
    ModelLazyStartPolicy,
    ModelLazyStartRequest,
)
from omnibase_infra.protocols.protocol_lazy_start_bus import ProtocolLazyStartBus
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/lazy_start_omn21753/dev_boot_log_2026-11-22.txt"
)
REDUCER = "node_schedule_run_reducer"
FAILING = "node_schedule_run_failing"
TOPIC = "onex.cmd.platform.schedule-run.v1"
FAILING_TOPIC = "onex.cmd.platform.schedule-run-failing.v1"

_HANDLED: list[str] = []


class _Handler:
    async def handle(self, request: Any) -> None:
        _HANDLED.append(type(request).__name__)


class _PlainBus:
    """A bus with no handover capability: records subscriptions."""

    def __init__(self, *, refuse_topics: frozenset[str] = frozenset()) -> None:
        self.environment = "local"
        self.refuse_topics = refuse_topics
        self.live: set[str] = set()
        self.published: list[str] = []

    async def subscribe(self, *args: object, **kwargs: object) -> object:
        topic = str(kwargs["topic"])
        if topic in self.refuse_topics:
            raise ConnectionError(f"join refused for {topic}")
        self.live.add(topic)

        async def _stop() -> None:
            self.live.discard(topic)

        return _stop

    async def publish_envelope(self, *args: object, **kwargs: object) -> object:
        return None

    async def publish(self, *args: object, **kwargs: object) -> object:
        self.published.append(str(args[0] if args else kwargs["topic"]))
        return None

    def get_consumer_groups(self) -> dict[tuple[str, str], str]:
        return {}


class _HandoverBus(_PlainBus):
    """A bus whose class supports handover: also records each handover."""

    def __init__(self, *, refuse_topics: frozenset[str] = frozenset()) -> None:
        super().__init__(refuse_topics=refuse_topics)
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


def _contract(
    tmp_path: Path, name: str, topic: str, *, handover: bool
) -> ModelDiscoveredContract:
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "contract.yaml"
    text = (
        f"name: {name}\nnode_type: REDUCER_GENERIC\ncontract_version: 1.0.0\n"
        f"description: OMN-21753 fixture\nevent_bus:\n  subscribe_topics:\n"
        f"    - {topic}\n"
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
                    handler=ModelHandlerRef(name="_Handler", module=__name__),
                    event_model=None,
                ),
            ),
        ),
    )


def _transaction(
    contracts: dict[str, ModelDiscoveredContract], bus: _PlainBus
) -> LazyStartRuntimeTransaction:
    return LazyStartRuntimeTransaction(
        contracts,
        cast("ProtocolLazyStartBus", bus),
        "local",
        None,
        ModelONEXContainer(),
        ModelTopicReadinessConfig(),
    )


def _recorded_refusal() -> str:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    (found,) = (line for line in lines if line.startswith("TypeError: contract '"))
    return found.removeprefix("TypeError: ")


def test_the_fixture_holds_the_recorded_refusal_and_wake_failure() -> None:
    """Positive control: the fixture parses and names the recorded chain."""
    text = FIXTURE.read_text(encoding="utf-8")
    refusal = _recorded_refusal()
    assert re.fullmatch(
        r"contract '(\w+)' declares consumer_group_handover, but the wired event "
        r"bus \((\w+)\) cannot hand a retired group's committed offsets over; "
        r"joining at the bus default would drop the handover window \(OMN-21753\)",
        refusal,
    )
    assert "RuntimeError: contract 'node_schedule_run_reducer' did not attach" in text
    assert "BOOT FAILED 1/5" in text


@pytest.mark.asyncio
async def test_a_lazy_wake_of_a_handover_declaring_contract_attaches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The recorded refusal does not recur: the wake attaches, handover applied."""
    contract = _contract(tmp_path, REDUCER, TOPIC, handover=True)
    bus = _HandoverBus()
    transaction = _transaction({REDUCER: contract}, bus)
    recorded = _recorded_refusal()

    await transaction.build(REDUCER)
    try:
        await transaction.attach(REDUCER)
    except RuntimeError as exc:
        # The wake's error names only the exception class; the refusal's own text
        # is in the log, as in the dev runtime's.
        assert recorded not in caplog.text, f"reproduced the recorded refusal: {exc}"
        raise

    assert bus.live == {TOPIC}
    assert [h[0] for h in bus.handovers] == [TOPIC]
    assert (
        bus.handovers[0][1]
        == "local.omnimarket-projections.schedule-run-writer.consume.v1"
    )


@pytest.mark.asyncio
async def test_the_wrapper_exposes_only_the_capabilities_the_bus_has(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Control: over a bus with no handover the wake is still refused, not widened."""
    contract = _contract(tmp_path, REDUCER, TOPIC, handover=True)

    bus = _PlainBus()
    transaction = _transaction({REDUCER: contract}, bus)
    await transaction.build(REDUCER)

    with pytest.raises(RuntimeError, match="did not attach"):
        await transaction.attach(REDUCER)
    assert bus.live == set()
    assert "declares consumer_group_handover" in caplog.text
    assert "(_PlainBus) cannot hand" in caplog.text


async def _door(
    tmp_path: Path, bus: _PlainBus
) -> tuple[LazyStartFrontDoor, LazyStartRuntimeTransaction]:
    contracts = {
        REDUCER: _contract(tmp_path, REDUCER, TOPIC, handover=True),
        FAILING: _contract(tmp_path, FAILING, FAILING_TOPIC, handover=False),
    }
    transaction = _transaction(contracts, bus)
    policy = ModelLazyStartPolicy(
        ingress_contract=REDUCER, budget_seconds=10.0, max_retries=0
    )
    door = LazyStartFrontDoor(transaction, contracts, policy)
    assert install_publish_gate(bus, door.wake_for_topic)
    return door, transaction


@pytest.mark.asyncio
async def test_a_failed_wake_fails_that_contract_only(tmp_path: Path) -> None:
    """A publish to a dormant consumer whose wake fails is not failed by the wake."""
    bus = _HandoverBus(refuse_topics=frozenset({FAILING_TOPIC}))
    door, transaction = await _door(tmp_path, bus)

    # The failing contract's topic is published to (as the heartbeat publish is):
    # the publish goes through and the failure is that contract's outcome only.
    await bus.publish(FAILING_TOPIC, None, b"{}")

    assert bus.published == [FAILING_TOPIC]
    assert FAILING in door.wake_failures
    assert "did not attach" in door.wake_failures[FAILING]
    assert bus.live == set()
    assert FAILING not in door.serving
    assert transaction.node_engine(FAILING) is None

    # The other contract still wakes and serves, and the heartbeat topic publishes.
    envelope = ModelEventEnvelope[object](
        payload={"ask": "first"}, correlation_id=uuid4(), event_type="Fixture"
    )
    answer = await door.handle(
        ModelLazyStartRequest(
            target_contract=REDUCER,
            topic=TOPIC,
            correlation_id=uuid4(),
            envelope=envelope,
        )
    )
    assert answer.outcome is EnumLazyStartOutcome.DELIVERED, answer.detail
    assert bus.live == {TOPIC}
    await bus.publish("onex.evt.platform.node-heartbeat.v1", None, b"{}")
    assert bus.published[-1] == "onex.evt.platform.node-heartbeat.v1"
