# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18852: the contract key must actually reach the bus, in the right order.

The loader and the consume loop are each tested on their own. This file tests
the link between them, which is the half that can rot silently: a
``consume_concurrency`` block that parses correctly and is never handed to
anything changes nothing on the lane and looks exactly like a working
declaration.

Two orderings are load-bearing and both are asserted rather than reasoned
about:

* the declaration happens **before** ``subscribe``, because ``subscribe``
  starts the consume loop and the loop reads the bound once at start. Declared
  afterwards it takes effect on the next consumer rebuild -- i.e. never, in
  the run the operator is watching;
* the declaration is scoped to the topics the contract actually subscribes to,
  so a bound is never recorded against a key nothing reads.

A bus that cannot bound in-flight records must REFUSE a contract that declares
one. Accepting it and running serial is the exact failure mode this ticket
exists to remove, one layer up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
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

SUBSCRIBE_TOPIC = "onex.cmd.omnibase-infra.delegation-inference-request.v1"
PUBLISH_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"

_CONTRACT_BASE = f"""
name: node_omn18852_effect
node_type: EFFECT_GENERIC
contract_version: 1.0.0
description: fixture
event_bus:
  subscribe_topics:
    - {SUBSCRIBE_TOPIC}
  publish_topics:
    - {PUBLISH_TOPIC}
"""


class _RecordingBus(EventBusInmemory):
    """The real in-memory bus, instrumented to record wiring call order."""

    def __init__(self) -> None:
        super().__init__(environment="test", group="omn18852-wiring")
        self.calls: list[tuple[str, str, int | None]] = []

    def declare_consume_concurrency(
        self, *, topic: str, group_id: str, max_in_flight_records: int
    ) -> None:
        self.calls.append(("declare", topic, max_in_flight_records))

    async def subscribe(self, *args: Any, **kwargs: Any) -> Any:
        topic = kwargs.get("topic") or (args[0] if args else "")
        self.calls.append(("subscribe", str(topic), None))
        return await super().subscribe(*args, **kwargs)


class _PlainRecordingBus(EventBusInmemory):
    """``EventBusInmemory`` as it ships: no in-flight bound at all."""

    def __init__(self) -> None:
        super().__init__(environment="test", group="omn18852-wiring-plain")
        self.calls: list[tuple[str, str, int | None]] = []

    async def subscribe(self, *args: Any, **kwargs: Any) -> Any:
        topic = kwargs.get("topic") or (args[0] if args else "")
        self.calls.append(("subscribe", str(topic), None))
        return await super().subscribe(*args, **kwargs)


class _Handler:
    async def handle(self, request: object) -> object:
        return request


def _contract(tmp_path: Path, *, declaration: str) -> ModelDiscoveredContract:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CONTRACT_BASE + declaration, encoding="utf-8")
    return ModelDiscoveredContract(
        name="node_omn18852_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_omn18852_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(SUBSCRIBE_TOPIC,),
            publish_topics=(PUBLISH_TOPIC,),
        ),
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


async def _wire(contract: ModelDiscoveredContract, bus: EventBusInmemory) -> None:
    """Wire through the REAL manifest path, not a hand-called internal."""
    engine = MessageDispatchEngine()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_Handler,
    ):
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(contract,)),
            engine,
            event_bus=bus,
            environment="local",
        )


@pytest.mark.asyncio
async def test_declaration_reaches_the_bus_before_subscribe(tmp_path: Path) -> None:
    """The bound must be recorded before the loop that reads it starts."""
    bus = _RecordingBus()
    contract = _contract(
        tmp_path,
        declaration="consume_concurrency:\n  max_in_flight_records: 4\n",
    )

    await _wire(contract, bus)

    assert bus.calls == [
        ("declare", SUBSCRIBE_TOPIC, 4),
        ("subscribe", SUBSCRIBE_TOPIC, None),
    ], (
        "the bound must be declared for the exact subscribed topic and strictly "
        f"before subscribe starts its consume loop; got {bus.calls}"
    )


@pytest.mark.asyncio
async def test_an_undeclared_contract_touches_nothing(tmp_path: Path) -> None:
    """The unchanged path must not even call the declarer."""
    bus = _RecordingBus()
    contract = _contract(tmp_path, declaration="")

    await _wire(contract, bus)

    assert bus.calls == [("subscribe", SUBSCRIBE_TOPIC, None)]


@pytest.mark.asyncio
async def test_an_undeclared_contract_wires_on_a_bus_without_the_capability(
    tmp_path: Path,
) -> None:
    """Every existing node must keep wiring on every existing transport."""
    bus = _PlainRecordingBus()
    contract = _contract(tmp_path, declaration="")

    await _wire(contract, bus)

    assert bus.calls == [("subscribe", SUBSCRIBE_TOPIC, None)]


@pytest.mark.asyncio
async def test_a_bus_that_cannot_bound_refuses_a_declared_contract(
    tmp_path: Path,
) -> None:
    """Silently running serial under a declared bound is the defect itself."""
    bus = _PlainRecordingBus()
    contract = _contract(
        tmp_path,
        declaration="consume_concurrency:\n  max_in_flight_records: 4\n",
    )

    with pytest.raises(TypeError, match="cannot bound in-flight records"):
        await _wire(contract, bus)

    assert bus.calls == [], "nothing may be subscribed once the bound is refused"


@pytest.mark.asyncio
async def test_a_malformed_declaration_fails_wiring(tmp_path: Path) -> None:
    """Not degraded to 1: an operator believing a bound is in force is the bug."""
    bus = _RecordingBus()
    contract = _contract(tmp_path, declaration="consume_concurrency: 4\n")

    with pytest.raises(ValueError, match="not a mapping"):
        await _wire(contract, bus)

    assert bus.calls == []
