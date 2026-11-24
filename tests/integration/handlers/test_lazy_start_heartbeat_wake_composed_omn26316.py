# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316 integration: the runtime start's first heartbeat is not held behind a serial wake.

Drives the real startup handler, publish gate and front door together.

The dev runtime started at 2026-11-23T21:22:29Z and published its first heartbeat
at 21:29:32Z. The heartbeat publish goes through the lazy-start publish gate, whose
``wake_for_topic`` woke the dormant consumers of ``onex.evt.platform.node-heartbeat.v1``
one after another before the publish went on: seven consumers, whose closures woke
20 contracts, four of which ran out the policy's 30 s budget (``did not wake for a
message to 'onex.evt.platform.node-heartbeat.v1'``). The runtime start awaits its
first heartbeat (OMN-24496), so ``/ready`` kept the ``heartbeat`` row false for the
whole serial wake.

The fixture below is wider than the dev boot: 20 dormant consumers of the heartbeat
topic, four of which never finish waking. Times are scaled down; the
bound asserted is one budget, not the sum of every wake.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from omnibase_infra.handlers.handler_runtime_introspection_startup import (
    HandlerRuntimeIntrospectionStartup,
)
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.handlers.lazy_start_publish_gate import install_publish_gate
from omnibase_infra.models.lazy_start import ModelLazyStartPolicy
from omnibase_infra.protocols.protocol_node_introspection import (
    ProtocolNodeIntrospection,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)

pytestmark = pytest.mark.integration

HEARTBEAT_TOPIC = "onex.evt.platform.node-heartbeat.v1"
INGRESS = "node_lazy_start_ingress"
BUDGET_SECONDS = 1.0
WAKE_SECONDS = 0.2
RESPONSIVE = 16
HANGING = 4


def _contract(tmp_path: Path, name: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=tmp_path / name / "contract.yaml",
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(HEARTBEAT_TOPIC,), publish_topics=()
        ),
    )


class _SlowTransaction:
    """Each dormant consumer takes ``WAKE_SECONDS`` to start; hanging ones never do."""

    def __init__(self, hanging: frozenset[str]) -> None:
        self._hanging = hanging
        self.attached: list[str] = []

    async def build(self, contract_name: str) -> None:
        if contract_name in self._hanging:
            await asyncio.Event().wait()
        await asyncio.sleep(WAKE_SECONDS)

    async def attach(self, contract_name: str) -> None:
        self.attached.append(contract_name)

    async def rollback(self, contract_name: str) -> None:
        return None

    async def deliver(self, contract_name: str, topic: str, envelope: object) -> object:
        return None


class _Bus:
    """A bus whose ``publish_envelope`` rides ``publish``, as the Kafka bus does."""

    def __init__(self) -> None:
        self.published: list[tuple[str, float]] = []

    async def publish(self, *args: object, **kwargs: object) -> object:
        self.published.append(
            (str(args[0] if args else kwargs["topic"]), time.monotonic())
        )
        return None

    async def publish_envelope(self, *, envelope: object, topic: str) -> object:
        return await self.publish(topic, None, b"{}")


class _Service:
    """The introspection service's startup confirmation: one heartbeat on the bus."""

    def __init__(self, bus: _Bus) -> None:
        self._bus = bus

    async def publish_heartbeat_now(self) -> bool:
        await self._bus.publish_envelope(envelope=object(), topic=HEARTBEAT_TOPIC)
        return True


def _boot_set(
    tmp_path: Path,
) -> tuple[LazyStartFrontDoor, _SlowTransaction, _Bus, frozenset[str]]:
    names = [
        f"node_dormant_heartbeat_consumer_{i:02d}" for i in range(RESPONSIVE + HANGING)
    ]
    hanging = frozenset(names[:HANGING])
    contracts = {name: _contract(tmp_path, name) for name in names}
    transaction = _SlowTransaction(hanging)
    door = LazyStartFrontDoor(
        transaction,
        contracts,
        ModelLazyStartPolicy(
            ingress_contract=INGRESS, budget_seconds=BUDGET_SECONDS, max_retries=0
        ),
    )
    bus = _Bus()
    assert install_publish_gate(bus, door.wake_for_topic)
    return door, transaction, bus, hanging


@pytest.mark.asyncio
async def test_first_heartbeat_publishes_within_one_wake_budget(tmp_path: Path) -> None:
    """The first heartbeat waits one budget for the slowest wake, not the sum of all.

    Serial: 16 wakes of 0.2 s plus 4 that run out the 1.0 s budget is about 7.2 s.
    Concurrent: about 1.0 s. The readiness window scales to 2.5 s here.
    """
    _door, _transaction, bus, _hanging = _boot_set(tmp_path)
    service = _Service(bus)

    started = time.monotonic()
    await HandlerRuntimeIntrospectionStartup()._confirm_first_heartbeat(
        uuid4(),
        service=cast("ProtocolNodeIntrospection", service),
    )
    elapsed = time.monotonic() - started

    assert [topic for topic, _ in bus.published] == [HEARTBEAT_TOPIC]
    assert elapsed < 2.5, (
        f"the first heartbeat waited {elapsed:.1f}s behind the wake of "
        f"{RESPONSIVE + HANGING} dormant consumers; one wake budget is "
        f"{BUDGET_SECONDS}s"
    )
