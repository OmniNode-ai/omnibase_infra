# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: a composed lazy-start boot reports its woken contracts on ``/ready``.

The boot is wired as the kernel wires it: the ingress contract's handler is built by
``wire_from_manifest``, which creates the front door and records it. The readiness
probe's ``lazy_start`` block then reads 0 serving before the first request, 1 after
the dormant reducer is woken by it, and names the contract whose wake failed. The
probe's ``ready`` value is never changed by the block.

RED at the merge base: the ingress handler records no front door and the probe
detail carries no ``lazy_start`` block.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.handlers.lazy_start_boot_split import boot_manifest
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.handlers.lazy_start_publish_gate import _GATE_ATTRIBUTE
from omnibase_infra.models.lazy_start import ModelLazyStartRequest
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    ContractAttachReadinessGate,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.integration.runtime.test_lazy_start_composed_boot_handover_omn21753 import (
    BROKEN,
    BROKEN_TOPIC,
    INGRESS,
    INGRESS_CONTRACT,
    REDUCER,
    REDUCER_TOPIC,
    _dormant,
    _HandoverInmemoryBus,
    lane,
)

pytestmark = pytest.mark.integration

__all__ = ["lane"]


def _lazy_start_block() -> dict[str, object]:
    ready, detail = ContractAttachReadinessGate(()).probe()
    assert ready is True, "the serving read must never change readiness"
    block = detail["lazy_start"]
    assert isinstance(block, dict)
    return block


@pytest.mark.asyncio
async def test_composed_boot_reports_the_woken_contract_and_the_failed_wake(
    lane: Path, tmp_path: Path
) -> None:
    ingress = discover_contracts_from_paths([INGRESS_CONTRACT]).contracts[0]
    reducer = _dormant(tmp_path, REDUCER, REDUCER_TOPIC, handover=True, importable=True)
    broken = _dormant(tmp_path, BROKEN, BROKEN_TOPIC, handover=False, importable=False)
    boot = boot_manifest(ModelAutoWiringManifest(contracts=(ingress, reducer, broken)))
    assert [c.name for c in boot.contracts] == [INGRESS]

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
    front_door = getattr(bus.publish, _GATE_ATTRIBUTE).__self__
    assert isinstance(front_door, LazyStartFrontDoor)

    before = _lazy_start_block()
    assert before["serving_count"] == 0
    assert before["dormant_contract_count"] == 2

    answer = await front_door.handle(
        ModelLazyStartRequest(
            target_contract=REDUCER,
            topic=REDUCER_TOPIC,
            correlation_id=uuid4(),
            envelope=ModelEventEnvelope[object](
                payload={"ask": "first"}, correlation_id=uuid4(), event_type="Fixture"
            ),
        )
    )
    assert answer.outcome is EnumLazyStartOutcome.DELIVERED, answer.detail
    after = _lazy_start_block()
    assert after["serving_contracts"] == [REDUCER]
    assert after["serving_count"] == before["serving_count"] + 1

    await bus.publish(BROKEN_TOPIC, None, b"{}")
    failed = _lazy_start_block()
    failures = failed["wake_failures"]
    assert isinstance(failures, dict)
    assert list(failures) == [BROKEN]
    assert failed["serving_contracts"] == [REDUCER]
    await bus.close()
