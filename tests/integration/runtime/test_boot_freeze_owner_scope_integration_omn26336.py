# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot order with the GC heap frozen and the owner scope kept at registration.

OMN-26336: the kernel wires auto-wired contracts, freezes the engine with
``freeze(freeze_gc_heap=True)``, then attaches consumers, and
``subscribe_wired_contract_topics`` proves each contract's dispatcher scope against the
owner scopes the engine kept at registration. This runs that sequence with the real
auto-wiring functions and the real engine.
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_CONTRACT_NAME = "node_boot_freeze_owner_scope_probe"
_TOPIC = "onex.cmd.omnimarket.boot-freeze-owner-scope-probe.v1"


class HandlerBootFreezeOwnerScopeProbe:
    async def handle(self, envelope: object) -> None:
        return None


@pytest.fixture
def clean_gc_permanent_generation() -> Iterator[None]:
    gc.unfreeze()
    try:
        yield
    finally:
        gc.unfreeze()


def _manifest() -> ModelAutoWiringManifest:
    contract = ModelDiscoveredContract(
        name=_CONTRACT_NAME,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("boot-freeze-owner-scope/contract.yaml"),
        entry_point_name=_CONTRACT_NAME,
        package_name="test-package",
        event_bus=ModelEventBusWiring(subscribe_topics=(_TOPIC,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerBootFreezeOwnerScopeProbe",
                        module=__name__,
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )
    return ModelAutoWiringManifest(contracts=(contract,), errors=())


@pytest.mark.asyncio
async def test_wired_contract_attaches_after_the_engine_freezes_with_the_gc_heap(
    clean_gc_permanent_generation: None,
) -> None:
    manifest = _manifest()
    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerBootFreezeOwnerScopeProbe,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
            subscribe_immediately=False,
        )
        assert report.total_wired == 1

        frozen_before = gc.get_freeze_count()
        dispatch_engine.freeze(freeze_gc_heap=True)
        assert gc.get_freeze_count() > frozen_before

        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    event_bus.subscribe.assert_awaited_once()
    assert subscriptions == {_CONTRACT_NAME: (_TOPIC,)}
