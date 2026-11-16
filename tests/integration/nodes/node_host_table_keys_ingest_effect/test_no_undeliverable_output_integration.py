# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof for OMN-24693 through the real auto-wiring boundary.

`node_host_table_keys_ingest_effect` (omnimarket) subscribes to
`onex.evt.omnimarket.overlay-store-loaded.v1` and declares no `publish_topics`.
Before the fix its contract also declared an `output_model` and its handler
returned a BaseModel. Auto-wiring builds a result applier only from
`publish_topics`, so with none declared every successful dispatch took the
undeliverable-output path (an ERROR line plus a dead-letter route as
`UndeliverableDispatchOutputError`). The fix mirrors OMN-18390: drop the unused
`output_model` and return no model from `handle()`.

This exercises the REAL `wire_from_manifest` + `MessageDispatchEngine` +
`EventBusInmemory` path against a stand-in contract of the fixed shape
(subscribe_topics, no publish_topics) and asserts the undeliverable-output DLQ
route (`_route_apply_publish_failure`) is never awaited -- then proves the
assertion is falsifiable by showing the same wiring DOES take that route for a
handler of the unfixed shape (returns a BaseModel, no publish_topics).

The handlers here are stand-ins with the signature of the real
`HandlerHostTableKeysIngest.handle(request) -> None` (omnimarket). omnibase_infra
does not depend on omnimarket, so the real handler cannot be imported in this
test environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
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

_SUBSCRIBE_TOPIC = "onex.evt.omnimarket.overlay-store-loaded.v1"


class ModelFixtureOverlayStoreRecord(BaseModel):
    """Stand-in for omnimarket's ModelOverlayStoreRecord (the handler's request)."""

    overlay_id: str
    digest: str


class ModelFixtureIngestResult(BaseModel):
    """Stand-in for the deleted ModelHostTableKeysIngestResult -- any BaseModel reproduces it."""

    outcome: str
    key: str


class _CountingHandler:
    """Counts the dispatches that reach a handler."""

    calls: ClassVar[int] = 0


class HandlerFixedShape(_CountingHandler):
    """Mirrors the post-fix handler: ``handle(request) -> None``, nothing to deliver."""

    def handle(self, request: ModelFixtureOverlayStoreRecord) -> None:
        type(self).calls += 1
        assert request.overlay_id == "lab_hosts"


class HandlerUnfixedShape(_CountingHandler):
    """Mirrors the pre-fix handler: ``handle(request) -> <BaseModel>``."""

    def handle(
        self, request: ModelFixtureOverlayStoreRecord
    ) -> ModelFixtureIngestResult:
        type(self).calls += 1
        assert request.overlay_id == "lab_hosts"
        return ModelFixtureIngestResult(
            outcome="written", key="synthetic:hosts:v1:table"
        )


def _no_publish_topics_contract(handler_qualname: str) -> ModelDiscoveredContract:
    """Reproduce this node's contract shape: subscribe, no publish."""
    return ModelDiscoveredContract(
        name="node_host_table_keys_ingest_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("node_host_table_keys_ingest_effect/contract.yaml"),
        entry_point_name="node_host_table_keys_ingest_effect",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_SUBSCRIBE_TOPIC,),
            # No publish_topics -- the OMN-24693 shape.
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="topic_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_qualname,
                        module=__name__,
                    ),
                    topic=_SUBSCRIBE_TOPIC,
                    operation="handle",
                    message_category="event",
                ),
            ),
        ),
    )


async def _dispatch_one(handler_cls: type[_CountingHandler]) -> None:
    handler_cls.calls = 0
    contract = _no_publish_topics_contract(handler_cls.__qualname__)
    bus = EventBusInmemory(environment="test", group="omn24693-no-undeliverable")
    await bus.start()
    try:
        engine = MessageDispatchEngine()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=handler_cls,
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(contract,)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        command = ModelEventEnvelope[object](
            payload={"overlay_id": "lab_hosts", "digest": "synthetic-digest"},
            correlation_id=uuid4(),
            event_type="omnimarket.overlay-store-loaded",
        )
        await bus.publish(
            _SUBSCRIBE_TOPIC,
            None,
            command.model_dump_json().encode("utf-8"),
            None,
        )
        # EventBusInmemory awaits subscriber callbacks inline before returning
        # from publish(), so this is a synchronization point for the dispatch:
        # the message reached the handler rather than passing vacuously.
        assert handler_cls.calls == 1
    finally:
        await bus.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fixed_handler_shape_never_routes_undeliverable_output() -> None:
    """AC2: the fixed handler shape produces no undeliverable dispatch output."""
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._route_apply_publish_failure",
        new_callable=AsyncMock,
    ) as route_failure:
        await _dispatch_one(HandlerFixedShape)

    route_failure.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unfixed_handler_shape_does_route_undeliverable_output() -> None:
    """Falsifier: the same wiring DOES flag the unfixed shape as undeliverable.

    Proves the assertion above is not vacuous -- this harness distinguishes
    the fixed shape from the defective one on the identical contract wiring
    (a handler returning a BaseModel, no publish_topics).
    """
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._route_apply_publish_failure",
        new_callable=AsyncMock,
    ) as route_failure:
        await _dispatch_one(HandlerUnfixedShape)

    route_failure.assert_awaited_once()
    assert route_failure.await_args is not None
    (exc,), kwargs = route_failure.await_args
    assert type(exc).__name__ == "UndeliverableDispatchOutputError"
    assert kwargs["topic"] == _SUBSCRIBE_TOPIC
