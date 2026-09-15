# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof for OMN-18390 through the real auto-wiring boundary.

`node_delegation_chain_ledger_effect`'s contract declares `subscribe_topics`
with no `publish_topics`. Before the fix, `handle()` returned a BaseModel on
`ModelHandlerOutput.result`, which the auto-wiring boundary
(`handler_wiring.py`) appends to `output_events` regardless of contract
wiring -- with no `publish_topics` there is no result applier, so every
dispatch produced undeliverable output and dead-lettered as
`UndeliverableDispatchOutputError` (one per chain-canary run).

This exercises the REAL `wire_from_manifest` + `MessageDispatchEngine` +
`EventBusInmemory` path (not a unit-level mock of the boundary) against a
contract shaped exactly like the real one (subscribe_topics, no
publish_topics), and asserts the undeliverable-output DLQ route
(`_route_apply_publish_failure`) is never taken for the fixed handler shape --
then proves the assertion is falsifiable by showing the same wiring DOES take
that route for a handler using the pre-fix shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.dispatch import ModelHandlerOutput
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

_SUBSCRIBE_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"


class ModelFixtureWriteResult(BaseModel):
    """Stand-in for ModelLedgerChainWriteResult -- any BaseModel reproduces it."""

    correlation_id: UUID
    rows_written: int


class HandlerFixedShape:
    """Mirrors the post-fix handler: effect output, nothing to deliver."""

    calls: ClassVar[int] = 0

    async def handle(self, envelope: object) -> ModelHandlerOutput[None]:
        type(self).calls += 1
        correlation_id = _extract_correlation_id(envelope)
        return ModelHandlerOutput.for_effect(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id="fixture-fixed-shape",
        )


class HandlerPreFixShape:
    """Mirrors the pre-fix handler: for_compute(result=<BaseModel>)."""

    calls: ClassVar[int] = 0

    async def handle(
        self, envelope: object
    ) -> ModelHandlerOutput[ModelFixtureWriteResult]:
        type(self).calls += 1
        correlation_id = _extract_correlation_id(envelope)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id="fixture-pre-fix-shape",
            result=ModelFixtureWriteResult(
                correlation_id=correlation_id, rows_written=4
            ),
        )


def _extract_correlation_id(envelope: object) -> UUID:
    if isinstance(envelope, dict):
        # The dispatch engine injects this trace before invoking handlers; the
        # assertion below proves the message reached the handler rather than
        # passing vacuously before dispatch.
        debug_trace = envelope["__debug_trace"]
        assert isinstance(debug_trace, dict)
        return UUID(str(debug_trace["correlation_id"]))
    return UUID(str(envelope.correlation_id))  # type: ignore[attr-defined]


def _no_publish_topics_contract(handler_qualname: str) -> ModelDiscoveredContract:
    """Reproduce this node's real contract shape: subscribe, no publish."""
    return ModelDiscoveredContract(
        name="node_delegation_chain_ledger_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(
            "/tmp/node_delegation_chain_ledger_effect/contract.yaml"  # noqa: S108
        ),
        entry_point_name="node_delegation_chain_ledger_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_SUBSCRIBE_TOPIC,),
            # No publish_topics -- the exact OMN-18390 shape.
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
                    operation="delegation_chain.record",
                    message_category="event",
                ),
            ),
        ),
    )


async def _dispatch_one(handler_cls: type) -> None:
    handler_cls.calls = 0
    contract = _no_publish_topics_contract(handler_cls.__qualname__)
    bus = EventBusInmemory(environment="test", group="omn18390-no-undeliverable")
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
            payload={"status": "completed"},
            correlation_id=uuid4(),
            event_type="omnimarket.delegate-skill-completed",
        )
        await bus.publish(
            _SUBSCRIBE_TOPIC,
            None,
            command.model_dump_json().encode("utf-8"),
            None,
        )
        # EventBusInmemory awaits subscriber callbacks inline before returning
        # from publish(), so this is a synchronization point for the dispatch.
        assert handler_cls.calls == 1
    finally:
        await bus.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fixed_handler_shape_never_routes_undeliverable_output() -> None:
    """AC1: the fixed handler shape produces no undeliverable dispatch output."""
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._route_apply_publish_failure",
        new_callable=AsyncMock,
    ) as route_failure:
        await _dispatch_one(HandlerFixedShape)

    route_failure.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pre_fix_handler_shape_does_route_undeliverable_output() -> None:
    """Falsifier: the same wiring DOES flag the pre-fix shape as undeliverable.

    Proves the assertion above is not vacuous -- this harness distinguishes
    the fixed shape from the defective one on the identical contract wiring.
    """
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._route_apply_publish_failure",
        new_callable=AsyncMock,
    ) as route_failure:
        await _dispatch_one(HandlerPreFixShape)

    route_failure.assert_awaited_once()
    assert route_failure.await_args is not None
    (exc,), kwargs = route_failure.await_args
    assert type(exc).__name__ == "UndeliverableDispatchOutputError"
    assert kwargs["topic"] == _SUBSCRIBE_TOPIC
