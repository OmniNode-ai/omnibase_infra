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
`EventBusInmemory` path against a stand-in contract that takes the same dispatch
branch as the deployed node: `payload_type_match` with a typed `event_model`
and no message category (the typed branch of `_make_dispatch_callback`, which
validates the payload into the request model and normalizes the handler
result). It asserts the handler completed once with a typed request, that the
run logged no ERROR record, and that the undeliverable-output DLQ route
(`_route_apply_publish_failure`) is never awaited -- then proves the assertion
is falsifiable by showing the same wiring DOES take that route for a handler of
the unfixed shape (returns a BaseModel, no publish_topics).

The handlers here are stand-ins with the signature of the real
`HandlerHostTableKeysIngest.handle(request) -> None` (omnimarket). omnibase_infra
does not depend on omnimarket, so the real handler cannot be imported in this
test environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    UndeliverableDispatchOutputError,
    wire_from_manifest,
)
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


_CONTRACT_YAML = f"""\
name: node_host_table_keys_ingest_effect
node_type: EFFECT_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - operation: handle
      handler:
        name: HandlerHostTableKeysIngest
        module: omnimarket.nodes.node_host_table_keys_ingest_effect.handlers.handler_host_table_keys_ingest
      event_model:
        name: ModelOverlayStoreRecord
        module: omnimarket.models.overlay_store.model_overlay_store
      topic: {_SUBSCRIBE_TOPIC}
event_bus:
  subscribe_topics:
    - {_SUBSCRIBE_TOPIC}
  publish_topics: []
"""


class _CountingHandler:
    """Counts the dispatches that reach a handler and the ones that complete.

    ``calls`` is bumped on entry. ``received`` is appended only after the
    handler's own checks pass, so a handler that crashes (the engine turns a
    handler exception into a failed result that the boundary swallows) leaves
    ``received`` empty instead of looking like a clean dispatch.
    """

    calls: ClassVar[int] = 0
    received: ClassVar[list[object]] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0
        cls.received = []


class HandlerFixedShape(_CountingHandler):
    """Mirrors the post-fix handler: ``handle(request) -> None``, nothing to deliver."""

    def handle(self, request: ModelFixtureOverlayStoreRecord) -> None:
        type(self).calls += 1
        assert isinstance(request, ModelFixtureOverlayStoreRecord)
        assert request.overlay_id == "lab_hosts"
        type(self).received.append(request)


class HandlerUnfixedShape(_CountingHandler):
    """Mirrors the pre-fix handler: ``handle(request) -> <BaseModel>``."""

    def handle(
        self, request: ModelFixtureOverlayStoreRecord
    ) -> ModelFixtureIngestResult:
        type(self).calls += 1
        assert isinstance(request, ModelFixtureOverlayStoreRecord)
        assert request.overlay_id == "lab_hosts"
        type(self).received.append(request)
        return ModelFixtureIngestResult(
            outcome="written", key="synthetic:hosts:v1:table"
        )


def _no_publish_topics_contract(
    handler_qualname: str, contract_path: Path
) -> ModelDiscoveredContract:
    """Reproduce this node's contract shape: payload_type_match, subscribe, no publish."""
    return ModelDiscoveredContract(
        name="node_host_table_keys_ingest_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_host_table_keys_ingest_effect",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_SUBSCRIBE_TOPIC,),
            # No publish_topics -- the OMN-24693 shape.
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_qualname,
                        module=__name__,
                    ),
                    event_model=ModelHandlerRef(
                        name=ModelFixtureOverlayStoreRecord.__name__,
                        module=__name__,
                    ),
                    topic=_SUBSCRIBE_TOPIC,
                    operation="handle",
                    # No message_category: the deployed contract declares none.
                ),
            ),
        ),
    )


async def _dispatch_one(
    handler_cls: type[_CountingHandler], contract_path: Path
) -> None:
    handler_cls.reset()
    contract = _no_publish_topics_contract(handler_cls.__qualname__, contract_path)
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


_NORMALIZE = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._normalize_handler_result"
)


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    """A contract file that exists, so wiring reads real declarations."""
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT_YAML, encoding="utf-8")
    return path


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fixed_handler_shape_never_routes_undeliverable_output(
    contract_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AC2: the fixed handler shape completes and produces no undeliverable output."""
    caplog.set_level(logging.WARNING)
    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._route_apply_publish_failure",
            new_callable=AsyncMock,
        ) as route_failure,
        patch(_NORMALIZE, wraps=handler_wiring._normalize_handler_result) as normalize,
    ):
        await _dispatch_one(HandlerFixedShape, contract_path)

    # AC1: the dispatch took the typed `payload_type_match` branch the deployed
    # node takes (it validates into `event_model`, then normalizes the result).
    # The untyped `topic_match` branch never calls the normalizer for a handler
    # that returns None, so this fails if the stand-in routing shape drifts.
    normalize.assert_called_once()
    assert normalize.call_args.args[2] == ModelFixtureOverlayStoreRecord.__name__

    # The handler ran to completion with the typed request. A handler that
    # crashed (or got an untyped payload) is swallowed by the boundary and would
    # otherwise look like "no undeliverable output".
    assert len(HandlerFixedShape.received) == 1
    request = HandlerFixedShape.received[0]
    assert isinstance(request, ModelFixtureOverlayStoreRecord)
    assert request.overlay_id == "lab_hosts"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not [r for r in caplog.records if "FileNotFoundError" in r.getMessage()]
    route_failure.assert_not_awaited()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unfixed_handler_shape_does_route_undeliverable_output(
    contract_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Falsifier: the same wiring DOES flag the unfixed shape as undeliverable.

    Proves the assertion above is not vacuous -- this harness distinguishes
    the fixed shape from the defective one on the identical contract wiring
    (a handler returning a BaseModel, no publish_topics).
    """
    caplog.set_level(logging.WARNING)
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._route_apply_publish_failure",
        new_callable=AsyncMock,
    ) as route_failure:
        await _dispatch_one(HandlerUnfixedShape, contract_path)

    assert len(HandlerUnfixedShape.received) == 1
    route_failure.assert_awaited_once()
    assert route_failure.await_args is not None
    (exc,), kwargs = route_failure.await_args
    assert isinstance(exc, UndeliverableDispatchOutputError)
    assert kwargs["topic"] == _SUBSCRIBE_TOPIC
    # The "no result applier" ERROR line is the signal the live-log check greps
    # for; it also proves caplog sees ERROR records in the fixed-shape test.
    assert any(
        r.levelno == logging.ERROR and "has no result applier" in r.getMessage()
        for r in caplog.records
    )
    assert not [r for r in caplog.records if "FileNotFoundError" in r.getMessage()]
