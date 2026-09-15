# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof for OMN-18398 through the real dispatch path.

`public.ledger_chain` held ZERO rows on the .201 compose dev lane while
`node_delegation_chain_ledger_effect` reported `status=success` on every
dispatch. The mechanism is two statements composing:
`_persist_rows` is a `for row in rows:` loop, so it executes no statement at
all over an empty row set, and `handle()` returned its effect output
unconditionally afterwards -- a write of zero rows was indistinguishable from
a write of four.

This exercises the REAL `wire_from_manifest` + `MessageDispatchEngine` +
`EventBusInmemory` path against the REAL handler class, with only the
`public.event_ledger` read stubbed (that relation is the input, not the
subject). It asserts that an empty evidence read reaches the wiring's
error route rather than completing, and proves the assertion is not vacuous
with a positive control: the SAME wiring, the SAME handler, a non-empty
evidence read, persists and takes no error route.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    HandlerDelegationChainLedger,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    ModelObservedHop,
)
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
_CHAIN = (
    "onex.cmd.omnimarket.delegate-skill.v1",
    "onex.cmd.omnibase-infra.delegation-routing-request.v1",
    "onex.evt.omnibase-infra.routing-decision.v1",
    _SUBSCRIBE_TOPIC,
)


def _observed(correlation_id: UUID) -> tuple[ModelObservedHop, ...]:
    envelope_ids = tuple(uuid4() for _ in _CHAIN)
    return tuple(
        ModelObservedHop(
            topic=topic,
            envelope_id=envelope_ids[index],
            parent_envelope_id=None if index == 0 else envelope_ids[index - 1],
            correlation_id=correlation_id,
        )
        for index, topic in enumerate(_CHAIN)
    )


class HandlerUnderTest(HandlerDelegationChainLedger):
    """The REAL handler with only its two database edges stubbed.

    ``handle()`` -- the subject -- is inherited untouched. The wiring
    constructs the handler itself, so the stubs are installed in ``__init__``
    rather than on a pre-built instance, and the constructor takes no required
    parameter (the auto-wiring resolver quarantines a handler whose required
    ``container`` argument it cannot supply in this harness).
    """

    observed: ClassVar[tuple[ModelObservedHop, ...]] = ()
    instances: ClassVar[list[HandlerUnderTest]] = []

    def __init__(self) -> None:
        super().__init__(
            cast("Any", object()),
            db_dsn="postgresql://test.invalid/test",
            declared_chain=_CHAIN,
            settle_attempts=1,
            settle_delay_seconds=0,
        )
        self._ensure_db_ready = AsyncMock()  # type: ignore[method-assign]
        self._read_observed = AsyncMock(  # type: ignore[method-assign]
            return_value=type(self).observed
        )
        self._persist_rows = AsyncMock()  # type: ignore[method-assign]
        type(self).instances.append(self)


def _real_contract() -> ModelDiscoveredContract:
    """This node's real shape: bus-triggered effect, no publish_topics."""
    return ModelDiscoveredContract(
        name="node_delegation_chain_ledger_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(
            "/tmp/node_delegation_chain_ledger_effect/contract.yaml"  # noqa: S108
        ),
        entry_point_name="node_delegation_chain_ledger_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(subscribe_topics=(_SUBSCRIBE_TOPIC,)),
        handler_routing=ModelHandlerRouting(
            routing_strategy="topic_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerUnderTest",
                        module=__name__,
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelDelegationTerminalPayload",
                        module=(
                            "omnibase_infra.nodes."
                            "node_delegation_chain_ledger_effect.models"
                        ),
                    ),
                    topic=_SUBSCRIBE_TOPIC,
                    operation="delegation_chain.record",
                    message_category="event",
                ),
            ),
        ),
    )


async def _dispatch_with_evidence(
    observed: tuple[ModelObservedHop, ...], correlation_id: UUID
) -> HandlerUnderTest:
    """Dispatch one terminal event through the real wiring; return the handler."""
    HandlerUnderTest.observed = observed
    HandlerUnderTest.instances = []

    bus = EventBusInmemory(environment="test", group="omn18398-absent-evidence")
    await bus.start()
    try:
        engine = MessageDispatchEngine()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerUnderTest,
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(_real_contract(),)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        envelope = ModelEventEnvelope[object](
            payload={"correlation_id": str(correlation_id), "status": "completed"},
            correlation_id=correlation_id,
            event_type="omnimarket.delegate-skill-completed",
        )
        # EventBusInmemory awaits subscriber callbacks inline before returning,
        # so publish() is the synchronization point for the dispatch.
        await bus.publish(
            _SUBSCRIBE_TOPIC,
            None,
            envelope.model_dump_json().encode("utf-8"),
            None,
        )
    finally:
        await bus.close()

    assert len(HandlerUnderTest.instances) == 1, (
        "the wiring did not construct exactly one handler, so nothing was "
        f"dispatched: {HandlerUnderTest.instances!r}"
    )
    return HandlerUnderTest.instances[0]


_BOUNDARY_ERROR_LOGGER = "omnibase_infra.runtime.auto_wiring.handler_wiring"
_ENGINE_ERROR_LOGGER = "omnibase_infra.runtime.message_dispatch_engine"
# The boundary truncates the error it quotes; the engine logs the refusal in
# full. Both are records of the real dispatch path, so read each for what it
# is authoritative about: the boundary for the dispatch STATUS, the engine for
# the refusal TEXT.
_ERROR_LOGGERS = (_BOUNDARY_ERROR_LOGGER, _ENGINE_ERROR_LOGGER)


def _errors(caplog: pytest.LogCaptureFixture, logger: str) -> list[str]:
    """One logger's ERROR records for this dispatch."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == logger and record.levelno >= logging.ERROR
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_absent_evidence_never_completes_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC2/AC3: an empty evidence read refuses instead of reporting success."""
    with caplog.at_level(logging.ERROR):
        handler = await _dispatch_with_evidence((), uuid4())

    handler._read_observed.assert_awaited()  # type: ignore[attr-defined]
    handler._persist_rows.assert_not_awaited()  # type: ignore[attr-defined]

    boundary = _errors(caplog, _BOUNDARY_ERROR_LOGGER)
    assert boundary, "the dispatch completed with no boundary error at all"
    assert any("handler_error" in message for message in boundary), (
        f"the boundary did not record a handler failure: {boundary!r}"
    )

    refusals = [
        message
        for message in _errors(caplog, _ENGINE_ERROR_LOGGER)
        if "no delegation-chain evidence to persist" in message
    ]
    assert refusals, "the engine recorded no typed refusal naming absent evidence"
    for topic in _CHAIN:
        assert any(topic in message for message in refusals), (
            f"the refusal does not name {topic!r}, so it does not say what "
            "evidence was looked for"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_present_evidence_persists_and_takes_no_error_route(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Positive control: the same wiring and handler complete on real evidence.

    Without this the assertion above could pass because the dispatch failed for
    some unrelated reason -- a wiring mistake in the harness would fail BOTH
    tests, not just one.
    """
    correlation_id = uuid4()
    with caplog.at_level(logging.ERROR):
        handler = await _dispatch_with_evidence(
            _observed(correlation_id), correlation_id
        )

    handler._read_observed.assert_awaited()  # type: ignore[attr-defined]
    persist_rows = handler._persist_rows  # type: ignore[attr-defined]
    persist_rows.assert_awaited_once()
    assert len(persist_rows.await_args.args[0]) == len(_CHAIN)
    for logger in _ERROR_LOGGERS:
        assert not _errors(caplog, logger), (
            f"{logger} recorded an error on the success path: "
            f"{_errors(caplog, logger)!r}"
        )
