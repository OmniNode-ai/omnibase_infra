# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16798 — the auto-wired consume boundary must never commit with no effect.

TWO DEFECTS, ONE PRINCIPLE. OMN-16767 established that ``db_io`` declares GOVERNED
DB ACCESS and says nothing about a handler's dispatch SHAPE; #2937 removed that
conflation from ARM SELECTION. It survived one hop later, in RESULT DELIVERY, and
it survived at the boundary that owns the consume offset.

DEFECT B — the dispatch that ran never published its decision. Both places that
build a ``DispatchResultApplier`` for an auto-wired subscription skipped any
contract declaring ``db_io.db_tables``:

* ``service_kernel.py`` — the OMN-12409 manifest scan, whose own comment says
  "contracts that return a model destined for a declared topic would have their
  handler output silently dropped without an applier", and which then skipped
  exactly such a contract.
* ``handler_wiring._subscribe_contract_topics`` — the fallback applier.

So on the .201 dev lane at 2026-08-27T21:51:32Z, ``HandlerRoutingIntent`` resolved
(``model=gemini-2.5-flash tier=cheap_cloud``), returned its ``ModelRoutingDecision``
— and ``onex.evt.omnibase-infra.routing-decision.v1`` stayed at a FLAT high-water
mark of 59. The delegation orchestrator subscribes to that topic, so it waited out
its ingress budget and emitted ``status=timeout``.

DEFECT A — silent consume+commit as a reachable outcome. The boundary had three
paths that consumed a record, committed the offset, and produced nothing
observable: a result with output but no applier (the shape above), and a
``NO_DISPATCHER`` result whose engine-derived ``dlq_topic`` nothing ever published
to. The sibling boundary (``EventBusSubcontractWiring``) has honored that field
since OMN-14936; the auto-wired one did not. That is the "consumes+commits but
never dispatches" signature first filed as OMN-14755 and re-measured live here.

WHY THESE TESTS ARE AT THE REAL SEAM. They drive ``wire_from_manifest`` with a real
``MessageDispatchEngine`` and a real ``EventBusInmemory``, publish the verbatim wire
record off the live topic, and assert on what reaches the declared publish topic —
the same reason the OMN-16767 suite exists. An isolation test that called the
handler directly passed throughout both outages.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback,
    _raise_if_no_dispatcher_drop,
    _undeliverable_dispatch_output,
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

_THIS_MODULE = "tests.unit.runtime.auto_wiring.test_omn16798_boundary_delivery_totality"

# The live topics and correlation from the OMN-16798 reproduction.
_SUBSCRIBE_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"  # onex-topic-allow: verbatim from the live incident trace
_PUBLISH_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"  # onex-topic-allow: verbatim from the live incident trace
_CORRELATION = "7a300827-3000-4000-8000-000000000001"

_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)


# ---------------------------------------------------------------------------
# Field-for-field mirrors of the omnimarket / omnibase_core production shapes
# ---------------------------------------------------------------------------


class ModelMirrorDelegationRequest(BaseModel):
    """Mirror of ``ModelDelegationRequest`` — the inner domain payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    task_type: str
    correlation_id: UUID
    max_tokens: int = 2048


class ModelMirrorRoutingIntent(BaseModel):
    """Mirror of ``omnibase_core.models.delegation.wire.ModelRoutingIntent``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: str = Field(default="routing_reducer")
    payload: ModelMirrorDelegationRequest
    min_tier_name: str | None = None
    excluded_backend_refs: tuple[str, ...] = Field(default_factory=tuple)


class ModelMirrorRoutingDecision(BaseModel):
    """Mirror of the ``ModelRoutingDecision`` the reducer publishes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_model: str
    correlation_id: UUID


class HandlerMirrorRoutingIntent:
    """Mirror of ``HandlerRoutingIntent``: a typed def-B handler on a db_io contract."""

    def handle(self, intent: ModelMirrorRoutingIntent) -> ModelMirrorRoutingDecision:
        return ModelMirrorRoutingDecision(
            selected_model="gemini-2.5-flash",
            correlation_id=intent.payload.correlation_id,
        )


# ---------------------------------------------------------------------------
# Contract construction — the live contract's own db_io + published_events
# ---------------------------------------------------------------------------

# ``published_events`` is what makes a REDUCER's declared-event return an EVENT
# rather than a projection capture (OMN-14794), and it is the map the kernel scan
# and the applier both resolve the destination topic from. Written to a real file
# so the classification under test is the one production performs.
_CONTRACT_YAML = f"""
name: "node_delegation_routing_reducer_mirror"
node_type: "REDUCER_GENERIC"
event_bus:
  subscribe_topics:
    - "{_SUBSCRIBE_TOPIC}"
  publish_topics:
    - "{_PUBLISH_TOPIC}"
published_events:
  - event_type: "MirrorRoutingDecision"
    topic: "{_PUBLISH_TOPIC}"
    description: "Routing decision emitted by the mirrored reducer."
"""


def _tenant_overlay_table() -> ModelDbTableDeclaration:
    """The exact ``db_io.db_tables`` entry that suppressed the applier in production."""
    return ModelDbTableDeclaration(
        name="delegation_routing_tenant_overlay",
        database_ref="application",
        schema="tenant",
        migration="0001_create_delegation_routing_tenant_overlay.sql",
        access="read_write",
        role="tenant_routing_overlay",
    )


def _contract(*, with_db_io: bool, contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_delegation_routing_reducer_mirror",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=3, patch=0),
        contract_path=contract_path,
        entry_point_name="node_delegation_routing_reducer_mirror",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_SUBSCRIBE_TOPIC,),
            publish_topics=(_PUBLISH_TOPIC,),
        ),
        db_io=(
            ModelDbOwnershipSubcontract(db_tables=[_tenant_overlay_table()])
            if with_db_io
            else None
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerMirrorRoutingIntent", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorRoutingIntent", module=_THIS_MODULE
                    ),
                    operation="delegation_routing",
                ),
            ),
        ),
    )


def _routing_intent_wire() -> dict[str, object]:
    """The ModelRoutingIntent as published, verbatim from topic offset 84."""
    return {
        "intent": "routing_reducer",
        "payload": {
            "prompt": "Reply with the single word: alive.",
            "task_type": "test",
            "correlation_id": _CORRELATION,
            "max_tokens": 32,
        },
        "min_tier_name": None,
        "excluded_backend_refs": [],
    }


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT_YAML, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# DEFECT B — RED: the returned decision never reached its declared topic
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_db_io_contract_publishes_its_declared_decision(
    contract_path: Path,
) -> None:
    """RED before the fix: db_io suppressed the applier, so the decision vanished.

    This is the live 21:51:32Z shape end to end — the handler resolves, returns a
    ``ModelRoutingDecision``, and pre-fix ``routing-decision.v1`` stayed flat while
    the orchestrator timed out waiting for it.
    """
    correlation_id = UUID(_CORRELATION)
    contract = _contract(with_db_io=True, contract_path=contract_path)

    bus = EventBusInmemory(environment="test", group="omn-16798-decision")
    await bus.start()
    try:
        published: asyncio.Queue[ModelEventEnvelope[object]] = asyncio.Queue()

        async def collect(message: ModelEventMessage) -> None:
            envelope = ModelEventEnvelope[object].model_validate_json(message.value)
            if envelope.correlation_id == correlation_id:
                await published.put(envelope)

        await bus.subscribe(
            _PUBLISH_TOPIC, group_id="decision-collector", on_message=collect
        )

        engine = MessageDispatchEngine()
        with patch(_PATCH_IMPORT_HANDLER, return_value=HandlerMirrorRoutingIntent):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(contract,)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        command = ModelEventEnvelope[object](
            payload=_routing_intent_wire(),
            correlation_id=correlation_id,
            event_type="omnibase-infra.delegation-routing-request",
        )
        await bus.publish(
            _SUBSCRIBE_TOPIC, None, command.model_dump_json().encode("utf-8"), None
        )

        envelope = await asyncio.wait_for(published.get(), timeout=5)
    finally:
        await bus.close()

    decision = ModelMirrorRoutingDecision.model_validate(envelope.payload)
    assert decision.selected_model == "gemini-2.5-flash"
    # AC2: correlation-exact, one record for one routing request.
    assert decision.correlation_id == correlation_id
    assert envelope.correlation_id == correlation_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kernel_scan_registers_an_applier_for_a_db_io_contract(
    contract_path: Path,
) -> None:
    """The kernel's OMN-12409 manifest scan must no longer skip db_io contracts.

    Mirrors ``service_kernel``'s scan exactly. Pre-fix the ``db_io`` guard removed
    ``node_delegation_routing_reducer`` — the ONE contract on the deployed lane
    declaring BOTH ``db_io.db_tables`` and ``published_events`` — from the registry
    the applier is read out of, which is where the decision was lost.
    """
    from omnibase_infra.protocols import ProtocolEventBusLike
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        load_published_events_map,
    )
    from omnibase_infra.runtime.service_dispatch_result_applier import (
        DispatchResultApplier,
    )

    contract = _contract(with_db_io=True, contract_path=contract_path)
    appliers: dict[str, object] = {}

    for _contract_row in (contract,):
        if _contract_row.name in appliers:
            continue
        if (
            _contract_row.event_bus is None
            or not _contract_row.event_bus.publish_topics
        ):
            continue
        pe_map = load_published_events_map(Path(_contract_row.contract_path))
        if not pe_map:
            continue
        topics = tuple(_contract_row.event_bus.publish_topics)
        appliers[_contract_row.name] = DispatchResultApplier(
            event_bus=AsyncMock(spec=ProtocolEventBusLike),
            output_topic=topics[0],
            output_topic_map=pe_map,
            allowed_output_topics=topics,
        )

    applier = appliers.get("node_delegation_routing_reducer_mirror")
    assert isinstance(applier, DispatchResultApplier), (
        "db_io removed the contract from the applier registry — the returned "
        "ModelRoutingDecision had nowhere to go"
    )
    assert applier.published_events_map == {"MirrorRoutingDecision": _PUBLISH_TOPIC}


# ---------------------------------------------------------------------------
# DEFECT A — RED: outcomes that consumed, committed, and produced nothing
# ---------------------------------------------------------------------------


def _dispatch_result(
    *,
    status: EnumDispatchStatus,
    output_events: list[object] | None = None,
    dlq_topic: str | None = None,
) -> ModelDispatchResult:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return ModelDispatchResult(
        status=status,
        topic=_SUBSCRIBE_TOPIC,
        started_at=now,
        completed_at=now,
        output_count=len(output_events or []),
        output_events=list(output_events or []),
        dlq_topic=dlq_topic,
        correlation_id=UUID(_CORRELATION),
        error_details={"failure_class": "publisher_malformed"}
        if status is EnumDispatchStatus.NO_DISPATCHER
        else {},
    )


class _RecordingDlqBus:
    """Minimal duck-typed bus exposing the boundary's DLQ contract."""

    def __init__(self, *, persisted: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self._persisted = persisted

    async def _publish_raw_to_dlq(self, **kwargs: object) -> bool:
        self.calls.append(kwargs)
        return self._persisted


def _boundary_message() -> ModelEventMessage:
    from datetime import UTC, datetime

    from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders

    envelope = ModelEventEnvelope[object](
        payload=_routing_intent_wire(),
        correlation_id=UUID(_CORRELATION),
        event_type="omnibase-infra.delegation-routing-request",
    )
    return ModelEventMessage(
        topic=_SUBSCRIBE_TOPIC,
        key=None,
        value=envelope.model_dump_json().encode("utf-8"),
        headers=ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="omn-16798-test",
            event_type="omnibase-infra.delegation-routing-request",
            correlation_id=UUID(_CORRELATION),
        ),
    )


@pytest.mark.unit
class TestUndeliverableOutputIsNeverSilent:
    def test_output_events_without_an_applier_are_reported(self) -> None:
        """RED: this exact result shape was returned, then dropped, then ACKed."""
        detail = _undeliverable_dispatch_output(
            _dispatch_result(
                status=EnumDispatchStatus.SUCCESS,
                output_events=[
                    ModelMirrorRoutingDecision(
                        selected_model="gemini-2.5-flash",
                        correlation_id=UUID(_CORRELATION),
                    )
                ],
            )
        )
        assert detail is not None
        assert "ModelMirrorRoutingDecision" in detail
        assert "no result applier is wired" in detail

    def test_a_result_with_no_output_is_not_a_drop(self) -> None:
        """Regression: the projection arm returns nothing — that is not a loss.

        Without this the guard would fire on every projection dispatch on the
        lane and turn a correct wiring into a DLQ storm.
        """
        assert (
            _undeliverable_dispatch_output(
                _dispatch_result(status=EnumDispatchStatus.SUCCESS)
            )
            is None
        )
        assert _undeliverable_dispatch_output(None) is None

    @pytest.mark.asyncio
    async def test_boundary_dlqs_output_it_cannot_deliver(self) -> None:
        """RED at the boundary: no applier + output must DLQ, never silently ACK.

        Drives the REAL ``_make_event_bus_callback`` with ``result_applier=None``
        — the wiring state the deployed lane was in — and asserts the record is
        durably captured instead of vanishing behind a committed offset.
        """
        dlq_bus = _RecordingDlqBus()
        engine = AsyncMock()
        engine.dispatch_scoped.return_value = _dispatch_result(
            status=EnumDispatchStatus.SUCCESS,
            output_events=[
                ModelMirrorRoutingDecision(
                    selected_model="gemini-2.5-flash",
                    correlation_id=UUID(_CORRELATION),
                )
            ],
        )

        callback = _make_event_bus_callback(
            _SUBSCRIBE_TOPIC,
            cast("object", engine),  # type: ignore[arg-type]
            result_applier=None,
            event_bus=dlq_bus,
            allowed_dispatcher_ids=("dispatcher.auto.mirror",),
        )
        await callback(_boundary_message())

        assert len(dlq_bus.calls) == 1, (
            "the boundary consumed a record carrying an undeliverable event and "
            "left no DLQ entry — a committed offset with no observable effect"
        )
        assert dlq_bus.calls[0]["original_topic"] == _SUBSCRIBE_TOPIC

    @pytest.mark.asyncio
    async def test_undeliverable_output_withholds_the_offset_when_the_dlq_fails(
        self,
    ) -> None:
        """A non-durable DLQ write must raise, not ACK over a record that exists nowhere."""
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            BoundaryApplyPublishError,
        )

        dlq_bus = _RecordingDlqBus(persisted=False)
        engine = AsyncMock()
        engine.dispatch_scoped.return_value = _dispatch_result(
            status=EnumDispatchStatus.SUCCESS,
            output_events=[
                ModelMirrorRoutingDecision(
                    selected_model="gemini-2.5-flash",
                    correlation_id=UUID(_CORRELATION),
                )
            ],
        )

        callback = _make_event_bus_callback(
            _SUBSCRIBE_TOPIC,
            cast("object", engine),  # type: ignore[arg-type]
            result_applier=None,
            event_bus=dlq_bus,
            allowed_dispatcher_ids=("dispatcher.auto.mirror",),
        )
        with pytest.raises(BoundaryApplyPublishError):
            await callback(_boundary_message())


@pytest.mark.unit
class TestNoDispatcherIsNeverSilent:
    def test_no_dispatcher_result_raises(self) -> None:
        """RED: the engine NAMES a DLQ topic and never publishes to it.

        This is the OMN-14755 signature — Stable group, lag 0, offsets advancing,
        zero handler invocations, nothing in any DLQ.
        """
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            HandlerDispatchFailureError,
        )

        with pytest.raises(HandlerDispatchFailureError) as excinfo:
            _raise_if_no_dispatcher_drop(
                _dispatch_result(
                    status=EnumDispatchStatus.NO_DISPATCHER,
                    dlq_topic="onex.dlq.omnibase-infra.delegation-routing-request.v1",  # onex-topic-allow: DLQ topic the engine derives for the incident topic
                ),
                _SUBSCRIBE_TOPIC,
            )
        assert "publisher_malformed" in str(excinfo.value)

    def test_a_successful_dispatch_is_untouched(self) -> None:
        """Regression: only NO_DISPATCHER is surfaced here."""
        _raise_if_no_dispatcher_drop(
            _dispatch_result(status=EnumDispatchStatus.SUCCESS), _SUBSCRIBE_TOPIC
        )
        _raise_if_no_dispatcher_drop(None, _SUBSCRIBE_TOPIC)

    @pytest.mark.asyncio
    async def test_boundary_dlqs_a_no_dispatcher_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The auto-wired boundary must reach the DLQ its sibling boundary already does."""
        monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")

        dlq_bus = _RecordingDlqBus()
        engine = AsyncMock()
        engine.dispatch_scoped.return_value = _dispatch_result(
            status=EnumDispatchStatus.NO_DISPATCHER,
            dlq_topic="onex.dlq.omnibase-infra.delegation-routing-request.v1",  # onex-topic-allow: DLQ topic the engine derives for the incident topic
        )

        callback = _make_event_bus_callback(
            _SUBSCRIBE_TOPIC,
            cast("object", engine),  # type: ignore[arg-type]
            result_applier=None,
            event_bus=dlq_bus,
            allowed_dispatcher_ids=("dispatcher.auto.mirror",),
        )
        await callback(_boundary_message())

        assert len(dlq_bus.calls) == 1, (
            "a NO_DISPATCHER record was consumed and committed with no DLQ entry"
        )

    @pytest.mark.asyncio
    async def test_a_record_already_on_a_dlq_topic_is_not_re_dlqd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RED for the amplification the first deployed cut of this fix caused.

        Measured live 2026-08-27T23:49-23:51Z: ``node_dlq_replay_effect`` and
        ``node_ledger_projection_compute`` both consume
        ``onex.dlq.omnibase-infra.commands.v1``; a record neither could route was
        re-DLQ'd onto that SAME topic (``get_dlq_topic_for_original`` resolves a
        dlq topic to itself), and 24 of the last 38 records on it carried
        ``original_topic: onex.dlq.omnibase-infra.commands.v1`` — records the
        guard had authored. A record already on a dead-letter sink is durably
        captured; the loud log is the evidence and the loop must stop there.
        """
        monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")

        dlq_topic = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: verbatim from the live amplification trace
        dlq_bus = _RecordingDlqBus()
        engine = AsyncMock()
        engine.dispatch_scoped.return_value = _dispatch_result(
            status=EnumDispatchStatus.NO_DISPATCHER,
            dlq_topic=dlq_topic,
        )

        callback = _make_event_bus_callback(
            dlq_topic,
            cast("object", engine),  # type: ignore[arg-type]
            result_applier=None,
            event_bus=dlq_bus,
            allowed_dispatcher_ids=("dispatcher.auto.mirror",),
        )
        await callback(_boundary_message())

        assert dlq_bus.calls == [], (
            "an unroutable record consumed FROM a dead-letter topic was "
            "republished onto that same topic — the guard amplifying its own "
            "output instead of recording it"
        )

    def test_dead_letter_source_topics_are_recognized(self) -> None:
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _is_dead_letter_source_topic,
        )

        assert _is_dead_letter_source_topic(
            "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow: verbatim from the live amplification trace
        )
        assert _is_dead_letter_source_topic(
            "onex.dlq.omnibase-infra.quarantine.v1"  # onex-topic-allow: the platform quarantine sink named in OMN-16769
        )
        assert not _is_dead_letter_source_topic(_SUBSCRIBE_TOPIC)
        assert not _is_dead_letter_source_topic(_PUBLISH_TOPIC)
