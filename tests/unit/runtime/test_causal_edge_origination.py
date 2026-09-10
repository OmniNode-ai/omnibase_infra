# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Origination and carriage of the causal edge (OMN-18116).

A chain replay re-derives what caused what and compares it to a record. Nothing
wrote that record. These tests pin the four places the edge has to survive for a
verifier to read it back:

1. **Origination.** The runtime dispatch-result applier is the one site that
   builds every outgoing envelope, and the consumed envelope is already bound to
   the dispatch on a contextvar. The edge is set there, from that, and nowhere
   else. No handler participates -- the canonical signature is
   ``handle(request: ModelX) -> ModelY`` and never sees an envelope.
2. **The wire header.** ``event_ledger`` populates its per-hop identity column
   from the ``message_id`` HEADER, not from the envelope body. Measured
   read-only on the dev lane over a two-hour window before this change: of 737
   rows whose body carried an ``envelope_id``, the column matched the body on
   **0** and differed on **737**. So the header identity is bound to the
   envelope's, and the edge rides beside it.
3. **Encode and decode.** The edge is absent-able. It is parsed WITHOUT the
   mint-on-absence fallback the identity headers use, because inventing a
   parent for a chain head would turn a broken chain into one that re-derives
   cleanly.
4. **The ledger readback.** The edge reaches `event_ledger` with no schema
   change, because the projection stores the whole header model in the
   `onex_headers` JSONB. A verifier reads it at
   `onex_headers ->> 'parent_message_id'`.

The negative controls matter as much as the positive ones here: a validator that
refused every edge, or an extractor that invented one, would satisfy a
one-sided test while breaking the feature.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit


class _OutputEvent(BaseModel):
    """Stub output event; the edge is an envelope property, not a payload one."""

    entity_id: str = "n-1"


class _ConsumedPayload(BaseModel):
    """Stub payload for the envelope a dispatch consumed."""

    value: str = "in"


def _make_result(**overrides: object) -> ModelDispatchResult:
    defaults: dict[str, object] = {
        "status": EnumDispatchStatus.SUCCESS,
        "topic": "test.topic",
        "started_at": datetime.now(UTC),
        "correlation_id": uuid4(),
        "dispatcher_id": "test-dispatcher",
    }
    defaults.update(overrides)
    return ModelDispatchResult(**defaults)


def _published_envelope(mock_bus: AsyncMock) -> ModelEventEnvelope[BaseModel]:
    """Return the single envelope handed to the bus."""
    mock_bus.publish_envelope.assert_called_once()
    envelope = mock_bus.publish_envelope.call_args.kwargs["envelope"]
    assert isinstance(envelope, ModelEventEnvelope)
    return envelope


class TestApplierOriginatesTheEdge:
    """The one origination site."""

    @pytest.mark.asyncio
    async def test_consumed_envelope_becomes_the_parent(self) -> None:
        """A hop published because another was consumed records that one."""
        consumed = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(),
            correlation_id=uuid4(),
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
        )
        with bind_dispatch_envelope(consumed):
            await applier.apply(_make_result(output_events=[_OutputEvent()]))

        assert _published_envelope(mock_bus).parent_envelope_id == consumed.envelope_id

    @pytest.mark.asyncio
    async def test_no_consumed_envelope_is_a_chain_head(self) -> None:
        """The negative control: nothing consumed, so no edge is invented."""
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
        )
        await applier.apply(_make_result(output_events=[_OutputEvent()]))

        assert _published_envelope(mock_bus).parent_envelope_id is None

    @pytest.mark.asyncio
    async def test_every_event_of_one_dispatch_carries_the_same_parent(self) -> None:
        """A dispatch emitting several events records one cause for all of them.

        Each output gets its own deterministic identity but shares the parent,
        because they were all caused by the same consumed event.
        """
        consumed = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(),
            correlation_id=uuid4(),
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        applier = DispatchResultApplier(
            event_bus=mock_bus,
            output_topic="fallback-topic",
        )
        with bind_dispatch_envelope(consumed):
            await applier.apply(
                _make_result(
                    output_events=[
                        _OutputEvent(entity_id="a"),
                        _OutputEvent(entity_id="b"),
                    ]
                )
            )

        envelopes = [
            call.kwargs["envelope"] for call in mock_bus.publish_envelope.call_args_list
        ]
        assert len(envelopes) == 2
        assert {e.parent_envelope_id for e in envelopes} == {consumed.envelope_id}
        assert len({e.envelope_id for e in envelopes}) == 2


class TestWireHeaderCarriesIdentityAndEdge:
    """The header identity is the envelope identity, and the edge rides with it."""

    def test_headers_declare_an_optional_parent(self) -> None:
        """Absent by default, which is the chain-head statement."""
        headers = ModelEventHeaders(
            source="s", event_type="e", timestamp=datetime.now(UTC)
        )
        assert headers.parent_message_id is None

    def test_edge_round_trips_through_kafka_headers(self) -> None:
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        bus = EventBusKafka.__new__(EventBusKafka)
        parent = uuid4()
        headers = ModelEventHeaders(
            source="s",
            event_type="e",
            timestamp=datetime.now(UTC),
            parent_message_id=parent,
        )
        wire = bus._model_headers_to_kafka(headers)
        assert ("parent_message_id", str(parent).encode("utf-8")) in wire

        decoded = bus._kafka_headers_to_model(wire)
        assert decoded.parent_message_id == parent

    def test_a_chain_head_emits_no_edge_header_at_all(self) -> None:
        """Absent and blank must not read alike on the wire."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        bus = EventBusKafka.__new__(EventBusKafka)
        headers = ModelEventHeaders(
            source="s", event_type="e", timestamp=datetime.now(UTC)
        )
        wire = bus._model_headers_to_kafka(headers)
        assert not [k for k, _ in wire if k == "parent_message_id"]

        assert bus._kafka_headers_to_model(wire).parent_message_id is None

    def test_a_malformed_edge_header_records_no_edge(self) -> None:
        """It must NOT mint one.

        ``_parse_uuid_header`` invents a UUID on a bad value, which is right for
        an identity every message must have. For an edge it would fabricate a
        parent nothing produced, and the verifier would then be comparing
        against a number with no origin.
        """
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        bus = EventBusKafka.__new__(EventBusKafka)
        decoded = bus._kafka_headers_to_model(
            [
                ("source", b"s"),
                ("event_type", b"e"),
                ("parent_message_id", b"not-a-uuid"),
            ]
        )
        assert decoded.parent_message_id is None
        # Negative control: the identity header on the same message DID get a
        # value, so the assertion above is about the edge and not about the
        # decoder failing wholesale.
        assert isinstance(decoded.message_id, UUID)

    def test_a_non_envelope_payload_contributes_no_identity(self) -> None:
        """The binding helper must not disturb non-envelope publishes.

        Returning an explicit ``None`` for a missing field would replace the
        header model's own minting with a value it refuses, breaking every
        publish that does not hand over an envelope.
        """
        from omnibase_infra.event_bus.envelope_header_identity import (
            header_identity_fields_from_envelope,
        )

        assert header_identity_fields_from_envelope({"not": "an envelope"}) == {}

    def test_helper_omits_the_edge_for_a_head(self) -> None:
        from omnibase_infra.event_bus.envelope_header_identity import (
            header_identity_fields_from_envelope,
        )

        envelope = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(), correlation_id=uuid4()
        )
        fields = header_identity_fields_from_envelope(envelope)
        assert fields["message_id"] == envelope.envelope_id
        assert "parent_message_id" not in fields

    @pytest.mark.asyncio
    async def test_publish_envelope_binds_identity_and_edge_from_the_envelope(
        self,
    ) -> None:
        """The measured defect: the header used to mint its own identity."""
        from omnibase_infra.event_bus.mixin_kafka_broadcast import MixinKafkaBroadcast

        published: dict[str, object] = {}

        class _Host(MixinKafkaBroadcast):
            _environment = "test"

            async def publish(
                self,
                topic: str,
                key: bytes | None,
                value: bytes,
                headers: ModelEventHeaders,
            ) -> None:
                published["headers"] = headers

        parent = uuid4()
        correlation = uuid4()
        envelope = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(),
            correlation_id=correlation,
            parent_envelope_id=parent,
        )
        await _Host().publish_envelope(envelope=envelope, topic="t")

        headers = published["headers"]
        assert isinstance(headers, ModelEventHeaders)
        assert headers.message_id == envelope.envelope_id
        assert headers.correlation_id == correlation
        assert headers.parent_message_id == parent

    @pytest.mark.asyncio
    async def test_publish_envelope_of_a_head_sets_no_edge(self) -> None:
        """Negative control for the binding above."""
        from omnibase_infra.event_bus.mixin_kafka_broadcast import MixinKafkaBroadcast

        published: dict[str, object] = {}

        class _Host(MixinKafkaBroadcast):
            _environment = "test"

            async def publish(
                self,
                topic: str,
                key: bytes | None,
                value: bytes,
                headers: ModelEventHeaders,
            ) -> None:
                published["headers"] = headers

        envelope = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(), correlation_id=uuid4()
        )
        await _Host().publish_envelope(envelope=envelope, topic="t")

        headers = published["headers"]
        assert isinstance(headers, ModelEventHeaders)
        assert headers.parent_message_id is None
        assert headers.message_id == envelope.envelope_id


class TestLedgerReadsTheEdgeWithoutASchemaChange:
    """The edge reaches the ledger today, in the headers the projection stores.

    ``HandlerLedgerProjection._normalize_headers`` dumps the whole header model
    into the ``onex_headers`` JSONB column, so a declared header field is
    recorded without a migration. A verifier reads the edge at
    ``onex_headers ->> 'parent_message_id'``.

    A dedicated ``event_ledger.parent_envelope_id`` column is a convenience for
    that verifier, not a precondition, and it is deliberately NOT in this
    change: the forward stream's ordinal pin is currently being re-scoped by
    the ledger_chain work, and adding a second migration against the same pin
    would collide with it for no gain this change needs.
    """

    def _normalized(self, headers: ModelEventHeaders) -> dict[str, object]:
        from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
            HandlerLedgerProjection,
        )

        handler = HandlerLedgerProjection.__new__(HandlerLedgerProjection)
        return handler._normalize_headers(headers)

    def test_edge_is_recorded_in_the_stored_headers(self) -> None:
        parent = uuid4()
        stored = self._normalized(
            ModelEventHeaders(
                source="s",
                event_type="e",
                timestamp=datetime.now(UTC),
                parent_message_id=parent,
            )
        )
        assert stored["parent_message_id"] == str(parent)

    def test_a_head_records_an_explicit_absence(self) -> None:
        stored = self._normalized(
            ModelEventHeaders(source="s", event_type="e", timestamp=datetime.now(UTC))
        )
        # Present-and-null, not missing: a reader can tell "recorded as a head"
        # from "written by a producer that predates the edge".
        assert "parent_message_id" in stored
        assert stored["parent_message_id"] is None
        # Negative control: identity is recorded on the same message, so the
        # None above is the edge being absent and not a serialization failure.
        assert stored["message_id"] is not None
