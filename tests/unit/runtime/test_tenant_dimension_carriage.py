# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Carriage of the tenant dimension across a runtime-originated hop (OMN-16831).

``ModelEventEnvelope.tenant_id`` is declared in omnibase_core as "Tenant this
event belongs to, recorded at write time". omnimarket's delegation projection
writer reads exactly that field -- ``envelope_tenant_identity`` in
``omnimarket/src/omnimarket/projection/envelope.py`` -- and REFUSES, fail-closed,
any tenant-classified row it cannot attribute, because a writer under ``FORCE
ROW LEVEL SECURITY`` cannot discover a row's tenant by reading.

The gateway forwarder DOES record it: ``service_gateway_forwarder.py`` builds
its synthesized inbound envelope with ``tenant_id=identity.tenant_slug`` from
the deploy-time-bound gateway identity. So on the real customer path the FIRST
envelope of a chain carries a verified tenant.

It then died at the first hop. ``DispatchResultApplier`` is the single site that
builds every envelope this runtime publishes as a consequence of consuming
another (the OMN-18116 causal-edge origination site). It already binds the
consumed envelope and propagates ``correlation_id`` and ``parent_envelope_id``
from it -- and discarded ``tenant_id``. Every event published downstream of the
gateway was therefore unattributed, which is why 145 ``quality-gate-result`` and
``delegation-judge-verdict`` events dead-lettered to
``onex.dlq.omnimarket.projection-delegation-malformed.v1`` rather than
projecting.

These tests pin both directions, because the negative controls are the half that
matters here. OMN-16831 AC2 and OMN-16804 AC3 both forbid inventing or
defaulting a tenant inside the runtime: a propagation that also SOURCED an
identity when none was consumed would satisfy a one-sided test while breaking
the property the projection writer's refusal exists to protect.

1. **Carriage.** A tenant recorded on the consumed envelope is recorded on every
   envelope the dispatch publishes because of it.
2. **No origination.** Nothing consumed means no tenant -- the hop is a chain
   head and says so, rather than acquiring one.
3. **No defaulting.** A consumed envelope that recorded no tenant yields
   outputs that record no tenant. An unattributed event stays unattributed and
   reaches the consumer's fail-closed refusal, which is the designed behaviour.
4. **The verified stamp writes the field its only reader reads.** The
   ``tenant_scoped_ingress`` wiring stamp derives a slug from the
   ``tenant-<slug>.`` wire prefix and overwrites any client-supplied value. It
   wrote that verified slug into the PAYLOAD only, while the sole reader of a
   tenant in the fleet reads the ENVELOPE -- a producer and a consumer split
   across two fields, with the consumer's own docstring asserting they were one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _stamp_tenant_id_from_topic_prefix,
)
from omnibase_infra.runtime.dispatch_envelope_context import bind_dispatch_envelope
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit

_TENANT = "acme"


class _OutputEvent(BaseModel):
    """Stub output event; the tenant is an envelope dimension, not a payload one."""

    entity_id: str = "n-1"


class _SecondOutputEvent(BaseModel):
    """A second stub, so multi-publish carriage is pinned rather than assumed."""

    entity_id: str = "n-2"


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


def _published_envelopes(
    mock_bus: AsyncMock,
) -> list[ModelEventEnvelope[BaseModel]]:
    """Return every envelope handed to the bus, in publish order."""
    envelopes: list[ModelEventEnvelope[BaseModel]] = []
    for call in mock_bus.publish_envelope.call_args_list:
        envelope = call.kwargs["envelope"]
        assert isinstance(envelope, ModelEventEnvelope)
        envelopes.append(envelope)
    return envelopes


def _applier(mock_bus: AsyncMock) -> DispatchResultApplier:
    return DispatchResultApplier(event_bus=mock_bus, output_topic="fallback-topic")


class TestApplierCarriesTheTenantDimension:
    """The one origination site carries what it consumed."""

    @pytest.mark.asyncio
    async def test_consumed_tenant_is_recorded_on_the_published_envelope(
        self,
    ) -> None:
        """A hop published because a tenant-attributed one was consumed keeps it."""
        consumed = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(),
            correlation_id=uuid4(),
            tenant_id=_TENANT,
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        with bind_dispatch_envelope(consumed):
            await _applier(mock_bus).apply(_make_result(output_events=[_OutputEvent()]))

        published = _published_envelopes(mock_bus)
        assert len(published) == 1
        assert published[0].tenant_id == _TENANT

    @pytest.mark.asyncio
    async def test_every_output_of_one_dispatch_carries_the_tenant(self) -> None:
        """The quality-gate reducer returns TWO events in one ModelHandlerOutput.

        ``quality-gate-result`` and ``delegation-judge-verdict`` are published
        from a single dispatch, and both were refused by the projection writer.
        Carrying the tenant onto only the first would fix one of the two
        dead-letter classes and leave the other, so the loop is pinned, not the
        single event.
        """
        consumed = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(),
            correlation_id=uuid4(),
            tenant_id=_TENANT,
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        with bind_dispatch_envelope(consumed):
            await _applier(mock_bus).apply(
                _make_result(
                    output_events=[_OutputEvent(), _SecondOutputEvent()],
                )
            )

        published = _published_envelopes(mock_bus)
        assert len(published) == 2
        assert [envelope.tenant_id for envelope in published] == [_TENANT, _TENANT]

    @pytest.mark.asyncio
    async def test_carriage_does_not_disturb_the_causal_edge(self) -> None:
        """Both dimensions come off the same consumed envelope; neither displaces
        the other. A change that carried the tenant by rebuilding the envelope
        without the OMN-18116 edge would pass every tenant assertion above."""
        consumed = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(),
            correlation_id=uuid4(),
            tenant_id=_TENANT,
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        with bind_dispatch_envelope(consumed):
            await _applier(mock_bus).apply(_make_result(output_events=[_OutputEvent()]))

        published = _published_envelopes(mock_bus)[0]
        assert published.tenant_id == _TENANT
        assert published.parent_envelope_id == consumed.envelope_id


class TestApplierNeverSourcesATenant:
    """The negative controls. OMN-16831 AC2 / OMN-16804 AC3: never invented,
    never defaulted."""

    @pytest.mark.asyncio
    async def test_nothing_consumed_records_no_tenant(self) -> None:
        """A chain head acquires no identity on the way out."""
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        await _applier(mock_bus).apply(_make_result(output_events=[_OutputEvent()]))

        assert _published_envelopes(mock_bus)[0].tenant_id is None

    @pytest.mark.asyncio
    async def test_unattributed_consumed_envelope_stays_unattributed(self) -> None:
        """The failure mode this whole seam exists to keep reachable.

        An event whose producer recorded no tenant must arrive at the projection
        writer still recording none, so the writer's fail-closed refusal fires.
        Defaulting here would stamp a house tenant onto a row the submitting
        tenant's reader could never see -- the exact outcome the refusal at
        ``handler_delegation.py`` was written to prevent.
        """
        consumed = ModelEventEnvelope[_ConsumedPayload](
            payload=_ConsumedPayload(),
            correlation_id=uuid4(),
        )
        assert consumed.tenant_id is None
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        with bind_dispatch_envelope(consumed):
            await _applier(mock_bus).apply(_make_result(output_events=[_OutputEvent()]))

        assert _published_envelopes(mock_bus)[0].tenant_id is None


class TestVerifiedStampWritesTheEnvelopeDimension:
    """The ``tenant_scoped_ingress`` stamp and its only reader must name one
    field."""

    def test_verified_slug_reaches_the_envelope_not_only_the_payload(self) -> None:
        """The stamp's slug is verified -- derived from the wire topic prefix and
        overwriting any client-supplied value. It belongs on the dimension the
        projection writer reads, not only in the payload."""
        envelope = ModelEventEnvelope[dict[str, object]](
            payload={"value": "in", "tenant_id": "attacker-supplied"},
            correlation_id=uuid4(),
        )

        stamped = _stamp_tenant_id_from_topic_prefix(
            f"tenant-{_TENANT}.onex.cmd.omnimarket.delegate-skill.v1", envelope
        )

        assert stamped.tenant_id == _TENANT
        assert stamped.payload["tenant_id"] == _TENANT

    def test_an_unprefixed_topic_stamps_nothing(self) -> None:
        """The negative control the stamp's own docstring states: a topic with no
        ``tenant-<slug>.`` prefix is left completely unstamped."""
        envelope = ModelEventEnvelope[dict[str, object]](
            payload={"value": "in"},
            correlation_id=uuid4(),
        )

        stamped = _stamp_tenant_id_from_topic_prefix(
            "onex.cmd.omnimarket.delegate-skill.v1", envelope
        )

        assert stamped.tenant_id is None
        assert "tenant_id" not in stamped.payload
