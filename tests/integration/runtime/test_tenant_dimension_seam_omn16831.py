# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam test for the tenant dimension (OMN-16831).

The unit tests beside this one pin each half of the fix at its own function.
This one proves the two halves MEET, by driving a raw wire record through the
ACTUAL auto-wiring subscription boundary -- ``wire_from_manifest`` ->
``event_bus.subscribe`` -> the wiring callback (which is where the
``tenant_scoped_ingress`` stamp runs) -> ``MessageDispatchEngine`` -> the REAL
``DispatchResultApplier`` -> ``publish_envelope``. Nothing on that path is
stubbed except the handler class and the transport, which is the same shape
`test_seam_dispatch_correlation_omn15474.py` uses for the correlation dimension.

A handler-level test cannot see this defect. The two seams that were broken sit
on either side of the dispatch: the stamp is in the subscribe callback, the
carriage is in the publish applier, and the handler between them never sees an
envelope at all -- the canonical signature is ``handle(request: ModelX) ->
ModelY``. Only a test that spans the whole boundary observes a verified tenant
arriving on the wire and leaving on the wire.

What the seam proves, in one pass:

1. A wire record whose envelope RECORDS a tenant -- the exact shape the gateway
   forwarder synthesizes for inbound customer traffic, where ``tenant_id`` is
   set from the deploy-time-bound gateway identity -- is deserialized with that
   tenant intact, and the envelope the runtime publishes as a consequence
   records the same one. That published field is what
   ``envelope_tenant_identity`` reads in omnimarket's delegation projection
   writer.
2. The negative control, on the same wiring: a wire record whose envelope
   records NO tenant publishes a consequence recording none either. Nothing on
   this path invents or defaults one, which is what OMN-16831 AC2 and OMN-16804
   AC3 require, and what keeps the projection writer's fail-closed refusal
   reachable rather than papered over.

Pre-fix, 1 fails and 2 passes.

The subscribe-side stamp -- the second half of the fix, which puts a
topic-verified slug on the envelope for the ``tenant_scoped_ingress`` path -- is
pinned in ``tests/unit/runtime/test_tenant_dimension_carriage.py`` rather than
here. A ``tenant-<slug>.``-prefixed topic derives no event-type alias, so it
matches no dispatcher in this seam and cannot reach the applier; that is
existing behaviour of ``_derive_event_type_from_topic`` and is not changed by
this PR.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ENV_SINGLE_OWNER_COMMAND_TOPICS,
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
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.asyncio

_VERIFIED_TENANT = "acme"
_CANONICAL_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_OUTPUT_TOPIC = "onex.evt.omnibase-infra.quality-gate-result.v1"


class _OutputEvent(BaseModel):
    """What the handler returns; the tenant is an envelope dimension, not this."""

    entity_id: str = "seam-1"


@dataclass
class _KafkaRecord:
    """Minimal record shape the wiring callback consumes (``.value`` bytes)."""

    value: bytes


@dataclass
class _RecordingEventBus:
    """In-process transport that records what the REAL applier publishes."""

    subscriptions: list[tuple[str, Callable[..., Awaitable[None]]]] = field(
        default_factory=list
    )
    published: list[ModelEventEnvelope[Any]] = field(default_factory=list)

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: Any,
        on_message: Callable[..., Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        self.subscriptions.append((topic, on_message))

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe

    async def publish(self, *args: object, **kwargs: object) -> None:
        return None

    async def publish_envelope(
        self, envelope: ModelEventEnvelope[Any], **kwargs: object
    ) -> None:
        self.published.append(envelope)

    async def deliver(self, topic: str, record: _KafkaRecord) -> None:
        for sub_topic, callback in list(self.subscriptions):
            if sub_topic == topic:
                await callback(record)


def _handler_cls() -> type:
    """A handler that returns one event, so the applier has something to publish.

    It never sees an envelope and never publishes -- that is the canonical
    definition-B shape, and it is exactly why neither seam this test spans can
    be reached from a handler-level test.
    """

    class _SeamHandler:
        async def handle(self, envelope: Any) -> ModelHandlerOutput[None]:
            correlation_id = getattr(envelope, "correlation_id", None) or uuid4()
            input_envelope_id = getattr(envelope, "envelope_id", None) or uuid4()
            return ModelHandlerOutput.for_effect(
                input_envelope_id=input_envelope_id,
                correlation_id=correlation_id,
                handler_id="seam-handler",
                events=(_OutputEvent(),),
            )

    return _SeamHandler


def _contract(topic: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_tenant_seam",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/node_tenant_seam/contract.yaml"),
        entry_point_name="node_tenant_seam",
        package_name="omnibase-infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=(_OUTPUT_TOPIC,),
            # The opt-in under test. The stamp only runs for a contract that
            # declares it, which is what keeps an unprefixed lane unstamped.
            tenant_scoped_ingress=True,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerTenantSeam",
                        module="fake.node_tenant_seam",
                    ),
                    event_model=None,
                    operation=None,
                    event_type="omnimarket.delegate-skill",
                    topic=topic,
                ),
            ),
        ),
    )


async def _drive(tenant_id: str | None) -> list[ModelEventEnvelope[Any]]:
    """Boot the real wiring, deliver one wire record, return what the applier shipped.

    ``tenant_id`` is what the INBOUND envelope records, which is the only thing
    that differs between the positive case and the negative control. Everything
    else -- wiring, contract, handler, applier, transport -- is identical, so a
    difference in what is published can only come from the dimension under test.
    """
    bus = _RecordingEventBus()
    applier = DispatchResultApplier(event_bus=bus, output_topic=_OUTPUT_TOPIC)
    engine = MessageDispatchEngine()

    with (
        patch.dict(os.environ, {ENV_SINGLE_OWNER_COMMAND_TOPICS: "0"}),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            side_effect=lambda ref, *a, **kw: _handler_cls(),
        ),
    ):
        contract = _contract(_CANONICAL_TOPIC)
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(contract,)),
            engine,
            event_bus=bus,
            environment="local",
            result_appliers_by_contract={contract.name: applier},
        )
    if not engine.is_frozen:
        engine.freeze()

    # The wire shape the gateway forwarder produces: a full envelope, with the
    # tenant recorded on the envelope rather than only in the payload.
    inbound: ModelEventEnvelope[dict[str, object]] = ModelEventEnvelope(
        payload={"task_type": "test"},
        correlation_id=uuid4(),
        event_type="omnimarket.delegate-skill",
        tenant_id=tenant_id,
    )
    wire = json.loads(inbound.model_dump_json())
    await bus.deliver(
        _CANONICAL_TOPIC, _KafkaRecord(value=json.dumps(wire).encode("utf-8"))
    )
    return bus.published


async def test_recorded_tenant_crosses_the_whole_seam() -> None:
    """A tenant recorded inbound is recorded on what the runtime publishes.

    This is the statement the delegation projection writer depends on: it reads
    ``_envelope["tenant_id"]`` off the published record, because a writer under
    ``FORCE ROW LEVEL SECURITY`` cannot discover a row's tenant by reading. On
    the live customer path the inbound tenant is put there by the gateway
    forwarder from its deploy-time-bound identity; this seam is everything that
    happens after.
    """
    published = await _drive(_VERIFIED_TENANT)

    assert len(published) == 1, (
        "one delivered record must publish exactly one consequence; "
        f"published {len(published)}"
    )
    envelope = published[0]

    assert envelope.tenant_id == _VERIFIED_TENANT, (
        "the runtime published an envelope recording "
        f"{envelope.tenant_id!r} for an inbound envelope recording "
        f"{_VERIFIED_TENANT!r}. An unattributed event here is what dead-letters "
        "to onex.dlq.omnimarket.projection-delegation-malformed.v1."
    )

    # The correlation dimension is unchanged by this fix. A change that carried
    # the tenant by rebuilding the envelope would break this silently, so it is
    # asserted here rather than assumed.
    assert isinstance(envelope.correlation_id, UUID)

    # The OMN-18116 causal edge rides the same contextvar and was equally absent
    # on this boundary before the fix -- measured on this exact seam:
    # parent_envelope_id None for a consumed envelope that had one. Asserting it
    # here is what keeps the widened binding honest; a later change that
    # narrowed it again would fail on both dimensions rather than silently on
    # the one nobody is watching.
    assert envelope.parent_envelope_id is not None, (
        "the published envelope records no causal parent; the applier could not "
        "see the consumed envelope, which is the same absence that leaves the "
        "tenant unattributed"
    )


async def test_unattributed_inbound_publishes_no_tenant() -> None:
    """The negative control, on identical wiring.

    An inbound envelope recording no tenant publishes a consequence recording
    none -- not a default, not a house tenant, not an invented one. The event
    reaches the projection writer unattributed and is refused, which is the
    designed behaviour and the property this whole seam exists to keep true.
    """
    published = await _drive(None)

    assert len(published) == 1
    assert published[0].tenant_id is None, (
        "an inbound envelope recording no tenant must publish a consequence "
        f"recording none; recorded {published[0].tenant_id!r}"
    )
