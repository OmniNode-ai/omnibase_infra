# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime's OWN manifest event must reach the reducer that subscribes to it (OMN-17296).

THE DEFECT
----------
``onex.evt.omnibase-infra.runtime-manifest-published.v1`` was the dev lane's dominant
DLQ source: 189 ``failure_class=no_dispatcher`` routings per runtime start on
``omninode-runtime`` (consumer group
``local.omnibase_infra.node_runtime_manifest_reducer.consume...``), 189 more on
``omninode-runtime-effects`` (group ``local.omnimarket.node_redeploy_orchestrator...``),
and 189 on the stability lane — a bounded replay of the retained manifest history that
recurs on EVERY restart.

The mechanism is a publisher/consumer **event_type alias mismatch**, not the
``operation_match`` routing strategy:

* ``MessageDispatchEngine.dispatch`` sets ``message_type = envelope.event_type`` verbatim
  and then skips any dispatcher whose ``message_types`` set does not contain it.
* ``derive_entry_message_types`` indexes a subscriber under the LITERAL subscribe topic
  and the topic-derived ``<producer>.<event-name>`` alias — here
  ``omnibase-infra.runtime-manifest-published``.
* ``publish_runtime_manifest`` hard-coded ``event_type="runtime-manifest-published"`` —
  the bare event name, with the producer segment missing. It matched neither key, so
  ``node_runtime_manifest_reducer`` consumed, DLQ'd and COMMITTED 100% of its traffic
  while its consumer group read Stable / LAG 0.

The same mismatch also mis-derives the DLQ topic: ``derive_dlq_topic_for_event_type``
reads the first dot-segment of ``event_type`` as the producer domain, so the bare alias
produced the nonsense ``onex.dlq.omnibase-infra.runtime-manifest-published.v1`` observed
live, rather than the ``onex.dlq.omnibase-infra.omnibase-infra.v1`` a correctly-aliased
envelope would take.

Consequence: OMN-15512 folded ``ModelRuntimeAttachReadiness`` into this exact event so
the boot NOT-READY blocker set would be durably queryable. Every instance DLQ'ing means
``runtime_manifests`` never gains a row and that outcome is not delivered.

WHY THESE TESTS ARE NOT SURROGATES
----------------------------------
The envelope under test is produced by the REAL ``publish_runtime_manifest`` — the exact
function ``service_kernel`` step 9.8 calls — captured off a recording bus. The dispatch
index it is matched against is built from the REAL ``node_runtime_manifest_reducer``
contract through the REAL production derivations (``discover_contracts_from_paths``,
``_topics_for_handler_entry``, ``derive_entry_message_category``,
``derive_entry_message_types``, ``_derive_topic_pattern_from_topic``,
``_derive_dispatcher_id``, ``_derive_route_id``) and dispatched through the REAL
``MessageDispatchEngine.dispatch_scoped`` — the same scoped call the auto-wired
subscription makes. Re-deriving any of those in the test is exactly how this class
survived earlier gates, so nothing here re-implements the runtime.

The only double is the handler body: the defect is entirely in the routing index, and a
recording callback lets the test assert the dispatcher was actually ENTERED rather than
merely "not NO_DISPATCHER" — which is the inverse failure this fix must not cause.

``TestGuardActuallyGuards`` re-runs the identical chain with the pre-fix bare alias and
asserts it reproduces the live symptom (NO_DISPATCHER + the nonsense DLQ topic), so a
green result here discriminates against the real defect rather than against a fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.topic_constants import derive_dlq_topic_for_event_type
from omnibase_infra.runtime._enum_coercion import coerce_message_category
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _derive_dispatcher_id,
    _derive_handler_entry_key,
    _derive_route_id,
    _derive_topic_pattern_from_topic,
    _topics_for_handler_entry,
    derive_entry_message_category,
    derive_entry_message_types,
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
from omnibase_infra.runtime.manifest_builder import publish_runtime_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

MANIFEST_TOPIC = "onex.evt.omnibase-infra.runtime-manifest-published.v1"
REDUCER_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_runtime_manifest_reducer"
    / "contract.yaml"
)

# The alias the topic-derived dispatcher index registers, and the bare pre-fix literal.
CANONICAL_ALIAS = "omnibase-infra.runtime-manifest-published"
PREFIX_MISSING_ALIAS = "runtime-manifest-published"

# The DLQ topic the live dev lane logged for every dropped manifest event.
LIVE_DLQ_TOPIC = "onex.dlq.omnibase-infra.runtime-manifest-published.v1"


class _RecordingBus:
    """Event bus double that captures every published envelope."""

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: object,
        on_message: object,
    ) -> object:
        async def _unsub() -> None:
            return None

        return _unsub

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
        *,
        key: bytes | None = None,
    ) -> None:
        self.published.append((topic, envelope))


def _throwaway_contract() -> ModelDiscoveredContract:
    """One minimal contract so ``wire_from_manifest`` yields a real wiring report."""
    return ModelDiscoveredContract(
        name="node_manifest_dispatch_probe",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_manifest_dispatch_probe",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.omnibase-infra.probe-alpha.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="FakeHandler", module="fake.module"),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


def _fake_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


async def _published_manifest_envelope() -> ModelEventEnvelope[object]:
    """Drive the REAL kernel publish seam and return the envelope it put on the bus."""
    manifest = ModelAutoWiringManifest(contracts=(_throwaway_contract(),))
    engine = MessageDispatchEngine()
    bus = _RecordingBus()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_fake_handler_cls(),
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )

    await publish_runtime_manifest(
        event_bus=bus,
        report=report,
        manifest=manifest,
        runtime_profile="dispatch-resolution-test",
        topic=MANIFEST_TOPIC,
        correlation_id=uuid4(),
        image_digest=None,
        attach_readiness=None,
    )

    published = [env for topic, env in bus.published if topic == MANIFEST_TOPIC]
    assert len(published) == 1, (
        f"expected exactly one publish to {MANIFEST_TOPIC}, got {len(published)}"
    )
    envelope = published[0]
    assert isinstance(envelope, ModelEventEnvelope)
    return envelope


def _reducer_contract() -> ModelDiscoveredContract:
    """The REAL node_runtime_manifest_reducer contract, through the REAL discovery path."""
    discovered = discover_contracts_from_paths([REDUCER_CONTRACT_PATH])
    contracts = list(getattr(discovered, "contracts", discovered))
    assert len(contracts) == 1, f"expected 1 discovered contract, got {len(contracts)}"
    return contracts[0]


def _engine_wired_from_reducer_contract() -> tuple[
    MessageDispatchEngine, str, list[Any]
]:
    """Register the reducer's dispatcher exactly as ``_prepare_handler_wiring`` does.

    Returns the frozen engine, the dispatcher id the subscription scopes to, and the
    list the dispatcher appends each envelope it is ENTERED with.
    """
    contract = _reducer_contract()
    assert contract.handler_routing is not None
    entries = list(contract.handler_routing.handlers)
    assert len(entries) == 1, f"expected 1 handler entry, got {len(entries)}"
    entry = entries[0]

    topics = _topics_for_handler_entry(contract, entry)
    assert MANIFEST_TOPIC in topics, (
        f"{contract.name} must own {MANIFEST_TOPIC}; _topics_for_handler_entry gave {topics}"
    )

    category = coerce_message_category(derive_entry_message_category(contract, entry))
    message_types = derive_entry_message_types(contract, entry)
    handler_key = _derive_handler_entry_key(entry)
    dispatcher_id = _derive_dispatcher_id(contract.name, handler_key)

    # The dispatch engine ALWAYS hands a dispatcher the materialized envelope dict, so
    # the recorder takes ``object`` exactly as the auto-wired callback does.
    entered: list[Any] = []

    async def _dispatcher(envelope: Any) -> None:
        entered.append(envelope)

    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=dispatcher_id,
        dispatcher=_dispatcher,
        category=category,
        message_types=message_types,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id=_derive_route_id(contract.name, handler_key, MANIFEST_TOPIC),
            topic_pattern=_derive_topic_pattern_from_topic(MANIFEST_TOPIC),
            message_category=category,
            handler_id=dispatcher_id,
        )
    )
    engine.freeze()
    return engine, dispatcher_id, entered


class TestManifestEventReachesItsSubscriber:
    """The headline: the published manifest event must dispatch, not DLQ."""

    @pytest.mark.asyncio
    async def test_published_manifest_dispatches_to_the_reducer(self) -> None:
        envelope = await _published_manifest_envelope()
        engine, dispatcher_id, entered = _engine_wired_from_reducer_contract()

        result = await engine.dispatch_scoped(
            topic=MANIFEST_TOPIC,
            envelope=envelope,
            allowed_dispatcher_ids={dispatcher_id},
        )

        assert result.status is not EnumDispatchStatus.NO_DISPATCHER, (
            "OMN-17296: the runtime's own runtime-manifest-published envelope did not "
            f"resolve to node_runtime_manifest_reducer. event_type="
            f"{envelope.event_type!r} is not a key the topic-derived dispatcher index "
            "registers, so the runtime consumes, DLQ-routes and COMMITS every manifest "
            "event at LAG 0."
        )
        assert result.status is EnumDispatchStatus.SUCCESS, result.error_message
        assert len(entered) == 1, (
            "the reducer's dispatcher must actually be ENTERED — a fix that only "
            "silences the DLQ without delivering the event is the inverse failure"
        )
        delivered = entered[0]
        assert isinstance(delivered, dict)
        # The reducer must receive THE MANIFEST, not merely be reached: the OMN-15512
        # attach-readiness payload rides this exact envelope into runtime_manifests.
        assert delivered["payload"]["runtime_profile"] == "dispatch-resolution-test"

    @pytest.mark.asyncio
    async def test_published_event_type_is_the_topic_derived_alias(self) -> None:
        """The published alias must be a key the subscriber's index actually holds."""
        envelope = await _published_manifest_envelope()
        contract = _reducer_contract()
        assert contract.handler_routing is not None
        entry = contract.handler_routing.handlers[0]
        message_types = derive_entry_message_types(contract, entry) or set()

        assert envelope.event_type == CANONICAL_ALIAS, (
            "publish_runtime_manifest must stamp the topic-derived "
            f"<producer>.<event-name> alias, got {envelope.event_type!r}"
        )
        assert envelope.event_type in message_types, (
            f"event_type {envelope.event_type!r} must be one of the dispatcher index "
            f"keys {sorted(message_types)!r} — MessageDispatchEngine.dispatch matches "
            "envelope.event_type against that set verbatim"
        )

    @pytest.mark.asyncio
    async def test_dlq_derivation_no_longer_invents_a_producer_domain(self) -> None:
        """A correctly-aliased envelope derives a real DLQ domain, not the event name."""
        envelope = await _published_manifest_envelope()
        derived = derive_dlq_topic_for_event_type(
            event_type=envelope.event_type,
            original_topic=MANIFEST_TOPIC,
        )
        assert derived != LIVE_DLQ_TOPIC, (
            "the bare alias made the DLQ derivation read the EVENT NAME as the producer "
            f"domain, producing {LIVE_DLQ_TOPIC}"
        )


class TestGuardActuallyGuards:
    """Re-run the identical chain with the pre-fix alias; it must reproduce the live drop."""

    @pytest.mark.asyncio
    async def test_bare_alias_reproduces_the_live_no_dispatcher_drop(self) -> None:
        envelope = await _published_manifest_envelope()
        prefix_missing: ModelEventEnvelope[object] = envelope.model_copy(
            update={"event_type": PREFIX_MISSING_ALIAS}
        )
        engine, dispatcher_id, entered = _engine_wired_from_reducer_contract()

        result = await engine.dispatch_scoped(
            topic=MANIFEST_TOPIC,
            envelope=prefix_missing,
            allowed_dispatcher_ids={dispatcher_id},
        )

        assert result.status is EnumDispatchStatus.NO_DISPATCHER, (
            "the pre-fix bare alias must still be unroutable — otherwise the passing "
            "test above proves nothing about the live defect"
        )
        assert entered == []

    def test_bare_alias_reproduces_the_live_dlq_topic(self) -> None:
        assert (
            derive_dlq_topic_for_event_type(
                event_type=PREFIX_MISSING_ALIAS,
                original_topic=MANIFEST_TOPIC,
            )
            == LIVE_DLQ_TOPIC
        ), "the reproduction must match the DLQ topic the dev lane actually logged"

    def test_bare_alias_is_absent_from_the_real_dispatcher_index(self) -> None:
        contract = _reducer_contract()
        assert contract.handler_routing is not None
        entry = contract.handler_routing.handlers[0]
        message_types = derive_entry_message_types(contract, entry) or set()

        assert PREFIX_MISSING_ALIAS not in message_types
        assert CANONICAL_ALIAS in message_types
        assert MANIFEST_TOPIC in message_types


class TestSubscribersDeclareOnlyWhatTheyDispatch:
    """AC2: every declared subscribe topic of the reducer resolves to its dispatcher."""

    def test_reducer_subscribes_to_exactly_the_manifest_topic(self) -> None:
        contract = _reducer_contract()
        assert contract.event_bus is not None
        assert tuple(contract.event_bus.subscribe_topics) == (MANIFEST_TOPIC,), (
            "a contract that declares a subscription it cannot service is a false "
            "statement in the contract graph — the topic reads as consumed when it is not"
        )
