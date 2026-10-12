# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21056 — one pattern candidate must reach validation-result-published.

On the dev lane a synthetic candidate published to
``validation-candidate-submitted`` stopped after ONE hop: the orchestrator built
a plan and published it to ``validation-plan-created`` (the contract's first
publish topic, the applier's fallback), and nothing happened after that:

* ``execution-requested`` and ``adjudication-requested`` were declared publish
  topics that no handler ever emitted, and the executor and the adjudicator
  declared NO ``subscribe_topics``, so neither had a consumer group;
* ``checks-completed`` and ``adjudication-completed`` were subscribed by the
  orchestrator but routed to ``HandlerBuildPlan`` (a candidate handler) because
  the contract carried one unscoped routing entry for all three topics;
* the executor and the adjudicator declared no ``published_events``, so the
  kernel built no result applier for them and any model they returned was
  dropped;
* ``result-published`` had no publisher at all.

These tests drive the REAL seam: the three real contracts through discovery and
``wire_from_manifest``, a real ``MessageDispatchEngine``, a real
``EventBusInmemory`` and the applier the kernel builds for a contract that
declares ``published_events``. An isolation test calling a handler directly
passes through this entire outage, because the defect is in the wiring.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumCheckSeverity
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.models.validation_ledger import ModelPayloadValidationLedgerAppend
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import (
    build_contract_result_applier,
)

pytestmark = pytest.mark.unit

_NODES_ROOT = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"
_PIPELINE_NODES = (
    "node_validation_orchestrator",
    "node_validation_executor",
    "node_validation_adjudicator",
)

_CANDIDATE = "onex.evt.platform.validation-candidate-submitted.v1"
_PLAN_CREATED = "onex.evt.platform.validation-plan-created.v1"
_EXECUTION_REQUESTED = "onex.evt.platform.validation-execution-requested.v1"
_CHECKS_COMPLETED = "onex.evt.platform.validation-checks-completed.v1"
_ADJUDICATION_REQUESTED = "onex.evt.platform.validation-adjudication-requested.v1"
_ADJUDICATION_COMPLETED = "onex.evt.platform.validation-adjudication-completed.v1"
_RESULT_PUBLISHED = "onex.evt.platform.validation-result-published.v1"
_LIFECYCLE_UPDATED = "onex.evt.platform.validation-lifecycle-updated.v1"

_HOP_TOPICS = (
    _PLAN_CREATED,
    _EXECUTION_REQUESTED,
    _CHECKS_COMPLETED,
    _ADJUDICATION_REQUESTED,
    _ADJUDICATION_COMPLETED,
    _RESULT_PUBLISHED,
)


def _raw(node: str) -> dict[str, object]:
    loaded = yaml.safe_load(
        (_NODES_ROOT / node / "contract.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def _event_bus(node: str, key: str) -> list[str]:
    block = _raw(node).get("event_bus") or {}
    assert isinstance(block, dict)
    return list(block.get(key) or ())


def _manifest() -> ModelAutoWiringManifest:
    paths = [_NODES_ROOT / node / "contract.yaml" for node in _PIPELINE_NODES]
    discovered = discover_contracts_from_paths(paths)
    return ModelAutoWiringManifest(
        contracts=tuple(getattr(discovered, "contracts", discovered))
    )


# ---------------------------------------------------------------------------
# Static closure — every hop has a producer AND a consumer in these contracts
# ---------------------------------------------------------------------------


def test_every_pipeline_hop_topic_has_a_publisher_and_a_subscriber() -> None:
    """Each request/response topic between two nodes has both ends declared.

    RED before the fix: ``execution-requested`` and ``adjudication-requested``
    had a publisher declaration and no subscriber, ``checks-completed`` and
    ``adjudication-completed`` were subscribed by the orchestrator but published
    by a node that nothing consumed from, and the executor / adjudicator
    subscribed to nothing.
    """
    published: dict[str, list[str]] = {t: [] for t in _HOP_TOPICS}
    subscribed: dict[str, list[str]] = {t: [] for t in _HOP_TOPICS}
    for node in _PIPELINE_NODES:
        for topic in _event_bus(node, "publish_topics"):
            if topic in published:
                published[topic].append(node)
        for topic in _event_bus(node, "subscribe_topics"):
            if topic in subscribed:
                subscribed[topic].append(node)

    # result-published is the pipeline's terminal event: omnidash and the
    # validation ledger consume it, so it needs a publisher here, not a
    # subscriber among these three nodes.
    unpublished = [t for t in _HOP_TOPICS if not published[t]]
    unsubscribed = [t for t in _HOP_TOPICS[:-1] if not subscribed[t]]
    assert not unpublished, (
        f"declared by no pipeline node as a publish topic: {unpublished}"
    )
    assert not unsubscribed, f"no pipeline node subscribes to: {unsubscribed}"


def test_every_subscribed_topic_has_its_own_routing_entry() -> None:
    """A subscribe topic that no topic-scoped routing entry owns is a dead lane.

    The orchestrator's single unscoped entry owned all three topics, so
    ``checks-completed`` and ``adjudication-completed`` dispatched to the
    candidate handler (OMN-18013 documents the same defect on another node).
    """
    for node in _PIPELINE_NODES:
        raw = _raw(node)
        routing = raw.get("handler_routing") or {}
        assert isinstance(routing, dict)
        entries = routing.get("handlers") or []
        scoped = {e.get("topic") for e in entries if isinstance(e, dict)}
        for topic in _event_bus(node, "subscribe_topics"):
            assert topic in scoped, f"{node}: no topic-scoped handler for {topic}"


def test_every_publishing_pipeline_contract_declares_its_published_events() -> None:
    """No ``published_events`` map means the kernel builds NO result applier.

    The kernel registers a ``DispatchResultApplier`` only for a contract whose
    ``published_events`` map is non-empty; a handler's returned model on any
    other contract is dropped with a green dispatch.
    """
    for node in _PIPELINE_NODES:
        path = _NODES_ROOT / node / "contract.yaml"
        assert load_published_events_map(path), f"{node}: published_events is empty"


# ---------------------------------------------------------------------------
# End to end — the real dispatch seam, hop by hop
# ---------------------------------------------------------------------------


class _Tap:
    """Records every envelope seen on the pipeline topics, keyed by topic."""

    def __init__(self) -> None:
        self.seen: dict[str, list[ModelEventEnvelope[object]]] = {
            t: [] for t in (*_HOP_TOPICS, _LIFECYCLE_UPDATED)
        }

    def collector(self, topic: str) -> Callable[[ModelEventMessage], Awaitable[None]]:
        async def collect(message: ModelEventMessage) -> None:
            self.seen[topic].append(
                ModelEventEnvelope[object].model_validate_json(message.value)
            )

        return collect


async def _boot(bus: EventBusInmemory) -> _Tap:
    """Wire the three real contracts the way the kernel does and tap every hop."""
    tap = _Tap()
    for topic in (*_HOP_TOPICS, _LIFECYCLE_UPDATED):
        await bus.subscribe(
            topic, group_id=f"omn21056-tap-{topic}", on_message=tap.collector(topic)
        )

    manifest = _manifest()
    appliers = {}
    for contract in manifest.contracts:
        if contract.event_bus is None or not contract.event_bus.publish_topics:
            continue
        published_events = load_published_events_map(Path(contract.contract_path))
        if not published_events:
            continue
        topics = tuple(contract.event_bus.publish_topics)
        appliers[contract.name] = build_contract_result_applier(
            event_bus=bus,
            contract_path=Path(contract.contract_path),
            publish_topics=topics,
            output_topic=topics[0],
            output_topic_map=published_events,
            allowed_output_topics=topics,
        )

    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="local",
        result_appliers_by_contract=appliers,
    )
    engine.freeze()
    outcomes = {r.contract_name: r.outcome for r in report.results}
    assert outcomes == dict.fromkeys(_PIPELINE_NODES, EnumWiringOutcome.WIRED), outcomes
    return tap


async def _publish(
    bus: EventBusInmemory, topic: str, payload: object, correlation_id: UUID
) -> None:
    envelope = ModelEventEnvelope[object](
        payload=payload,
        correlation_id=correlation_id,
        event_type=derive_event_type_alias_for_topic(topic),
    )
    await bus.publish(topic, None, envelope.model_dump_json().encode("utf-8"), None)


async def _wait_for(
    tap: _Tap, topic: str, correlation_id: UUID
) -> list[ModelEventEnvelope[object]]:
    for _ in range(100):
        hits = [e for e in tap.seen[topic] if e.correlation_id == correlation_id]
        if hits:
            await asyncio.sleep(0.1)  # let a duplicate publish land, if any
            return [e for e in tap.seen[topic] if e.correlation_id == correlation_id]
        await asyncio.sleep(0.05)
    return []


def _candidate_payload() -> dict[str, object]:
    return {
        "candidate_id": str(uuid4()),
        "pattern_id": str(uuid4()),
        "source_path": "synthetic/omn21056",
        "diff_summary": "synthetic candidate",
        "changed_files": ["synthetic/a.py"],
        "risk_tags": [],
    }


@pytest.mark.asyncio
async def test_one_candidate_reaches_exactly_one_result_published() -> None:
    """The candidate walks every hop and ends in ONE result-published event."""
    bus = EventBusInmemory(environment="test", group="omn21056-e2e")
    await bus.start()
    try:
        tap = await _boot(bus)
        correlation_id = uuid4()
        await _publish(bus, _CANDIDATE, _candidate_payload(), correlation_id)

        results = await _wait_for(tap, _RESULT_PUBLISHED, correlation_id)
        per_hop = {
            t: len([e for e in tap.seen[t] if e.correlation_id == correlation_id])
            for t in _HOP_TOPICS
        }
    finally:
        await bus.close()

    assert per_hop == dict.fromkeys(_HOP_TOPICS, 1), per_hop
    payload = results[0].payload
    assert isinstance(payload, dict)
    assert payload["verdict"] == "pass"
    assert payload["total_checks"] == 12


def _check(
    code: str, severity: EnumCheckSeverity, *, passed: bool
) -> dict[str, object]:
    return {
        "check_code": code,
        "label": code,
        "severity": severity.value,
        "passed": passed,
        "skipped": False,
        "message": "" if passed else f"{code} failed",
    }


async def _adjudicate(
    checks: list[dict[str, object]], *, group: str
) -> tuple[list[ModelEventEnvelope[object]], list[ModelEventEnvelope[object]]]:
    """Enter at the adjudicator's own subscribe topic, the way the runtime does.

    The skeleton executor passes every enabled check, so a failed check can only
    be injected here, with the payload the orchestrator emits. Returns the
    result-published envelopes and the lifecycle-updated envelopes seen for the
    correlation id.
    """
    bus = EventBusInmemory(environment="test", group=group)
    await bus.start()
    try:
        tap = await _boot(bus)
        correlation_id = uuid4()
        await _publish(
            bus,
            _ADJUDICATION_REQUESTED,
            {
                "plan_id": str(uuid4()),
                "candidate_id": str(uuid4()),
                "check_results": checks,
                "score_threshold": 0.8,
            },
            correlation_id,
        )
        results = await _wait_for(tap, _RESULT_PUBLISHED, correlation_id)
        lifecycle = [
            e
            for e in tap.seen[_LIFECYCLE_UPDATED]
            if e.correlation_id == correlation_id
        ]
    finally:
        await bus.close()
    return results, lifecycle


@pytest.mark.asyncio
async def test_validation_pipeline_failed_required_check_publishes_a_fail_result() -> (
    None
):
    """AC3: a failed REQUIRED check is a FAIL result and promotes nothing."""
    results, lifecycle = await _adjudicate(
        [
            _check("CHECK-PY-001", EnumCheckSeverity.REQUIRED, passed=False),
            _check("CHECK-PY-002", EnumCheckSeverity.REQUIRED, passed=True),
        ],
        group="omn21056-fail",
    )

    assert len(results) == 1
    payload = results[0].payload
    assert isinstance(payload, dict)
    assert payload["verdict"] == "fail"
    assert payload["blocking_failures"] == ["CHECK-PY-001"]
    assert lifecycle == [], "a FAIL must not reach the lifecycle (promotion) topic"


@pytest.mark.asyncio
async def test_validation_pipeline_quarantine_when_score_is_below_threshold() -> None:
    """AC3: failed RECOMMENDED checks under the threshold are a QUARANTINE result."""
    results, lifecycle = await _adjudicate(
        [
            _check("CHECK-PY-001", EnumCheckSeverity.REQUIRED, passed=True),
            _check("CHECK-TEST-002", EnumCheckSeverity.RECOMMENDED, passed=False),
            _check("CHECK-VAL-001", EnumCheckSeverity.RECOMMENDED, passed=False),
            _check("CHECK-RISK-002", EnumCheckSeverity.RECOMMENDED, passed=True),
        ],
        group="omn21056-quarantine",
    )

    assert len(results) == 1
    payload = results[0].payload
    assert isinstance(payload, dict)
    assert payload["verdict"] == "quarantine"
    assert payload["blocking_failures"] == []
    assert payload["score"] == 0.5
    assert lifecycle == [], (
        "a QUARANTINE must not reach the lifecycle (promotion) topic"
    )


# ---------------------------------------------------------------------------
# Ledger — a published result must leave a row in validation_event_ledger
# ---------------------------------------------------------------------------


def test_ledger_projection_subscribes_to_result_published_with_its_own_entry() -> None:
    """The terminal event is projected into the validation ledger.

    RED before the fix: the projection read only the three cross-repo
    validation topics, so a pattern-candidate result never became a ledger row.
    """
    node = "node_validation_ledger_projection_compute"
    assert _RESULT_PUBLISHED in _event_bus(node, "subscribe_topics")
    routing = _raw(node)["handler_routing"]
    assert isinstance(routing, dict)
    assert _RESULT_PUBLISHED in {e.get("topic") for e in routing["handlers"]}


@pytest.mark.asyncio
async def test_result_published_record_projects_to_a_ledger_append_intent() -> None:
    """The real projection handler turns a published result into the append intent."""
    from datetime import UTC, datetime

    from omnibase_core.models.reducer.model_intent import ModelIntent
    from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
    from omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers.handler_validation_ledger_projection import (
        HandlerValidationLedgerProjection,
    )

    correlation_id = uuid4()
    envelope = ModelEventEnvelope[object](
        payload={"verdict": "pass", "candidate_id": str(uuid4())},
        correlation_id=correlation_id,
        event_type=derive_event_type_alias_for_topic(_RESULT_PUBLISHED),
    )
    message = ModelEventMessage(
        topic=_RESULT_PUBLISHED,
        value=envelope.model_dump_json().encode("utf-8"),
        headers=ModelEventHeaders(
            correlation_id=correlation_id,
            message_id=uuid4(),
            timestamp=datetime.now(UTC),
            source="local",
            event_type=derive_event_type_alias_for_topic(_RESULT_PUBLISHED),
        ),
        offset="0",
        partition=0,
    )

    output = await HandlerValidationLedgerProjection().handle(message)

    intent = output.result
    assert isinstance(intent, ModelIntent)
    payload = intent.payload
    assert isinstance(payload, ModelPayloadValidationLedgerAppend)
    assert payload.intent_type == "validation_ledger.append"
    assert payload.kafka_topic == _RESULT_PUBLISHED
    stored = base64.b64decode(payload.envelope_bytes)
    assert str(correlation_id).encode() in stored
