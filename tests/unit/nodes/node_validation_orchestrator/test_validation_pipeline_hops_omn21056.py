# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21056 -- every hop of the validation pipeline has a publisher and a subscriber.

Live finding (the lab dev lane runtime, read 2026-10-12): a ``validation-candidate-submitted``
event reached ``validation-plan-created`` and stopped. ``execution-requested``,
``checks-completed``, ``adjudication-requested``, ``adjudication-completed`` and
``result-published`` all sat at offset 0 because

* the orchestrator declared ``execution-requested`` and ``adjudication-requested`` but
  its sole handler entry owned all three subscribe topics (it only builds plans);
* the executor and the adjudicator declared no ``subscribe_topics`` at all;
* the adjudicator declared no handler, and nothing consumed ``adjudication-completed``
  to publish ``result-published``.

These tests drive the REAL contracts through the REAL runtime entry points: contract
discovery, handler-entry topic ownership, the production dispatch callback
(``handler_wiring._make_dispatch_callback``) over the REAL handlers, and the real
contract-derived result applier (``build_contract_result_applier``) that picks the
publish topic. Nothing is faked except the event bus the applier publishes onto.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import yaml
from pydantic import BaseModel

import omnibase_infra.nodes as nodes_pkg
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumCheckSeverity, EnumValidationVerdict
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.validation import (
    ModelCheckResult,
    ModelExecutorResult,
)
from omnibase_infra.nodes.node_validation_orchestrator.models import (
    ModelPatternCandidate,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
    _topics_for_handler_entry,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract
from omnibase_infra.runtime.service_dispatch_result_applier import (
    build_contract_result_applier,
)

pytestmark = [pytest.mark.unit]

_NODES_DIR = Path(cast("str", nodes_pkg.__file__)).parent

_CANDIDATE = "onex.evt.platform.validation-candidate-submitted.v1"
_PLAN_CREATED = "onex.evt.platform.validation-plan-created.v1"
_EXECUTION_REQUESTED = "onex.evt.platform.validation-execution-requested.v1"
_CHECKS_COMPLETED = "onex.evt.platform.validation-checks-completed.v1"
_ADJUDICATION_REQUESTED = "onex.evt.platform.validation-adjudication-requested.v1"
_ADJUDICATION_COMPLETED = "onex.evt.platform.validation-adjudication-completed.v1"
_RESULT_PUBLISHED = "onex.evt.platform.validation-result-published.v1"

_PIPELINE_NODES = (
    "node_validation_orchestrator",
    "node_validation_executor",
    "node_validation_adjudicator",
    "node_validation_ledger_projection_compute",
)

# Topics that are lifecycle notifications with no pipeline consumer by design.
_NOTIFICATION_ONLY = frozenset(
    {
        _PLAN_CREATED,
        "onex.evt.platform.validation-lifecycle-updated.v1",
        "onex.evt.validation.cross-repo-run-started.v1",
        "onex.evt.validation.cross-repo-run-completed.v1",
        "onex.evt.validation.cross-repo-violations-batch.v1",
    }
)


def _contract_path(node: str) -> Path:
    return _NODES_DIR / node / "contract.yaml"


def _event_bus_topics(node: str, key: str) -> set[str]:
    raw = yaml.safe_load(_contract_path(node).read_text(encoding="utf-8"))
    return set((raw.get("event_bus") or {}).get(key) or [])


def _discovered() -> dict[str, ModelDiscoveredContract]:
    manifest = discover_contracts_from_paths(
        [_contract_path(node) for node in _PIPELINE_NODES]
    )
    assert not manifest.errors, manifest.errors
    return {contract.name: contract for contract in manifest.contracts}


# ---------------------------------------------------------------------------
# Static topic graph: no published request without a subscriber
# ---------------------------------------------------------------------------


def test_every_published_pipeline_request_has_a_subscriber() -> None:
    subscribed = set().union(
        *(_event_bus_topics(n, "subscribe_topics") for n in _PIPELINE_NODES)
    )
    published = set().union(
        *(_event_bus_topics(n, "publish_topics") for n in _PIPELINE_NODES)
    )
    orphans = sorted(
        topic
        for topic in published
        if topic.startswith("onex.evt.platform.validation-")
        and topic not in subscribed
        and topic not in _NOTIFICATION_ONLY
    )
    assert orphans == [], f"published with no subscriber: {orphans}"


def test_every_subscribed_pipeline_topic_has_a_publisher() -> None:
    subscribed = set().union(
        *(
            _event_bus_topics(n, "subscribe_topics")
            for n in _PIPELINE_NODES
            if n != "node_validation_ledger_projection_compute"
        )
    )
    published = set().union(
        *(_event_bus_topics(n, "publish_topics") for n in _PIPELINE_NODES)
    )
    external_entry = {_CANDIDATE}
    assert sorted(subscribed - published - external_entry) == []


def test_result_topic_reaches_the_validation_ledger() -> None:
    assert _RESULT_PUBLISHED in _event_bus_topics(
        "node_validation_ledger_projection_compute", "subscribe_topics"
    )


# ---------------------------------------------------------------------------
# Handler-entry ownership: each subscribed topic is owned by exactly one entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("node", "topic"),
    [
        ("node_validation_orchestrator", _CANDIDATE),
        ("node_validation_orchestrator", _CHECKS_COMPLETED),
        ("node_validation_orchestrator", _ADJUDICATION_COMPLETED),
        ("node_validation_executor", _EXECUTION_REQUESTED),
        ("node_validation_adjudicator", _ADJUDICATION_REQUESTED),
    ],
)
def test_each_subscribed_topic_has_exactly_one_owning_handler(
    node: str, topic: str
) -> None:
    contract = _discovered()[node]
    assert contract.handler_routing is not None
    owners = [
        entry.handler.name
        for entry in contract.handler_routing.handlers
        if topic in _topics_for_handler_entry(contract, entry)
    ]
    assert len(owners) == 1, f"{node}: topic {topic} owned by {owners}"


# ---------------------------------------------------------------------------
# End-to-end hops through the production dispatch callback + applier
# ---------------------------------------------------------------------------


async def _hop(node: str, topic: str, payload: BaseModel) -> tuple[str, BaseModel]:
    """Dispatch ``payload`` on ``topic`` to ``node`` exactly as the runtime does.

    Returns ``(published_topic, published_model)``.
    """
    contract = _discovered()[node]
    assert contract.handler_routing is not None
    assert contract.event_bus is not None
    (entry,) = [
        e
        for e in contract.handler_routing.handlers
        if topic in _topics_for_handler_entry(contract, e)
    ]
    module = importlib.import_module(entry.handler.module)
    handler = getattr(module, entry.handler.name)()
    callback = _make_dispatch_callback(
        cast("ProtocolHandleable", handler), entry.event_model
    )
    correlation_id = uuid4()
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=payload.model_dump(mode="json"),
        correlation_id=correlation_id,
        event_type=topic,
    )
    result = await callback(envelope)
    assert isinstance(result, ModelDispatchResult), f"handler never ran: {result!r}"
    assert len(result.output_events) == 1, result.output_events

    bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = build_contract_result_applier(
        event_bus=bus,
        contract_path=contract.contract_path,
        publish_topics=contract.event_bus.publish_topics,
    )
    await applier.apply(result, correlation_id=correlation_id)
    bus.publish_envelope.assert_awaited_once()
    return str(bus.publish_envelope.await_args.kwargs["topic"]), result.output_events[0]


def _field(event: BaseModel, name: str) -> object:
    return getattr(event, name)


def _candidate() -> ModelPatternCandidate:
    return ModelPatternCandidate(
        candidate_id=uuid4(),
        pattern_id=uuid4(),
        source_path="synthetic/omn-21056",
        diff_summary="synthetic candidate",
    )


def _check(code: str, severity: EnumCheckSeverity, *, passed: bool) -> ModelCheckResult:
    return ModelCheckResult(
        check_code=code,
        label=code,
        severity=severity,
        passed=passed,
        executed_at=datetime.now(tz=UTC),
    )


@pytest.mark.asyncio
async def test_candidate_becomes_an_execution_request() -> None:
    topic, event = await _hop("node_validation_orchestrator", _CANDIDATE, _candidate())
    assert topic == _EXECUTION_REQUESTED
    assert len(cast("tuple[object, ...]", _field(event, "checks"))) == 12


@pytest.mark.asyncio
async def test_execution_request_becomes_checks_completed() -> None:
    _, plan = await _hop("node_validation_orchestrator", _CANDIDATE, _candidate())
    topic, event = await _hop("node_validation_executor", _EXECUTION_REQUESTED, plan)
    assert topic == _CHECKS_COMPLETED
    assert isinstance(event, ModelExecutorResult)
    assert event.plan_id == _field(plan, "plan_id")


@pytest.mark.asyncio
async def test_checks_completed_becomes_an_adjudication_request() -> None:
    executor_result = ModelExecutorResult(
        plan_id=uuid4(),
        candidate_id=uuid4(),
        check_results=(
            _check("CHECK-PY-001", EnumCheckSeverity.REQUIRED, passed=True),
        ),
    )
    topic, _ = await _hop(
        "node_validation_orchestrator", _CHECKS_COMPLETED, executor_result
    )
    assert topic == _ADJUDICATION_REQUESTED


@pytest.mark.asyncio
async def test_failed_required_check_publishes_a_fail_result() -> None:
    """AC3: a failed REQUIRED check travels checks-completed -> result-published as FAIL."""
    plan_id, candidate_id = uuid4(), uuid4()
    executor_result = ModelExecutorResult(
        plan_id=plan_id,
        candidate_id=candidate_id,
        check_results=(
            _check("CHECK-PY-001", EnumCheckSeverity.REQUIRED, passed=False),
            _check("CHECK-PY-002", EnumCheckSeverity.REQUIRED, passed=True),
        ),
    )
    topic, request = await _hop(
        "node_validation_orchestrator", _CHECKS_COMPLETED, executor_result
    )
    assert topic == _ADJUDICATION_REQUESTED

    topic, verdict = await _hop(
        "node_validation_adjudicator", _ADJUDICATION_REQUESTED, request
    )
    assert topic == _ADJUDICATION_COMPLETED
    assert _field(verdict, "verdict") == EnumValidationVerdict.FAIL
    assert _field(verdict, "blocking_failures") == ("CHECK-PY-001",)

    topic, result = await _hop(
        "node_validation_orchestrator", _ADJUDICATION_COMPLETED, verdict
    )
    assert topic == _RESULT_PUBLISHED
    assert _field(result, "verdict") == EnumValidationVerdict.FAIL
    assert _field(result, "candidate_id") == candidate_id
    assert isinstance(_field(result, "candidate_id"), UUID)


@pytest.mark.asyncio
async def test_all_checks_passing_publishes_a_pass_result() -> None:
    _, plan = await _hop("node_validation_orchestrator", _CANDIDATE, _candidate())
    _, executed = await _hop("node_validation_executor", _EXECUTION_REQUESTED, plan)
    _, request = await _hop("node_validation_orchestrator", _CHECKS_COMPLETED, executed)
    _, verdict = await _hop(
        "node_validation_adjudicator", _ADJUDICATION_REQUESTED, request
    )
    topic, result = await _hop(
        "node_validation_orchestrator", _ADJUDICATION_COMPLETED, verdict
    )
    assert topic == _RESULT_PUBLISHED
    assert _field(result, "verdict") == EnumValidationVerdict.PASS


@pytest.mark.asyncio
async def test_validation_pipeline_quarantine_when_recommended_check_fails() -> None:
    """AC3: a failed RECOMMENDED check quarantines the candidate; nothing is promoted.

    Required checks all pass, but the weighted score (1 of 2) is under the plan's 0.8
    threshold, so the adjudicator quarantines. The orchestrator's only emission is the
    terminal ``result-published`` carrying the QUARANTINE verdict and its reasons.
    """
    candidate_id = uuid4()
    executor_result = ModelExecutorResult(
        plan_id=uuid4(),
        candidate_id=candidate_id,
        check_results=(
            _check("CHECK-PY-001", EnumCheckSeverity.REQUIRED, passed=True),
            _check("CHECK-TEST-002", EnumCheckSeverity.RECOMMENDED, passed=False),
        ),
    )
    _, request = await _hop(
        "node_validation_orchestrator", _CHECKS_COMPLETED, executor_result
    )
    _, verdict = await _hop(
        "node_validation_adjudicator", _ADJUDICATION_REQUESTED, request
    )
    assert _field(verdict, "verdict") == EnumValidationVerdict.QUARANTINE
    assert _field(verdict, "blocking_failures") == ()

    topic, result = await _hop(
        "node_validation_orchestrator", _ADJUDICATION_COMPLETED, verdict
    )
    assert topic == _RESULT_PUBLISHED
    assert _field(result, "verdict") == EnumValidationVerdict.QUARANTINE
    assert _field(result, "quarantine_reasons")
    assert _field(result, "candidate_id") == candidate_id
