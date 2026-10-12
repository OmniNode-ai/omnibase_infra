# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21056: one pattern candidate reaches a validation-result-published event.

The validation pipeline was declared across five node contracts but every hop
after the plan was dead on the runtime: the orchestrator published an
execution request and an adjudication request that nothing subscribed to, the
executor and adjudicator declared no subscription, and the validation ledger
projection never saw the result topic. On the lab dev lane
``execution-requested`` had a high watermark of 0 and ``validation_event_ledger``
held 0 rows.

These tests read the REAL contracts and drive each hop through the production
dispatch path: ``_make_dispatch_callback`` (the auto-wiring callback, with the
contract's own ``event_model`` and node kind) followed by the contract-derived
``DispatchResultApplier`` (which resolves each output event's topic from
``published_events``). The only substitute is the bus: published envelopes are
captured and routed to whichever contract declares a handler for that topic,
which is what the broker consumer groups do on the lane.
"""

from __future__ import annotations

import base64
import importlib
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

import omnibase_infra.nodes as _nodes_pkg
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumCheckSeverity, EnumValidationVerdict
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.models.validation import (
    ModelCheckResult,
    ModelValidationChecksCompleted,
    ModelValidationResultPublished,
)
from omnibase_infra.nodes.node_validation_orchestrator.models import (
    ModelPatternCandidate,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_dispatch_callback,
    _node_kind_from_node_type,
    _topics_for_handler_entry,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelDiscoveredContract,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    build_contract_result_applier,
)

pytestmark = [pytest.mark.unit]

_NODES_ROOT = Path(next(iter(_nodes_pkg.__path__)))
_PIPELINE_NODES = (
    "node_validation_orchestrator",
    "node_validation_executor",
    "node_validation_adjudicator",
    "node_validation_ledger_projection_compute",
    "node_validation_ledger_write_effect",
)


def _contracts() -> dict[str, ModelDiscoveredContract]:
    manifest = discover_contracts_from_paths(
        [_NODES_ROOT / node / "contract.yaml" for node in _PIPELINE_NODES]
    )
    assert not manifest.errors, manifest.errors
    return {contract.name: contract for contract in manifest.contracts}


def _topic(contracts: Mapping[str, ModelDiscoveredContract], suffix: str) -> str:
    """Resolve the one declared platform topic ending in ``suffix``."""
    declared = {
        topic
        for contract in contracts.values()
        if contract.event_bus is not None
        for topic in (
            *contract.event_bus.publish_topics,
            *contract.event_bus.subscribe_topics,
        )
        if topic.endswith(suffix)
    }
    assert len(declared) == 1, (suffix, declared)
    return next(iter(declared))


def _consumers(
    contracts: Mapping[str, ModelDiscoveredContract], topic: str
) -> list[tuple[ModelDiscoveredContract, ModelHandlerRoutingEntry]]:
    """Every (contract, handler entry) the runtime wires to consume ``topic``."""
    found: list[tuple[ModelDiscoveredContract, ModelHandlerRoutingEntry]] = []
    for contract in contracts.values():
        if contract.handler_routing is None or contract.event_bus is None:
            continue
        for entry in contract.handler_routing.handlers:
            if topic in _topics_for_handler_entry(contract, entry):
                found.append((contract, entry))
    return found


class _CaptureBus:
    """Stands in for the broker: records what the applier publishes."""

    def __init__(self) -> None:
        self.published: list[tuple[str, ModelEventEnvelope[BaseModel]]] = []

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
        *,
        key: bytes | None = None,
    ) -> None:
        assert isinstance(envelope, ModelEventEnvelope)
        self.published.append((topic, envelope))

    async def publish(
        self, topic: str, key: bytes | None, value: bytes
    ) -> ModelPublishReceipt | None:
        raise AssertionError("the applier publishes envelopes, not raw bytes")

    def get_consumer_groups(self) -> dict[tuple[str, str], str]:
        return {}


@dataclass(frozen=True)
class _Run:
    """What one submission did on the simulated bus."""

    delivered: list[tuple[str, BaseModel]]
    ledger_rows: list[dict[str, object]]


def _as_kafka_message(
    topic: str, envelope: ModelEventEnvelope[BaseModel]
) -> ModelEventMessage:
    """The raw record a ``ModelEventMessage`` consumer receives for ``envelope``."""
    assert envelope.correlation_id is not None
    return ModelEventMessage(
        topic=topic,
        value=envelope.model_dump_json().encode("utf-8"),
        headers=ModelEventHeaders(
            correlation_id=envelope.correlation_id,
            timestamp=envelope.envelope_timestamp,
            source="omn-21056-test",
            event_type=topic,
        ),
        offset="0",
        partition=0,
    )


async def _run_pipeline(
    contracts: Mapping[str, ModelDiscoveredContract],
    start_topic: str,
    start_payload: BaseModel,
    correlation_id: UUID,
) -> _Run:
    """Deliver ``start_payload`` and follow every published event to its consumers."""
    queue: deque[tuple[str, BaseModel, ModelEventEnvelope[BaseModel] | None]] = deque(
        [(start_topic, start_payload, None)]
    )
    delivered: list[tuple[str, BaseModel]] = []
    ledger_rows: list[dict[str, object]] = []
    while queue:
        topic, payload, source = queue.popleft()
        delivered.append((topic, payload))
        for contract, entry in _consumers(contracts, topic):
            module = importlib.import_module(entry.handler.module)
            instance = getattr(module, entry.handler.name)()
            callback = _make_dispatch_callback(
                instance,
                entry.event_model,
                _node_kind_from_node_type(contract.node_type),
                frozenset(load_published_events_map(contract.contract_path)),
            )
            raw_consumer = (
                entry.event_model is not None
                and entry.event_model.name == "ModelEventMessage"
            )
            wire_payload: object
            if raw_consumer:
                assert source is not None
                wire_payload = _as_kafka_message(topic, source)
            else:
                wire_payload = payload.model_dump(mode="json")
            result = await callback(
                ModelEventEnvelope[object](
                    payload=wire_payload, correlation_id=correlation_id
                )
            )
            assert result is not None
            if raw_consumer:
                # The projection emits a ledger-append intent; the intent
                # executor routes it to the write effect, outside this test.
                for intent in result.output_intents:
                    assert isinstance(intent, ModelIntent)
                    assert isinstance(intent.payload, BaseModel)
                    ledger_rows.append(intent.payload.model_dump(mode="json"))
                continue
            bus = _CaptureBus()
            assert contract.event_bus is not None
            applier = build_contract_result_applier(
                event_bus=bus,
                contract_path=contract.contract_path,
                publish_topics=contract.event_bus.publish_topics,
            )
            await applier.apply(result)
            for out_topic, envelope in bus.published:
                assert envelope.correlation_id == correlation_id
                queue.append((out_topic, envelope.payload, envelope))
        assert len(delivered) < 50, "pipeline did not terminate"
    return _Run(delivered=delivered, ledger_rows=ledger_rows)


def test_every_pipeline_topic_has_a_subscriber() -> None:
    """The report's finding: requests the orchestrator publishes had no subscriber."""
    contracts = _contracts()
    chain = (
        "validation-candidate-submitted.v1",
        "validation-plan-created.v1",
        "validation-execution-requested.v1",
        "validation-checks-completed.v1",
        "validation-adjudication-requested.v1",
        "validation-adjudication-completed.v1",
        "validation-result-published.v1",
    )
    orphans = [
        suffix
        for suffix in chain
        if not _consumers(contracts, _topic(contracts, suffix))
    ]
    assert orphans == [], f"published with no dispatched consumer: {orphans}"


@pytest.mark.asyncio
async def test_candidate_reaches_result_published_through_real_dispatch() -> None:
    """A candidate submitted on the bus ends as one result-published event."""
    contracts = _contracts()
    correlation_id = uuid4()
    candidate = ModelPatternCandidate(
        candidate_id=uuid4(),
        pattern_id=uuid4(),
        source_path="synthetic/omn-21056",
    )

    run = await _run_pipeline(
        contracts,
        _topic(contracts, "validation-candidate-submitted.v1"),
        candidate,
        correlation_id,
    )

    assert [topic for topic, _ in run.delivered] == [
        _topic(contracts, "validation-candidate-submitted.v1"),
        _topic(contracts, "validation-plan-created.v1"),
        _topic(contracts, "validation-execution-requested.v1"),
        _topic(contracts, "validation-checks-completed.v1"),
        _topic(contracts, "validation-adjudication-requested.v1"),
        _topic(contracts, "validation-adjudication-completed.v1"),
        _topic(contracts, "validation-result-published.v1"),
    ]
    published = run.delivered[-1][1]
    assert isinstance(published, ModelValidationResultPublished)
    assert published.candidate_id == candidate.candidate_id
    assert published.verdict is EnumValidationVerdict.PASS

    # The result is projected into the validation ledger, under the same
    # correlation id the candidate was submitted with.
    assert len(run.ledger_rows) == 1
    row = run.ledger_rows[0]
    assert row["kafka_topic"] == _topic(contracts, "validation-result-published.v1")
    assert row["correlation_id"] == str(correlation_id)
    stored = base64.b64decode(str(row["envelope_bytes"])).decode("utf-8")
    assert str(candidate.candidate_id) in stored


@pytest.mark.asyncio
async def test_failed_required_check_is_published_as_a_fail_verdict() -> None:
    """AC3: a failed REQUIRED check reaches result-published as FAIL, not PASS.

    Enters at ``checks-completed`` (the executor's output topic) with one failed
    REQUIRED check, through the same dispatch path the runtime uses.
    """
    contracts = _contracts()
    now = datetime(2026, 10, 12, tzinfo=UTC)
    checks_completed = ModelValidationChecksCompleted(
        plan_id=uuid4(),
        candidate_id=uuid4(),
        score_threshold=0.8,
        check_results=(
            ModelCheckResult(
                check_code="CHECK-PY-001",
                label="Typecheck (mypy)",
                severity=EnumCheckSeverity.REQUIRED,
                passed=False,
                message="mypy found 3 errors",
                executed_at=now,
            ),
            ModelCheckResult(
                check_code="CHECK-PY-002",
                label="Lint/format (ruff)",
                severity=EnumCheckSeverity.REQUIRED,
                passed=True,
                executed_at=now,
            ),
        ),
    )

    run = await _run_pipeline(
        contracts,
        _topic(contracts, "validation-checks-completed.v1"),
        checks_completed,
        uuid4(),
    )

    final_topic, published = run.delivered[-1]
    assert final_topic == _topic(contracts, "validation-result-published.v1")
    assert isinstance(published, ModelValidationResultPublished)
    assert published.verdict is EnumValidationVerdict.FAIL
    assert published.blocking_failures == ("CHECK-PY-001",)
    assert published.candidate_id == checks_completed.candidate_id
    assert len(run.ledger_rows) == 1


@pytest.mark.asyncio
async def test_validation_pipeline_quarantine_when_only_recommended_checks_fail() -> (
    None
):
    """AC3: recommended failures below the threshold are QUARANTINE, never PASS.

    No required check fails, so the verdict is the soft block. The run ends at
    result-published: nothing is published to a lifecycle or promotion topic.
    """
    contracts = _contracts()
    now = datetime(2026, 10, 12, tzinfo=UTC)

    def _check(
        code: str, severity: EnumCheckSeverity, passed: bool
    ) -> ModelCheckResult:
        return ModelCheckResult(
            check_code=code,
            label=code,
            severity=severity,
            passed=passed,
            executed_at=now,
        )

    checks_completed = ModelValidationChecksCompleted(
        plan_id=uuid4(),
        candidate_id=uuid4(),
        score_threshold=0.8,
        check_results=(
            _check("CHECK-PY-001", EnumCheckSeverity.REQUIRED, True),
            _check("CHECK-PY-002", EnumCheckSeverity.REQUIRED, True),
            _check("CHECK-TEST-002", EnumCheckSeverity.RECOMMENDED, False),
            _check("CHECK-VAL-001", EnumCheckSeverity.RECOMMENDED, False),
            _check("CHECK-RISK-002", EnumCheckSeverity.RECOMMENDED, False),
        ),
    )

    run = await _run_pipeline(
        contracts,
        _topic(contracts, "validation-checks-completed.v1"),
        checks_completed,
        uuid4(),
    )

    final_topic, published = run.delivered[-1]
    assert final_topic == _topic(contracts, "validation-result-published.v1")
    assert isinstance(published, ModelValidationResultPublished)
    assert published.verdict is EnumValidationVerdict.QUARANTINE
    assert published.blocking_failures == ()
    assert published.quarantine_reasons
    lifecycle_topic = _topic(contracts, "validation-lifecycle-updated.v1")
    assert lifecycle_topic not in [topic for topic, _ in run.delivered]


def test_result_topic_is_projected_into_the_validation_ledger() -> None:
    """The ledger projection must subscribe to the result topic, or no row exists."""
    contracts = _contracts()
    result_topic = _topic(contracts, "validation-result-published.v1")
    projection = contracts["node_validation_ledger_projection_compute"]
    assert projection.event_bus is not None
    assert result_topic in projection.event_bus.subscribe_topics
