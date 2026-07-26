# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15095 acceptance tests: replay-then-diff verifier for
``workflow_receipt.json``.

Drives the REAL ``HandlerKafkaReplay`` (``kafka.replay`` operation, the exact
handler named in ``node_kafka_replay_compute``'s ``contract.yaml``) through
its documented ``consumer_factory``/``envelope_deserializer`` test seams --
not a hand-rolled mock of the verifier's own logic.

Ticket acceptance tests:
    1. No replay/verify wrapper existed before this change (RED confirmed by
       this module's existence and the import below).
    2. Given a receipt and the original event log, replay reproduces
       byte-identical projection state and reports PASS.
    3. A deliberately mutated replay input reports FAIL with a clear
       hash-mismatch diagnostic (never a silent pass, never an unhandled
       exception).
"""

from __future__ import annotations

import json
import uuid

from omnibase_infra.verification.workflow_receipt import (
    ModelWorkflowReceiptReplayInput,
    replay_and_diff,
)
from omnibase_infra.verification.workflow_receipt.receipt_hash import (
    sha256_of,
    terminal_event_payload,
)
from omnibase_infra.verification.workflow_receipt.wire_shape import (
    DELEGATION_COMPLETED_TOPIC,
)

from ._fake_replay_consumer import FakeReplayConsumer

_BOOTSTRAP = "isolated-redpanda:9092"
_GROUP = "omn-15095-replay-verify-test"


def _delegation_completed_message(
    *, correlation_id: uuid.UUID, model_used: str, total_tokens: int, latency_ms: int
) -> bytes:
    """Build one raw delegation-completed wire message -- the same shape
    ``workflow_terminal_consumer.parse_terminal_envelope`` expects."""
    return json.dumps(
        {
            "correlation_id": str(correlation_id),
            "payload": {
                "correlation_id": str(correlation_id),
                "model_used": model_used,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
            },
        }
    ).encode("utf-8")


def _receipt_for(
    *,
    correlation_id: uuid.UUID,
    model_used: str,
    total_tokens: int,
    latency_ms: int,
) -> ModelWorkflowReceiptReplayInput:
    terminal_hash = sha256_of(
        terminal_event_payload(
            correlation_id=str(correlation_id),
            status="completed",
            terminal_model_used=model_used,
            terminal_total_tokens=total_tokens,
            terminal_latency_ms=latency_ms,
        )
    )
    return ModelWorkflowReceiptReplayInput(
        workflow_id=uuid.uuid4(),
        correlation_id=correlation_id,
        status="completed",
        terminal_model_used=model_used,
        terminal_total_tokens=total_tokens,
        terminal_latency_ms=latency_ms,
        projection_row_hash="0" * 64,
        terminal_event_hash=terminal_hash,
    )


class TestReplayReproducesReceipt:
    """Acceptance test 2: replay reproduces byte-identical terminal state."""

    async def test_replay_matches_receipt_reports_pass(self) -> None:
        correlation_id = uuid.uuid4()
        receipt = _receipt_for(
            correlation_id=correlation_id,
            model_used="glm-4.6",
            total_tokens=456,
            latency_ms=1234,
        )
        message = _delegation_completed_message(
            correlation_id=correlation_id,
            model_used="glm-4.6",
            total_tokens=456,
            latency_ms=1234,
        )

        def consumer_factory(_command: object) -> FakeReplayConsumer:
            return FakeReplayConsumer({DELEGATION_COMPLETED_TOPIC: [message]})

        result = await replay_and_diff(
            receipt,
            target_cluster_bootstrap=_BOOTSTRAP,
            target_consumer_group=_GROUP,
            consumer_factory=consumer_factory,
        )

        assert result.verdict == "PASS"
        assert result.terminal_event_hash_match is True
        assert result.events_replayed == 1
        assert result.diagnostics == ()
        assert result.replayed_topic == DELEGATION_COMPLETED_TOPIC


class TestMutatedReplayReportsFail:
    """Acceptance test 3: a mutated replay input fails with a clear diagnostic."""

    async def test_mutated_total_tokens_reports_fail_with_diagnostic(self) -> None:
        correlation_id = uuid.uuid4()
        receipt = _receipt_for(
            correlation_id=correlation_id,
            model_used="glm-4.6",
            total_tokens=456,
            latency_ms=1234,
        )
        # Deliberately mutated: total_tokens differs from what the receipt
        # (and its terminal_event_hash) recorded.
        mutated_message = _delegation_completed_message(
            correlation_id=correlation_id,
            model_used="glm-4.6",
            total_tokens=999,
            latency_ms=1234,
        )

        def consumer_factory(_command: object) -> FakeReplayConsumer:
            return FakeReplayConsumer({DELEGATION_COMPLETED_TOPIC: [mutated_message]})

        result = await replay_and_diff(
            receipt,
            target_cluster_bootstrap=_BOOTSTRAP,
            target_consumer_group=_GROUP,
            consumer_factory=consumer_factory,
        )

        assert result.verdict == "FAIL"
        assert result.terminal_event_hash_match is False
        assert len(result.diagnostics) == 1
        diagnostic = result.diagnostics[0]
        assert diagnostic.field == "terminal_total_tokens"
        assert diagnostic.receipt_value == "456"
        assert diagnostic.replayed_value == "999"

    async def test_no_matching_correlation_id_yields_fail_diagnostic(self) -> None:
        correlation_id = uuid.uuid4()
        other_correlation_id = uuid.uuid4()
        receipt = _receipt_for(
            correlation_id=correlation_id,
            model_used="glm-4.6",
            total_tokens=456,
            latency_ms=1234,
        )
        unrelated_message = _delegation_completed_message(
            correlation_id=other_correlation_id,
            model_used="glm-4.6",
            total_tokens=456,
            latency_ms=1234,
        )

        def consumer_factory(_command: object) -> FakeReplayConsumer:
            return FakeReplayConsumer({DELEGATION_COMPLETED_TOPIC: [unrelated_message]})

        result = await replay_and_diff(
            receipt,
            target_cluster_bootstrap=_BOOTSTRAP,
            target_consumer_group=_GROUP,
            consumer_factory=consumer_factory,
        )

        assert result.verdict == "FAIL"
        assert result.terminal_event_hash_match is False
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].field == "correlation_id"
