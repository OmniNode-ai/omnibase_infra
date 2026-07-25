# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay-then-diff verifier for ``workflow_receipt.json`` (OMN-15095).

Filed from adversarial decomposition of OMN-10858. Blocked by OMN-15094
(#663, merged 2026-07-25T22:25:23Z as f262cfe8). Re-verified against the
ACTUAL landed implementation before building (per this ticket's own
instruction) and found two real deviations from the ticket's assumptions:

1. ``node_kafka_replay_compute`` lives in THIS repo (``omnibase_infra``),
   not co-located with OMN-15094's renderer (``omninode_infra``,
   ``docker/onex-api``) -- confirmed via its ``contract.yaml``
   (``handler_routing: [{operation: "kafka.replay", handler: HandlerKafkaReplay}]``).
   Adding ``omnibase-infra`` as a pip dependency of ``docker/onex-api`` is
   NOT viable: omnibase-infra 0.36.1 requires ``fastapi<0.137``,
   ``redis<8.0``, ``uvicorn<0.50``, all three violated by onex-api's live
   pins (``fastapi==0.139.2``, ``redis==8.0.1``, ``uvicorn==0.51.0``) --
   verified against the published PyPI metadata 2026-07-25. This verifier
   therefore lives here, in the repo that already owns the replay
   capability, and consumes OMN-15094's rendered ``workflow_receipt.json``
   as a durable artifact rather than importing its renderer.

2. ``HandlerKafkaReplay.handle()`` (``ModelKafkaReplayOutput``) does not
   expose decoded envelope payload content -- only ``correlation_id_chain``,
   offsets, and counts. Its ``envelope_deserializer`` extension point is used
   here to run the SAME parsing this repo would use for any other envelope,
   with the decoded terminal-event fields captured into a side-channel
   (``ReplayCapture``) this verifier owns, since the handler's own return
   contract has no field for them.

Because a delegation-terminal event carries only the four terminal fields
(correlation_id, status, model_used, total_tokens, latency_ms) -- not the
whole ``gateway_workflows`` row (submitted_at, workflow_type, etc.) -- this
verifier can only recompute and diff ``terminal_event_hash`` from a Kafka
replay. ``projection_row_hash`` covers row fields the terminal event never
carries and is out of this verifier's replay-provable scope; it is reported
alongside the receipt for visibility but is not diffed here (documented,
not silently dropped).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from uuid import UUID

from omnibase_infra.nodes.node_kafka_replay_compute.handlers.handler_replay import (
    HandlerKafkaReplay,
)
from omnibase_infra.nodes.node_kafka_replay_compute.models import (
    ModelKafkaReplayInput,
)
from omnibase_infra.nodes.node_kafka_replay_compute.protocols import ConsumerFactory
from omnibase_infra.verification.workflow_receipt.enum_workflow_receipt_replay_verdict import (
    EnumWorkflowReceiptReplayVerdict,
)
from omnibase_infra.verification.workflow_receipt.enum_workflow_receipt_terminal_status import (
    EnumWorkflowReceiptTerminalStatus,
)
from omnibase_infra.verification.workflow_receipt.model_workflow_receipt_replay_diagnostic import (
    ModelWorkflowReceiptReplayDiagnostic,
)
from omnibase_infra.verification.workflow_receipt.model_workflow_receipt_replay_input import (
    ModelWorkflowReceiptReplayInput,
)
from omnibase_infra.verification.workflow_receipt.model_workflow_receipt_replay_result import (
    ModelWorkflowReceiptReplayResult,
)
from omnibase_infra.verification.workflow_receipt.receipt_hash import (
    sha256_of,
    terminal_event_payload,
)
from omnibase_infra.verification.workflow_receipt.wire_shape import (
    DELEGATION_COMPLETED_TOPIC,
    DELEGATION_FAILED_TOPIC,
    WorkflowReceiptReplayEnvelopeError,
    parse_delegation_terminal_envelope,
)

logger = logging.getLogger(__name__)

_STATUS_TOPIC: dict[EnumWorkflowReceiptTerminalStatus, str] = {
    EnumWorkflowReceiptTerminalStatus.COMPLETED: DELEGATION_COMPLETED_TOPIC,
    EnumWorkflowReceiptTerminalStatus.FAILED: DELEGATION_FAILED_TOPIC,
}


@dataclass
class ReplayCapture:
    """Side-channel the capturing deserializer writes into.

    ``HandlerKafkaReplay`` only needs ``.correlation_id`` off whatever the
    deserializer returns (``ProtocolReplayEnvelope``); this dataclass
    satisfies that structural protocol while the closure below stashes the
    full decoded terminal payload here for the verifier to read after
    ``handle()`` returns.
    """

    correlation_id: UUID | None
    decoded: list[tuple[UUID, str, str, int, int]] = field(default_factory=list)


def _make_capturing_deserializer(topic: str, capture: ReplayCapture) -> object:
    def _deserialize(value: bytes) -> ReplayCapture:
        envelope = json.loads(value.decode("utf-8"))
        try:
            decoded = parse_delegation_terminal_envelope(topic, envelope)
        except WorkflowReceiptReplayEnvelopeError:
            logger.exception(
                "replay verifier: failed to decode delegation-terminal "
                "envelope on topic=%s",
                topic,
            )
            return ReplayCapture(correlation_id=None)
        capture.decoded.append(decoded)
        return ReplayCapture(correlation_id=decoded[0])

    return _deserialize


async def replay_and_diff(
    receipt: ModelWorkflowReceiptReplayInput,
    *,
    target_cluster_bootstrap: str,
    target_consumer_group: str,
    consumer_factory: ConsumerFactory | None = None,
) -> ModelWorkflowReceiptReplayResult:
    """Replay the recorded delegation-terminal event for ``receipt`` via the
    real ``kafka.replay`` operation and diff the reconstructed
    ``terminal_event_hash`` against the one already on the receipt.

    Args:
        receipt: the already-rendered OMN-15094 receipt subset to verify.
        target_cluster_bootstrap: injected isolated target cluster (never
            ``.201`` -- ``ModelKafkaReplayInput`` itself rejects that).
        target_consumer_group: consumer group for this replay run.
        consumer_factory: test seam for an isolated/fake Kafka fixture.

    Returns:
        PASS if the replayed event reproduces the receipt's
        ``terminal_event_hash`` byte-for-byte; FAIL with a named diagnostic
        otherwise -- never a silent pass, never a bare exception escaping to
        the caller for a hash mismatch.
    """
    topic = _STATUS_TOPIC[receipt.status]
    capture = ReplayCapture(correlation_id=None)
    handler = HandlerKafkaReplay(
        consumer_factory=consumer_factory,
        envelope_deserializer=_make_capturing_deserializer(topic, capture),  # type: ignore[arg-type]
    )
    replay_input = ModelKafkaReplayInput(
        topics=[topic],
        target_cluster_bootstrap=target_cluster_bootstrap,
        target_consumer_group=target_consumer_group,
        expected_event_count=1,
    )
    output = await handler.handle(replay_input)

    diagnostics: list[ModelWorkflowReceiptReplayDiagnostic] = []

    matching = [d for d in capture.decoded if d[0] == receipt.correlation_id]
    if not matching:
        diagnostics.append(
            ModelWorkflowReceiptReplayDiagnostic(
                field="correlation_id",
                receipt_value=str(receipt.correlation_id),
                replayed_value=(
                    str([str(d[0]) for d in capture.decoded])
                    if capture.decoded
                    else "<no event decoded>"
                ),
            )
        )
        return ModelWorkflowReceiptReplayResult(
            workflow_id=receipt.workflow_id,
            correlation_id=receipt.correlation_id,
            verdict=EnumWorkflowReceiptReplayVerdict.FAIL,
            events_replayed=output.events_replayed,
            replayed_topic=topic,
            terminal_event_hash_match=False,
            diagnostics=tuple(diagnostics),
        )

    _, replayed_status, model_used, total_tokens, latency_ms = matching[0]
    replayed_hash = sha256_of(
        terminal_event_payload(
            correlation_id=str(receipt.correlation_id),
            status=replayed_status,
            terminal_model_used=model_used,
            terminal_total_tokens=total_tokens,
            terminal_latency_ms=latency_ms,
        )
    )

    if replayed_hash == receipt.terminal_event_hash:
        return ModelWorkflowReceiptReplayResult(
            workflow_id=receipt.workflow_id,
            correlation_id=receipt.correlation_id,
            verdict=EnumWorkflowReceiptReplayVerdict.PASS,
            events_replayed=output.events_replayed,
            replayed_topic=topic,
            terminal_event_hash_match=True,
        )

    if replayed_status != receipt.status.value:
        diagnostics.append(
            ModelWorkflowReceiptReplayDiagnostic(
                field="status",
                receipt_value=receipt.status.value,
                replayed_value=replayed_status,
            )
        )
    if model_used != receipt.terminal_model_used:
        diagnostics.append(
            ModelWorkflowReceiptReplayDiagnostic(
                field="terminal_model_used",
                receipt_value=receipt.terminal_model_used,
                replayed_value=model_used,
            )
        )
    if total_tokens != receipt.terminal_total_tokens:
        diagnostics.append(
            ModelWorkflowReceiptReplayDiagnostic(
                field="terminal_total_tokens",
                receipt_value=str(receipt.terminal_total_tokens),
                replayed_value=str(total_tokens),
            )
        )
    if latency_ms != receipt.terminal_latency_ms:
        diagnostics.append(
            ModelWorkflowReceiptReplayDiagnostic(
                field="terminal_latency_ms",
                receipt_value=str(receipt.terminal_latency_ms),
                replayed_value=str(latency_ms),
            )
        )
    if not diagnostics:
        # Hashes differ despite every known field matching -- report the
        # hash itself rather than claiming a match that didn't happen.
        diagnostics.append(
            ModelWorkflowReceiptReplayDiagnostic(
                field="terminal_event_hash",
                receipt_value=receipt.terminal_event_hash,
                replayed_value=replayed_hash,
            )
        )

    return ModelWorkflowReceiptReplayResult(
        workflow_id=receipt.workflow_id,
        correlation_id=receipt.correlation_id,
        verdict=EnumWorkflowReceiptReplayVerdict.FAIL,
        events_replayed=output.events_replayed,
        replayed_topic=topic,
        terminal_event_hash_match=False,
        diagnostics=tuple(diagnostics),
    )


__all__ = ["ReplayCapture", "replay_and_diff"]
