# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26609 -- the dead letter of an unsequenced live ledger row survives the chain unreplayed.

The unit test of this ticket proves the replay decision on a hand-built ``ModelDlqMessage``.
That leaves the seam where the live defect lived: the publishing side writes the envelope, the
replay side reads it back, and the two agree only by convention. A replay decision that is right
on a hand-built message can still replay the real record if the real publisher names the failure
differently than the test assumed.

So this composes the real components:

    MixinKafkaDlq._publish_raw_to_dlq     (the real publisher; failure text from the 2026-11-24 incident)
        -> the published envelope BYTES
        -> ModelDlqMessage.from_kafka_message   (the real replay-side reader)
        -> should_replay                        (the real eligibility decision)

Only the Kafka producer is a double: the assertion is about the bytes it is handed, which are
byte-identical to what a live broker receives. No broker is needed, so this runs everywhere and
is never skipped.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.integration

_ORIGINAL_TOPIC = "onex.evt.omnimarket.work-ledger-status.v1"
_DLQ_TOPIC = "onex.dlq.omnibase-infra.intents.v1"
_ROW_ID = "be19e77f16e6008e3dcedc1c3aa3227256f3c8f3d6a4e496a2084619c02f431b"
# The failure text of the real dead letter, as the wedge guard flattened it.
_FAILURE_REASON = (
    "[ONEX_CORE_081_OPERATION_FAILED] projection withhold exhausted after 6 consecutive "
    f"identical refusals of {_ORIGINAL_TOPIC} partition 0 offset 271060 (bound 5): "
    "[ONEX_CORE_081_OPERATION_FAILED] projection handler WorkLedgerProjectionWriter consumed "
    f"an event from {_ORIGINAL_TOPIC} and wrote no row because its write path failed "
    "(UnsequencedLiveRowError: unsequenced-live-row: source 'onex-ledger' requires a "
    f"ledger_seq; row_id {_ROW_ID})"
)
_TRANSIENT_REASON = (
    "write path failed (InfraConnectionError: broker unreachable) after 6 consecutive "
    "identical refusals"
)
_CONFIG = ModelDlqReplayEngineConfig(
    bootstrap_servers="localhost:9092",
    dlq_topic=_DLQ_TOPIC,
)


class ProjectionWedgeExhaustedError(Exception):
    """Carries the wrapper's real class name, which is what the publisher records."""


class _CapturingProducer:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_and_wait(
        self,
        topic: str,
        *,
        value: bytes,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object:
        self.sent.append(
            {"topic": topic, "value": value, "key": key, "headers": headers}
        )
        return object()


class _DlqHost(MixinKafkaDlq):
    def __init__(self) -> None:
        self._config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            dead_letter_topic=_DLQ_TOPIC,
        )
        self._environment = "integration"
        self._group = "omn26609-chain"
        self.capture = _CapturingProducer()
        self._producer = cast("Any", self.capture)
        self._producer_lock = asyncio.Lock()
        self._timeout_seconds = 5
        self._init_dlq()

    def _model_headers_to_kafka(self, headers: object) -> list[tuple[str, bytes]]:
        return []


class _InboundRecord:
    def __init__(self, value: bytes) -> None:
        self.key = _ROW_ID.encode("utf-8")
        self.value = value
        self.offset = 271060
        self.partition = 0
        self.headers = ()


async def _round_trip(reason: str, correlation_id: UUID) -> ModelDlqMessage:
    host = _DlqHost()
    published = await host._publish_raw_to_dlq(
        original_topic=_ORIGINAL_TOPIC,
        raw_msg=_InboundRecord(json.dumps({"row_id": _ROW_ID}).encode("utf-8")),
        error=ProjectionWedgeExhaustedError(reason),
        correlation_id=correlation_id,
        failure_type="handler_exception",
        consumer_group="omn26609-chain",
    )
    assert published is True, "the publisher reported the write as not persisted"
    sent = host.capture.sent
    assert len(sent) == 1, sent
    envelope = json.loads(sent[0]["value"].decode("utf-8"))
    return ModelDlqMessage.from_kafka_message(
        payload=envelope, dlq_offset=1434803, dlq_partition=0
    )


@pytest.mark.asyncio
async def test_the_published_dead_letter_of_an_unsequenced_row_is_not_replayed() -> (
    None
):
    parsed = await _round_trip(_FAILURE_REASON, uuid4())

    # Control on the reader: the class name survived the publish/parse hop, so the
    # refusal below is the replay decision and not an artefact of a lost failure text.
    assert "UnsequencedLiveRowError" in parsed.failure_reason

    eligible, why = should_replay(parsed, _CONFIG)
    assert eligible is False, why
    assert "non-retryable" in why.lower()


@pytest.mark.asyncio
async def test_the_published_dead_letter_of_a_transient_failure_is_still_replayed() -> (
    None
):
    """Positive control: the chain does not refuse every dead letter of this topic."""
    parsed = await _round_trip(_TRANSIENT_REASON, uuid4())
    eligible, why = should_replay(parsed, _CONFIG)
    assert eligible is True, why
