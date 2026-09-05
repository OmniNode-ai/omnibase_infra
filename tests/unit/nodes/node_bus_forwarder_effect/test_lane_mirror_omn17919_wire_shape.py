# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17919: the lane mirror must read the wire shape the hook edge publishes.

OMN-17034 shipped the lane-mirror leg and it deployed correctly -- the forwarder
joined the stability broker, was assigned all four governed hook partitions, and
then mirrored **zero** records. 261 of 261 lane-mirror log lines since that
deploy were ``Lane mirror could not decode record``; the consumer group ended
``STATE Dead, MEMBERS 0`` and the dev mirror-target high-water marks never moved.

The cause is a fixture-vs-wire mismatch, not a wiring bug. ``NodeLaneMirror``
obtained its idempotency key by round-tripping the record through
``ModelEventEnvelope`` (``ServiceGatewayForwarder.decode_message``), which is
``extra="forbid"`` and requires a ``payload`` field. The hook edge publishes a
**flat** hook payload with the envelope metadata in Kafka **headers**. Every
fixture in ``test_lane_mirror_omn17034.py`` constructs a ``ModelEventEnvelope``,
which is precisely the shape the real lane does not carry, so the unit suite was
green while the live leg was inert.

These tests close that gap the only way it stays closed: the first one is a
record **captured off the stability lane with rpk**, value bytes and header set
exactly as the broker held them, rather than a shape this repo made up. The
mirror crosses no trust boundary and republishes byte-for-byte, so it never
needed the envelope at all -- the idempotency key it does need is already on the
wire as the mandatory ``message_id`` header that
``event_bus_kafka._model_headers_to_kafka`` stamps on every publish.

The refusal tests are the other half. Dropping the envelope round-trip must not
turn an unreadable record into a silent no-op: a record carrying no usable
identity is refused with a typed error and counted, so a systematically
malformed producer shows up as a rising counter instead of a mirror that quietly
moves nothing -- which is the exact failure mode this ticket exists to fix.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The captured record (AC5)
# ---------------------------------------------------------------------------
# Read off the live stability lane 2026-09-05, read-only:
#
#   docker exec omnibase-infra-stability-test-redpanda rpk topic consume \
#     onex.evt.omniclaude.prompt-submitted.v1 -n 1 -o -1 --brokers localhost:9092
#
# -> topic onex.evt.omniclaude.prompt-submitted.v1, partition 2, offset 285.
#
# Verbatim except for one substitution: the captured record's live agent-session
# UUID appears three times (value.session_id, value.correlation_id,
# value.entity_id) and once more as the ``correlation_id`` header, and is
# replaced here by a fixed placeholder UUID of identical form. Nothing else is
# altered -- key order, spacing, the explicit ``"causation_id": null``, the
# header set and its order are all as the broker held them. The record carries
# no credential or token material; ``prompt_length`` is a count, not content.
_CAPTURED_SESSION_UUID = "00000000-0000-4000-8000-000000000001"
_CAPTURED_MESSAGE_ID = "3022f95a-4496-453b-918b-366bee6e974a"
_CAPTURED_TOPIC = "onex.evt.omniclaude.prompt-submitted.v1"

_CAPTURED_VALUE: bytes = (
    '{"hook_source": "user_prompt_submit", "prompt_length": 9739, '
    f'"session_id": "{_CAPTURED_SESSION_UUID}", '
    '"working_directory": "drafts", '
    f'"correlation_id": "{_CAPTURED_SESSION_UUID}", '
    '"causation_id": null, '
    '"emitted_at": "2026-09-05T06:58:03.120765+00:00", '
    f'"entity_id": "{_CAPTURED_SESSION_UUID}", '
    '"schema_version": "1.0.0"}'
).encode()

_CAPTURED_HEADERS: dict[str, bytes] = {
    "content_type": b"application/json",
    "correlation_id": _CAPTURED_SESSION_UUID.encode("utf-8"),
    "message_id": _CAPTURED_MESSAGE_ID.encode("utf-8"),
    "timestamp": b"2026-09-05T06:58:03.144519+00:00",
    "source": b"node_event_emit_effect",
    "event_type": _CAPTURED_TOPIC.encode("utf-8"),
    "schema_version": b"1.0.0",
    "priority": b"normal",
    "retry_count": b"0",
    "max_retries": b"3",
}


def _captured_record(
    *,
    offset: int = 285,
    headers: dict[str, bytes] | None = None,
    topic: str = _CAPTURED_TOPIC,
) -> ModelTransportMessage:
    """The captured stability-lane record, optionally with mutated headers."""
    return ModelTransportMessage(
        topic=topic,
        partition=2,
        offset=offset,
        key=None,
        value=_CAPTURED_VALUE,
        headers=dict(_CAPTURED_HEADERS if headers is None else headers),
        ack_token=f"{topic}:2:{offset}",
    )


def _mirror(harness: Any) -> Any:
    from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
        NodeLaneMirror,
    )

    return NodeLaneMirror(**harness.kwargs)


# ---------------------------------------------------------------------------
# 1. The wire shape the hook edge actually publishes
# ---------------------------------------------------------------------------


def test_the_captured_record_is_not_a_model_event_envelope() -> None:
    """Pin the premise: the captured value is flat, with no ``payload`` key.

    If this ever fails, the hook edge changed its wire format and the rest of
    this file is testing a shape that no longer exists on the lane.
    """
    decoded = json.loads(_CAPTURED_VALUE)
    assert "payload" not in decoded
    assert "envelope_id" not in decoded
    assert decoded["hook_source"] == "user_prompt_submit"
    # The identity the mirror needs is on the wire, but only in the headers.
    assert "message_id" in _CAPTURED_HEADERS


@pytest.mark.asyncio
async def test_a_captured_stability_lane_record_is_mirrored(
    lane_mirror_harness: Any,
) -> None:
    """The live defect, as a test: 261/261 of these were rejected, 0 mirrored."""
    harness = lane_mirror_harness
    harness.source.offer(_captured_record())

    await _mirror(harness).drain_once()

    for lane, producer in harness.mirrors.items():
        assert len(producer.sent) == 1, (
            f"lane {lane} mirrored nothing -- this is the OMN-17919 defect: the "
            "hook edge publishes a flat payload with the envelope metadata in "
            "Kafka headers, and the mirror decoded it as a ModelEventEnvelope"
        )


@pytest.mark.asyncio
async def test_the_mirrored_record_is_byte_for_byte_the_source_record(
    lane_mirror_harness: Any,
) -> None:
    """The leg's stated contract is republish-unchanged; prove the bytes."""
    harness = lane_mirror_harness
    harness.source.offer(_captured_record())

    await _mirror(harness).drain_once()

    sent = harness.mirrors["dev"].sent[0]
    assert sent.topic == _CAPTURED_TOPIC
    assert sent.value == _CAPTURED_VALUE
    assert dict(sent.headers) == _CAPTURED_HEADERS


@pytest.mark.asyncio
async def test_the_source_offset_commits_after_a_captured_record_is_mirrored(
    lane_mirror_harness: Any,
) -> None:
    """A mirrored record advances the source group; a Dead group moves nothing."""
    harness = lane_mirror_harness
    harness.source.offer(_captured_record())

    await _mirror(harness).drain_once()

    assert len(harness.source.committed) == 1
    assert harness.source.nacked == []


# ---------------------------------------------------------------------------
# 2. The idempotency key is the header, and it holds across redelivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_idempotency_key_is_the_message_id_header(
    lane_mirror_harness: Any,
) -> None:
    """The key is read from the wire, not minted -- redelivery must be exact."""
    harness = lane_mirror_harness
    store = harness.kwargs["idempotency_store"]
    harness.source.offer(_captured_record())

    await _mirror(harness).drain_once()

    assert store.marked_ids() == {UUID(_CAPTURED_MESSAGE_ID)}


@pytest.mark.asyncio
async def test_redelivery_of_a_captured_record_publishes_nothing_further(
    lane_mirror_harness: Any,
) -> None:
    """At-least-once source redelivery must not duplicate on any mirror."""
    harness = lane_mirror_harness
    service = _mirror(harness)
    harness.source.offer(_captured_record())
    await service.drain_once()
    harness.source.offer(_captured_record())
    await service.drain_once()

    for lane, producer in harness.mirrors.items():
        assert len(producer.sent) == 1, lane


@pytest.mark.asyncio
async def test_two_records_with_distinct_message_ids_both_cross(
    lane_mirror_harness: Any,
) -> None:
    """Negative control on the idempotency key: it must not collapse the topic.

    A key derived from something constant across records (the topic, the value
    shape) would suppress every record after the first and look exactly like a
    working idempotent mirror.
    """
    harness = lane_mirror_harness
    second = dict(_CAPTURED_HEADERS)
    second["message_id"] = b"11111111-2222-4333-8444-555555555555"
    harness.source.offer(_captured_record())
    harness.source.offer(_captured_record(offset=286, headers=second))

    await _mirror(harness).drain_once()

    assert len(harness.mirrors["dev"].sent) == 2


# ---------------------------------------------------------------------------
# 3. A genuinely malformed record is REFUSED -- typed, counted, never silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_record_with_no_message_id_header_is_refused_and_counted(
    lane_mirror_harness: Any,
) -> None:
    """Dropping the envelope round-trip must not make unreadable records silent.

    A mirror that quietly commits past everything it cannot read is
    indistinguishable from the Dead-group state this ticket is fixing.
    """
    harness = lane_mirror_harness
    headers = dict(_CAPTURED_HEADERS)
    del headers["message_id"]
    service = _mirror(harness)
    harness.source.offer(_captured_record(headers=headers))

    await service.drain_once()

    assert harness.mirrors["dev"].sent == []
    assert service.refused_record_count == 1


@pytest.mark.asyncio
async def test_a_record_with_an_unparseable_message_id_header_is_refused(
    lane_mirror_harness: Any,
) -> None:
    """A header that is present but not a UUID is malformed, not an identity."""
    harness = lane_mirror_harness
    headers = dict(_CAPTURED_HEADERS)
    headers["message_id"] = b"not-a-uuid"
    service = _mirror(harness)
    harness.source.offer(_captured_record(headers=headers))

    await service.drain_once()

    assert harness.mirrors["dev"].sent == []
    assert service.refused_record_count == 1


def test_the_refusal_is_a_typed_error_not_a_bare_exception() -> None:
    """The refusal must be greppable and classifiable, not a bare ValueError."""
    from omnibase_infra.errors import (
        LaneMirrorRecordRefusedError,
        RuntimeHostError,
    )

    assert issubclass(LaneMirrorRecordRefusedError, RuntimeHostError)


@pytest.mark.asyncio
async def test_a_refused_record_is_committed_past_not_redelivered_forever(
    lane_mirror_harness: Any,
) -> None:
    """OMN-15748 poison-pill policy: a record that can never be read must not
    seek back to its own offset on every redelivery and wedge the leg."""
    harness = lane_mirror_harness
    headers = dict(_CAPTURED_HEADERS)
    del headers["message_id"]
    harness.source.offer(_captured_record(headers=headers))

    await _mirror(harness).drain_once()

    assert len(harness.source.committed) == 1
    assert harness.source.nacked == []


@pytest.mark.asyncio
async def test_a_refused_record_does_not_stop_the_next_valid_record(
    lane_mirror_harness: Any,
) -> None:
    """One malformed record must not be a denial of service on the whole leg."""
    harness = lane_mirror_harness
    bad = dict(_CAPTURED_HEADERS)
    del bad["message_id"]
    service = _mirror(harness)
    harness.source.offer(_captured_record(headers=bad))
    harness.source.offer(_captured_record(offset=286))

    await service.drain_once()

    assert len(harness.mirrors["dev"].sent) == 1
    assert service.refused_record_count == 1


@pytest.mark.asyncio
async def test_the_refusal_counter_starts_at_zero_and_only_counts_refusals(
    lane_mirror_harness: Any,
) -> None:
    """A counter that also ticks on healthy records cannot signal a wedge."""
    harness = lane_mirror_harness
    service = _mirror(harness)
    assert service.refused_record_count == 0
    harness.source.offer(_captured_record())

    await service.drain_once()

    assert service.refused_record_count == 0


@pytest.mark.asyncio
async def test_an_undeclared_topic_is_skipped_without_counting_as_a_refusal(
    lane_mirror_harness: Any,
) -> None:
    """AC4 still holds, and a contract skip is not a malformed record.

    Conflating the two would make the refusal counter tick on correct,
    deny-by-default behaviour and destroy its value as a wedge signal.
    """
    harness = lane_mirror_harness
    service = _mirror(harness)
    harness.source.offer(
        _captured_record(topic="onex.evt.omnibase-infra.gateway-heartbeat.v1")
    )

    await service.drain_once()

    assert harness.mirrors["dev"].sent == []
    assert service.refused_record_count == 0
