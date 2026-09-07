# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17201: an acknowledgement is not proof the DESTINATION took the record.

The third revision of the same leg. OMN-17034 wired it, OMN-17919 taught it to
read the wire shape and to dial the source lane's external listener -- and on
2026-09-06T20:44:38Z the dev broker container was recreated, the dev producer's
reconnect re-resolved the bare ``redpanda`` alias onto the STABILITY network,
and for the next three hours the leg published every record back onto the lane
it was reading from. Nothing raised: ``KafkaTransport.send`` is
``send_and_wait``, so each publish really was acknowledged -- by the wrong
broker. 754 ``Lane mirror delivered`` lines were emitted while the dev lane's
``onex.evt.omniclaude.tool-executed.v1`` p0 took 4000/4000 of its records from
``node_dlq_replay_effect`` and ZERO carrying a ``message_id`` header.

Measured on .201 2026-09-07, read-only, and the reason these tests exist. Equal
1500-record windows across the six stability partitions:

    offset base 40000 (16:01:58Z, before) -> 389 identifiable records,  0 duplicate ids
    offset base 46000 (21:05:55Z, after)  -> 396 identifiable records, 82 duplicate ids
    offset base 57000 (23:46:54Z, after)  -> 398 identifiable records, 106 duplicate ids

The zero is positive-controlled by the 389 identifiable records in the same
window: the extractor works, the duplicates genuinely were not there before.

Every assertion below fails on the parent commit, and each fails for the reason
named in its docstring rather than incidentally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayLaneMirrorHealth,
    ModelGatewayPublishReceipt,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_lane_mirror_health import (
    evaluate_lane_mirror_health,
    lane_mirror_lag,
    record_accepted_then_failed,
    record_confirmed_delivery,
    record_consumed,
    record_loop_detected,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
    NodeLaneMirror,
)

pytestmark = pytest.mark.unit

_TOPIC = "onex.evt.omniclaude.tool-executed.v1"


class _ReceiptProducer:
    """A mirror producer that reports where the destination put the record.

    ``destination`` is what the broker on the OTHER end of this producer
    assigns. Setting it to a coordinate the source consumer will subsequently
    deliver is precisely the .201 condition: the destination broker IS the
    source broker.
    """

    def __init__(self, *, destination: tuple[str, int, int] | None = None) -> None:
        self.sent: list[tuple[str, bytes | None, bytes]] = []
        self.destination = destination
        self.fail_next = False

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> ModelGatewayPublishReceipt | None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("destination broker refused the publish")
        self.sent.append((topic, key, value))
        if self.destination is None:
            return None
        return ModelGatewayPublishReceipt(
            topic=self.destination[0],
            partition=self.destination[1],
            offset=self.destination[2],
        )


# ---------------------------------------------------------------------------
# 1. The publish boundary reports WHERE, not only THAT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_confirmed_delivery_records_the_source_offset_per_lane(
    lane_mirror_harness: Any,
) -> None:
    """RED on parent: no counter exists that a lane confirmed a source offset.

    ``mirrored_total`` alone cannot distinguish "this lane is behind" from
    "this lane has never taken a single record", and the second is what was
    live for three hours.
    """
    harness = lane_mirror_harness
    harness.kwargs["mirror_producers"] = {"dev": _ReceiptProducer()}
    service = NodeLaneMirror(**harness.kwargs)
    harness.source.offer(harness.record(envelope_id="e-1", offset=41))

    await service.drain_once()

    assert service.health.confirmed_delivered_offsets == {f"dev|{_TOPIC}:0": 41}
    assert service.health.consumed_source_offsets == {f"{_TOPIC}:0": 41}
    assert lane_mirror_lag(service.health) == {f"dev|{_TOPIC}:0": 0}


@pytest.mark.asyncio
async def test_a_record_accepted_then_failed_is_counted_not_only_logged(
    lane_mirror_harness: Any,
) -> None:
    """RED on parent: ``refused_record_count`` structurally cannot cover this.

    A refusal is decided BEFORE any publish is attempted, so a record that
    passed identity and then failed at the broker was counted by nothing.
    """
    harness = lane_mirror_harness
    producer = _ReceiptProducer()
    producer.fail_next = True
    harness.kwargs["mirror_producers"] = {"dev": producer}
    service = NodeLaneMirror(**harness.kwargs)
    harness.source.offer(harness.record(envelope_id="e-1", offset=7))

    await service.drain_once()

    assert service.health.accepted_then_failed_total == 1
    assert service.health.refused_total == 0
    assert harness.source.committed == []
    assert harness.source.nacked != []


# ---------------------------------------------------------------------------
# 2. The loop -- the condition an acknowledgement cannot rule out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_record_read_back_at_a_coordinate_this_process_wrote_is_a_loop(
    lane_mirror_harness: Any,
) -> None:
    """RED on parent: the echo was suppressed as a duplicate and committed.

    The idempotency marker makes redelivery safe and makes an echo INVISIBLE.
    Suppression is correct for a redelivery and is a hard error for a record
    this process itself published, and the two are told apart by whether the
    coordinate is one this process wrote to a destination.
    """
    harness = lane_mirror_harness
    # The destination hands back (topic, 0, 991) -- and the source lane then
    # delivers exactly that coordinate. Only one broker can do both.
    harness.kwargs["mirror_producers"] = {
        "dev": _ReceiptProducer(destination=(_TOPIC, 0, 991))
    }
    service = NodeLaneMirror(**harness.kwargs)
    harness.source.offer(harness.record(envelope_id="e-1", offset=990))
    await service.drain_once()
    harness.source.offer(harness.record(envelope_id="e-1", offset=991))

    await service.drain_once()

    assert service.health.loop_detected_total == 1
    assert service.health.last_loop_detected_lane == "dev"
    assert "dev" in service.refused_lanes


@pytest.mark.asyncio
async def test_a_refused_lane_publishes_nothing_and_commits_nothing(
    lane_mirror_harness: Any,
) -> None:
    """RED on parent: ``refuse_lane`` does not exist, so nothing can hold the leg.

    Holding is the fail-closed half. Committing past a record that never
    reached the destination is silent loss; nacking retains it on the source
    for redelivery once the lane is repaired.
    """
    harness = lane_mirror_harness
    producer = _ReceiptProducer()
    harness.kwargs["mirror_producers"] = {"dev": producer}
    service = NodeLaneMirror(**harness.kwargs)
    service.refuse_lane("dev", "destination broker is the source broker")
    harness.source.offer(harness.record(envelope_id="e-1", offset=12))

    await service.drain_once()

    assert producer.sent == []
    assert harness.source.committed == []
    assert harness.source.nacked != []
    assert service.health.accepted_then_failed_total == 1


@pytest.mark.asyncio
async def test_clearing_a_refusal_lets_the_held_records_cross(
    lane_mirror_harness: Any,
) -> None:
    """A refusal must be recoverable, or a transient re-home wedges the leg."""
    harness = lane_mirror_harness
    producer = _ReceiptProducer()
    harness.kwargs["mirror_producers"] = {"dev": producer}
    service = NodeLaneMirror(**harness.kwargs)
    service.refuse_lane("dev", "destination broker is the source broker")
    harness.source.offer(harness.record(envelope_id="e-1", offset=12))
    await service.drain_once()

    service.clear_lane_refusal("dev")
    harness.source.offer(harness.record(envelope_id="e-1", offset=12))
    await service.drain_once()

    assert len(producer.sent) == 1
    assert harness.source.committed != []


def test_refusing_an_undeclared_lane_is_refused_not_silently_ignored(
    lane_mirror_harness: Any,
) -> None:
    """A typo'd lane name must not read as a successfully-refused lane."""
    service = NodeLaneMirror(**lane_mirror_harness.kwargs)
    with pytest.raises(ValueError, match="not a declared mirror lane"):
        service.refuse_lane("stability-test", "typo")


# ---------------------------------------------------------------------------
# 3. The healthcheck verdict names the lane
# ---------------------------------------------------------------------------


def test_a_detected_loop_fails_the_healthcheck_and_names_the_lane() -> None:
    """RED on parent: no lane-mirror verdict exists at all.

    Both canary legs produce to the canary topic and read it back on the SAME
    lane, so neither can observe a mirror acknowledging into the wrong broker.
    """
    now = datetime.now(UTC)
    state = record_loop_detected(ModelGatewayLaneMirrorHealth(), lane="dev", now=now)

    passed, detail = evaluate_lane_mirror_health(state, now=now)

    assert passed is False
    assert "dev" in detail


def test_a_leg_that_never_looped_and_never_failed_passes() -> None:
    """Absence is not failure: a two-leg forwarder writes no counters."""
    passed, detail = evaluate_lane_mirror_health(None, now=datetime.now(UTC))
    assert passed is True
    assert "no lane mirror" in detail


def test_a_total_stall_inside_the_window_fails_naming_the_lane() -> None:
    """Accepted-then-failed with nothing confirmed is the stall shape."""
    now = datetime.now(UTC)
    state = record_accepted_then_failed(
        ModelGatewayLaneMirrorHealth(), lane="dev", now=now
    )

    passed, detail = evaluate_lane_mirror_health(state, now=now)

    assert passed is False
    assert "dev" in detail


def test_an_old_failure_with_recent_confirmations_reads_degraded_not_dead() -> None:
    """A mixed leg is degraded; flapping the verdict per record helps nobody."""
    now = datetime.now(UTC)
    state = record_accepted_then_failed(
        ModelGatewayLaneMirrorHealth(), lane="dev", now=now - timedelta(seconds=600)
    )

    passed, detail = evaluate_lane_mirror_health(state, now=now)

    assert passed is True
    assert "outside the" in detail


def test_lag_reports_the_full_consumed_offset_for_a_lane_that_never_confirmed() -> None:
    """ "Behind by 3" and "has never taken one" must not read the same."""
    state = record_consumed(
        ModelGatewayLaneMirrorHealth(), topic=_TOPIC, partition=0, offset=55985
    )
    state = record_confirmed_delivery(
        state,
        lane="dev",
        topic=_TOPIC,
        partition=0,
        offset=0,
        now=datetime.now(UTC),
    )
    assert lane_mirror_lag(state) == {f"dev|{_TOPIC}:0": 55985}


# ---------------------------------------------------------------------------
# 4. The health file both sides read is the same file
# ---------------------------------------------------------------------------


def test_the_forwarder_and_the_healthcheck_name_the_same_health_files() -> None:
    """A counter written where nothing reads it is not an observability surface.

    RED on parent for the lane-mirror path (the flag did not exist) and a real
    pre-existing gap for the egress path: ``--egress-health-file`` was parsed by
    ``_build_parser``, named in compose, and read by the healthcheck -- and
    ``_async_main`` never passed it to ``run_gateway_forwarder``, so the live
    process wrote no egress counters at all.
    """
    import inspect
    from pathlib import Path

    import yaml

    from omnibase_infra.runtime import gateway_forwarder

    compose_path = (
        Path(__file__).resolve().parents[4] / "docker" / "docker-compose.gateway.yml"
    )
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    command = compose["services"]["gateway-forwarder"]["command"]
    healthcheck = compose["services"]["gateway-forwarder"]["healthcheck"]["test"]

    for flag in ("--egress-health-file", "--lane-mirror-health-file"):
        assert flag in command, f"the forwarder is not given {flag}"
        assert flag in healthcheck, f"the healthcheck does not read {flag}"
        assert (
            command[command.index(flag) + 1] == healthcheck[healthcheck.index(flag) + 1]
        ), f"{flag} differs between the writer and the reader"

    source = inspect.getsource(gateway_forwarder._async_main)
    assert "egress_health_path=args.egress_health_file" in source
    assert "lane_mirror_health_path=args.lane_mirror_health_file" in source
