# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane-mirror delivery accounting and its health verdict (OMN-17201).

Same two-halves-one-contract shape as ``service_gateway_egress_health``, and
for the same reason: the writer runs in the forwarder process and the reader
runs in the container healthcheck, so the file format has exactly one home.

What this adds over the egress counters is a VERIFICATION relation rather than
a tally. The egress leg can say "N delivered" because for the trust-boundary
legs a broker acknowledgement is the whole claim. For the lane mirror it is
not: the interesting failure is a record that was genuinely acknowledged by a
broker that is not the destination lane's. Two things make that reportable
here -- a per-lane confirmed-offset map compared against the consumed-offset
map (the lag), and an explicit loop counter for the case where the destination
turns out to BE the source.

Free functions over a stateful class for the same canon reason the egress
module gives: the counters belong to ``NodeLaneMirror``, and a second
lifecycle-owning object beside it is the shape the OMN-14350 ratchet exists to
keep out.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayLaneMirrorHealth,
)

logger = logging.getLogger(__name__)

DEFAULT_LANE_MIRROR_HEALTH_PATH = "/tmp/gateway-lane-mirror-health.json"  # noqa: S108 -- container-local scratch state, same convention as the egress counters
"""Shared default so the writer and the healthcheck cannot drift apart.

Container-local and NOT the ``/app/data`` named volume, deliberately: these
counters describe the CURRENT process, and a loop detected by a previous
process must not report a freshly repointed one unhealthy.
"""

LANE_MIRROR_STALL_WINDOW_SECONDS = 120
"""How long a lane may go accepting-and-failing before the container is unhealthy.

Same width and same reasoning as ``EGRESS_DENIAL_WINDOW_SECONDS``: wider than
one canary cadence so a leg carrying a MIX of confirmed and failed records
reads as degraded rather than flapping, and narrow enough that a total stall is
caught within two healthcheck ticks.
"""


def _source_key(topic: str, partition: int) -> str:
    return f"{topic}:{partition}"


def _destination_key(lane: str, topic: str, partition: int) -> str:
    return f"{lane}|{topic}:{partition}"


def record_consumed(
    state: ModelGatewayLaneMirrorHealth,
    *,
    topic: str,
    partition: int,
    offset: int,
) -> ModelGatewayLaneMirrorHealth:
    """Return ``state`` with one source record accounted as consumed.

    Monotonic per (topic, partition): a redelivery after a nack rewinds the
    broker fetch position but must not rewind the watermark this leg reports,
    or the lag would read as negative during recovery.
    """
    key = _source_key(topic, partition)
    consumed = dict(state.consumed_source_offsets)
    consumed[key] = max(consumed.get(key, offset), offset)
    return state.model_copy(update={"consumed_source_offsets": consumed})


def record_confirmed_delivery(
    state: ModelGatewayLaneMirrorHealth,
    *,
    lane: str,
    topic: str,
    partition: int,
    offset: int,
    now: datetime,
) -> ModelGatewayLaneMirrorHealth:
    """Return ``state`` with one destination-broker acknowledgement accounted.

    Called only AFTER the publish coroutine returns, which for the Kafka
    transport means ``send_and_wait`` resolved -- a real broker ack, not a
    queued send.
    """
    key = _destination_key(lane, topic, partition)
    confirmed = dict(state.confirmed_delivered_offsets)
    confirmed[key] = max(confirmed.get(key, offset), offset)
    return state.model_copy(
        update={
            "confirmed_delivered_offsets": confirmed,
            "last_mirrored_at": now,
        }
    )


def record_mirrored(
    state: ModelGatewayLaneMirrorHealth,
    *,
    now: datetime,
) -> ModelGatewayLaneMirrorHealth:
    """Return ``state`` with one record accounted as crossing to every lane."""
    return state.model_copy(
        update={
            "mirrored_total": state.mirrored_total + 1,
            "last_mirrored_at": now,
        }
    )


def record_refusal(
    state: ModelGatewayLaneMirrorHealth,
) -> ModelGatewayLaneMirrorHealth:
    """Return ``state`` with one unidentifiable record accounted (OMN-17919)."""
    return state.model_copy(update={"refused_total": state.refused_total + 1})


def record_accepted_then_failed(
    state: ModelGatewayLaneMirrorHealth,
    *,
    lane: str | None,
    now: datetime,
) -> ModelGatewayLaneMirrorHealth:
    """Return ``state`` with one accepted-then-failed record accounted.

    ``refused_total`` structurally cannot cover this class: a refusal is
    decided before any publish is attempted, so a record that passed identity
    and then failed at the broker was previously counted by nothing at all.
    """
    return state.model_copy(
        update={
            "accepted_then_failed_total": state.accepted_then_failed_total + 1,
            "last_failure_at": now,
            "last_failure_lane": lane,
        }
    )


def record_loop_detected(
    state: ModelGatewayLaneMirrorHealth,
    *,
    lane: str | None,
    now: datetime,
) -> ModelGatewayLaneMirrorHealth:
    """Return ``state`` with one mirror-into-own-source echo accounted."""
    return state.model_copy(
        update={
            "loop_detected_total": state.loop_detected_total + 1,
            "last_loop_detected_at": now,
            "last_loop_detected_lane": lane,
        }
    )


def lane_mirror_lag(state: ModelGatewayLaneMirrorHealth) -> dict[str, int]:
    """Return ``{"<lane>|<topic>:<partition>": consumed - confirmed}``.

    A destination key that has never confirmed anything for a source partition
    the leg HAS consumed reports the full consumed offset as its lag, which is
    the reading that distinguishes "this lane is behind" from "this lane has
    never taken a single record" -- the actual .201 condition.
    """
    lag: dict[str, int] = {}
    for destination_key, confirmed_offset in state.confirmed_delivered_offsets.items():
        _lane, _, source_key = destination_key.partition("|")
        consumed_offset = state.consumed_source_offsets.get(source_key)
        if consumed_offset is None:
            continue
        lag[destination_key] = consumed_offset - confirmed_offset
    return lag


def publish_lane_mirror_health(
    state: ModelGatewayLaneMirrorHealth,
    state_path: Path | None,
) -> None:
    """Write the counters where the container healthcheck can read them.

    Best-effort and never raises, for the reason the egress writer gives: an
    observability write must not turn a record that actually crossed into an
    exception. A ``None`` path counts in memory and publishes nothing.
    """
    if state_path is None:
        return
    temporary = state_path.with_name(f"{state_path.name}.tmp")
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(state.model_dump_json(), encoding="utf-8")
        temporary.replace(state_path)
    except OSError:
        logger.warning(
            "Gateway lane-mirror health state write failed path=%s",
            state_path,
            exc_info=True,
        )


def load_lane_mirror_health(
    state_path: Path,
) -> ModelGatewayLaneMirrorHealth | None:
    """Read the counters the forwarder published, or ``None`` if unreadable.

    ``None`` means "no verdict available", never healthy and never unhealthy.
    A missing file is the normal state of a forwarder deployed without a lane
    mirror at all, which is a valid two-leg deployment.
    """
    try:
        raw = state_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ModelGatewayLaneMirrorHealth.model_validate_json(raw)
    except ValueError:
        logger.warning(
            "Gateway lane-mirror health state is unparseable path=%s", state_path
        )
        return None


def evaluate_lane_mirror_health(
    state: ModelGatewayLaneMirrorHealth | None,
    *,
    now: datetime,
    window_seconds: int = LANE_MIRROR_STALL_WINDOW_SECONDS,
) -> tuple[bool, str]:
    """Return ``(passed, detail)`` for the lane-mirror leg of the healthcheck.

    Fails on exactly two conditions, both of which mean the leg's own
    ``Lane mirror delivered`` line is not evidence any more:

    1. a LOOP was detected -- a destination is the source lane, so every
       "delivered" claim this process made is about the wrong broker. This is
       sticky for the life of the process on purpose: it is a deployment
       misconfiguration, it does not self-heal, and a leg that reported a loop
       and then went quiet must not read as recovered;
    2. records were accepted and then failed inside the window with nothing
       confirmed inside that same window -- the total-stall shape.

    Absence passes. ``state is None`` is a forwarder with no lane mirror
    configured, and a mirror that has not moved anything has not proven
    anything wrong.
    """
    if state is None:
        return True, "lane-mirror leg: no lane mirror configured or nothing published"
    if state.loop_detected_total > 0:
        return (
            False,
            f"lane-mirror leg: destination lane "
            f"{state.last_loop_detected_lane} IS the source lane -- "
            f"{state.loop_detected_total} record(s) this process mirrored came "
            f"back on the source at a later offset. Every "
            f"'Lane mirror delivered' line from this process names a broker "
            f"that is not the destination. Repoint that lane's "
            f"bootstrap_servers at an endpoint whose ADVERTISED listener is "
            f"unique across the lane networks.",
        )
    if state.last_failure_at is None:
        return (
            True,
            f"lane-mirror leg: no accepted-then-failed record "
            f"(mirrored={state.mirrored_total} refused={state.refused_total})",
        )
    failure_age = (now - state.last_failure_at).total_seconds()
    if failure_age >= window_seconds:
        return (
            True,
            f"lane-mirror leg: last publish failure was {failure_age:.0f}s ago, "
            f"outside the {window_seconds}s window "
            f"(mirrored={state.mirrored_total} "
            f"accepted_then_failed={state.accepted_then_failed_total})",
        )
    confirmed_recently = (
        state.last_mirrored_at is not None
        and (now - state.last_mirrored_at).total_seconds() < window_seconds
    )
    if confirmed_recently:
        return (
            True,
            f"lane-mirror leg: degraded but crossing -- "
            f"{state.accepted_then_failed_total} record(s) accepted then failed, "
            f"last on lane {state.last_failure_lane}; records are still being "
            f"confirmed (mirrored={state.mirrored_total}, "
            f"lag={lane_mirror_lag(state)})",
        )
    return (
        False,
        f"lane-mirror leg: nothing has been confirmed by a destination broker "
        f"in the last {window_seconds}s and "
        f"{state.accepted_then_failed_total} record(s) were accepted then "
        f"failed. Last failure {failure_age:.0f}s ago on lane "
        f"{state.last_failure_lane}. Per-lane lag {lane_mirror_lag(state)}.",
    )


__all__ = [
    "DEFAULT_LANE_MIRROR_HEALTH_PATH",
    "LANE_MIRROR_STALL_WINDOW_SECONDS",
    "evaluate_lane_mirror_health",
    "lane_mirror_lag",
    "load_lane_mirror_health",
    "publish_lane_mirror_health",
    "record_accepted_then_failed",
    "record_confirmed_delivery",
    "record_consumed",
    "record_loop_detected",
    "record_mirrored",
    "record_refusal",
]
