# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""runner_fleet_event.py — build the typed runner-fleet observation event (OMN-18768).

WHY THIS EXISTS
    The runner monitor (docker/runners/runner-monitor.sh) has known the state of
    every runner in the fleet on a 3-minute cadence since OMN-13109, and posted
    it to a chat channel. It emitted NO bus event (OMN-16943), so nothing
    downstream could read it: not alert triage, not a projection, not the
    dashboard. A sweep of every `onex.snapshot.projection.*` topic across the
    runtime sources returns 60+ topics and not one runner, lane, fleet or host
    topic. "What runners are running" had no producer at all.

    This module is the producer half. It is a PURE BUILDER so the wire schema is
    deterministic and unit-testable: the shell measures (it already holds the org
    runners JSON), this builds, and the shell publishes the single JSON object on
    stdout to onex.evt.infra.runner-fleet.v1 — the same split as
    scripts/disk_watermark_event.py (OMN-13008).

ONE EVENT PER CYCLE, NOT PER RUNNER
    A ~69-runner fleet is one message. Per-runner messages would put the fleet's
    cardinality on the bus every three minutes for a fact whose only consumer
    wants the whole picture, and would make a partial observation (some runners
    published, some not) indistinguishable from a partial outage.

AN OFFLINE RUNNER IS REPORTED, NEVER OMITTED
    The single most important property here. If an offline runner were dropped
    from the row set, a fleet outage would render downstream as a SMALLER,
    entirely healthy fleet — the exact false-green this whole observability epic
    exists to close. Every runner the org API returns inside this fleet's name
    prefix gets a row, whatever its status.

WHAT THIS DELIBERATELY DOES NOT INVENT
    The GitHub org runners API carries `busy` but no job identity. When the
    caller cannot resolve a job id for a busy runner, `current_job_id` is NULL.
    NULL means "this runner is executing something we could not name"; it is a
    different fact from an idle runner, and both are different from a fabricated
    id. Nothing here fills that gap with a plausible value.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "1.0.0"
EVENT_TYPE = "runner-fleet-observation"

# The label classes that partition this fleet operationally. Which class is down
# is the question a fleet panel exists to answer: a `omnibase-prod-deploy`
# outage (a single runner serving production promotion) is not an `omnibase-ci`
# outage (dozens of interchangeable runners), and a fleet-wide healthy count
# hides the first completely.
#
# `self-hosted`, `Linux`, `X64` and the like are carried in `labels` but never
# chosen as the class: they are on every runner and classify nothing.
KNOWN_LABEL_CLASSES: tuple[str, ...] = (
    "omnibase-ci",
    "omnibase-verify",
    "omnibase-deploy",
    "omnibase-prod-deploy",
    "omnibase-customer-plane",
)

UNCLASSIFIED = "unclassified"

# Runners carry their own host as a `host-<id>` label (`host-101`, `host-105`,
# `host-201` in the live pool on 2026-09-18). That is the runner's ACTUAL host
# and it is what a fleet panel needs: on that date the one offline runner in
# the org pool was `omninode-air-runner-1`, which lives on .105 — attributing
# it to whichever host happened to run the monitor would have pointed an
# operator at the wrong machine. When no such label is present the observing
# host is used, and the row records both so the two are never confused.
HOST_LABEL_PREFIX = "host-"

# A runner is exactly one of these. `busy` is deliberately a status of its own
# rather than a flag on `online`: folding it in makes "the fleet is saturated"
# indistinguishable from "the fleet is idle".
STATUS_ONLINE = "online"
STATUS_OFFLINE = "offline"
STATUS_BUSY = "busy"


def _label_names(runner: Mapping[str, Any]) -> list[str]:
    """Flatten GitHub's `[{name: ...}]` label shape to plain names."""
    labels = runner.get("labels")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return []
    names: list[str] = []
    for label in labels:
        if isinstance(label, Mapping):
            value = label.get("name")
        else:
            value = label
        if value is None:
            continue
        names.append(str(value))
    return names


def classify(labels: Sequence[str]) -> str:
    """Resolve the operational class of a runner from its labels.

    Declared order wins, so a runner carrying two fleet classes resolves the
    same way on every cycle rather than by dict iteration accident.
    """
    present = set(labels)
    for known in KNOWN_LABEL_CLASSES:
        if known in present:
            return known
    return UNCLASSIFIED


def resolve_host(labels: Sequence[str], observing_host: str) -> str:
    """Resolve a runner's own host from its labels, else the observing host."""
    for label in labels:
        if label.startswith(HOST_LABEL_PREFIX) and len(label) > len(HOST_LABEL_PREFIX):
            return label
    return observing_host


def _resolve_status(runner: Mapping[str, Any]) -> str:
    """Return the wire status.

    A busy runner is still online — it is executing work — so it counts toward
    `online_count` while reporting `busy` as its own status. The two counts are
    derived from this one answer below rather than returned alongside it, so
    they cannot disagree with it.
    """
    raw = str(runner.get("status") or "").strip().lower()
    online = raw == STATUS_ONLINE
    busy = bool(runner.get("busy"))
    if online and busy:
        return STATUS_BUSY
    if online:
        return STATUS_ONLINE
    # Anything the API does not call online is offline as far as this fleet is
    # concerned. Inventing a third "unknown" state here would let a status
    # string GitHub renames read as neither up nor down.
    return STATUS_OFFLINE


def build_event(
    *,
    runners_payload: Mapping[str, Any],
    host: str,
    name_prefix: str,
    runner_group: str,
    topic: str,
    job_by_runner: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Construct one fleet-observation event from the org runners payload.

    `runners_payload` is the verbatim body of
    `GET /orgs/{org}/actions/runners?per_page=100`.
    """
    now = now or datetime.now(UTC)
    observed_at = now.isoformat()

    if not host:
        raise ValueError(
            "host attribution is required; refusing to emit an unattributed fleet observation"
        )
    if not topic:
        raise ValueError("topic is required")

    raw_runners = runners_payload.get("runners")
    if not isinstance(raw_runners, Sequence) or isinstance(raw_runners, (str, bytes)):
        # A monitor that cannot see the fleet must fail LOUDLY. Emitting an empty
        # fleet on an unparseable payload would read downstream as "every runner
        # is gone" — a GitHub API blip rendered as a total outage.
        raise ValueError(
            "runners_payload has no 'runners' list; refusing to emit an empty fleet "
            "observation for an unreadable payload"
        )

    jobs: Mapping[str, Any] = job_by_runner or {}

    rows: list[dict[str, Any]] = []
    for runner in raw_runners:
        if not isinstance(runner, Mapping):
            raise ValueError(f"runner entry is not an object: {runner!r}")
        name = runner.get("name")
        if name is None or str(name) == "":
            raise ValueError(
                "runner entry carries no name; refusing to silently skip an "
                "unidentifiable runner"
            )
        name = str(name)
        if name_prefix and not name.startswith(name_prefix):
            # A foreign org runner is not this fleet's liveness and must not
            # dilute its counts.
            continue

        labels = _label_names(runner)
        status = _resolve_status(runner)

        current_job_id: str | None = None
        if status == STATUS_BUSY:
            resolved = jobs.get(name)
            if resolved is not None and str(resolved) != "":
                current_job_id = str(resolved)

        runner_id = runner.get("id")
        rows.append(
            {
                "runner_name": name,
                "runner_id": int(runner_id)
                if isinstance(runner_id, (int, float))
                else None,
                "label_class": classify(labels),
                "labels": labels,
                # The runner's OWN host, from its host-<id> label when it has
                # one. `observing_host` below is the machine that took the
                # observation; the two are different facts and conflating them
                # points an operator at the wrong machine.
                "host": resolve_host(labels, host),
                "observing_host": host,
                "status": status,
                "current_job_id": current_job_id,
                "observed_at": observed_at,
            }
        )

    # Deterministic order so replaying an observation reproduces the same event
    # byte for byte rather than one that merely means the same thing.
    rows.sort(key=lambda row: str(row["runner_name"]))

    online_count = sum(
        1 for row in rows if row["status"] in (STATUS_ONLINE, STATUS_BUSY)
    )
    busy_count = sum(1 for row in rows if row["status"] == STATUS_BUSY)
    offline_count = sum(1 for row in rows if row["status"] == STATUS_OFFLINE)

    class_rollup: dict[str, dict[str, Any]] = {}
    for row in rows:
        label_class = str(row["label_class"])
        entry = class_rollup.setdefault(
            label_class,
            {
                "label_class": label_class,
                "total": 0,
                "online": 0,
                "busy": 0,
                "offline": 0,
            },
        )
        entry["total"] += 1
        if row["status"] == STATUS_OFFLINE:
            entry["offline"] += 1
        else:
            entry["online"] += 1
            if row["status"] == STATUS_BUSY:
                entry["busy"] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "topic": topic,
        "host": host,
        "runner_group": runner_group,
        "name_prefix": name_prefix,
        "observed_at": observed_at,
        "runner_count": len(rows),
        "online_count": online_count,
        "busy_count": busy_count,
        "offline_count": offline_count,
        "runners": rows,
        "class_rollup": [class_rollup[key] for key in sorted(class_rollup)],
    }


def main() -> int:
    """Read the org runners JSON on stdin, write one event on stdout.

    Env is fail-fast (Operating Rule 8): a missing HOST or TOPIC raises rather
    than defaulting to a value that would attribute the observation to the wrong
    host or publish it to the wrong topic.
    """
    payload_text = sys.stdin.read()
    try:
        runners_payload = json.loads(payload_text) if payload_text.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"stdin is not valid JSON: {exc}") from exc

    raw_jobs = os.environ.get("RUNNER_JOB_MAP_JSON", "").strip()
    job_by_runner: dict[str, Any] = {}
    if raw_jobs:
        try:
            decoded = json.loads(raw_jobs)
        except json.JSONDecodeError:
            # A job-id resolution that failed is a MISSING job id, never a
            # reason to drop the whole fleet observation.
            decoded = {}
        if isinstance(decoded, dict):
            job_by_runner = decoded

    event = build_event(
        runners_payload=runners_payload,
        host=os.environ["RUNNER_FLEET_HOST"],
        name_prefix=os.environ.get("RUNNER_NAME_PREFIX", ""),
        runner_group=os.environ.get("RUNNER_GROUP", ""),
        topic=os.environ["TOPIC"],
        job_by_runner=job_by_runner,
    )
    json.dump(event, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
