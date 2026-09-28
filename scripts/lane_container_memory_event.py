# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""lane_container_memory_event.py — build the lane container memory event (OMN-19959).

WHY THIS EXISTS
    Every OOM kill on the .202 lab host on 2026-09-28 happened inside a
    container cgroup, and nothing read the counters that recorded it. Two lane
    consumers ran pinned at their 256 MiB limit (cgroup ``memory.events``
    ``max 1053`` and ``max 565``) with no signal anywhere. The lane census
    already inventories every lab host's containers once an hour; this module
    turns the memory counters it now reads into one typed event per host per
    pass, published to ``onex.evt.omnibase-infra.lane-container-memory.v1``.

ONE CONCEPT PER TOPIC
    The census drift topic stays about declared versus observed lane SHAPE. A
    memory observation is not drift, so it gets a topic of its own rather than
    a new ``kind`` on the drift topic.

WHAT A PEAK MEANS HERE
    cgroup v2 ``memory.peak`` is the high-water mark since the cgroup was
    created, which is the container's start. It is NOT a per-pass peak. Each
    record therefore carries ``peak_window_start`` (the container's start) and
    ``peak_window_end`` (this pass's read time). A peak covers a CI job when the
    job started after ``peak_window_start`` and completed before
    ``peak_window_end``.

WHAT THE CI JOBS ARE FOR
    ``ci_runs`` lists every GitHub Actions job a runner container on this host
    ran inside ``[window_start, window_end]``, read from the runner's own
    per-job worker log (``_diag/Worker_<UTC start>-utc.log``). A lane peak can
    then be set beside the CI burst that ran on the same host at the same time.

PURE BUILDER
    The collector (``lane_census_inventory.py --memory-out``) reads the host;
    this module performs no I/O in its functions and turns one observation plus
    the previous pass's state into the event, the next state and the alert
    lines. The shell (``lane-census-check.sh --memory``) publishes the event and
    commits the next state only after the publish succeeded, so an unpublished
    window is folded into the next one instead of being lost.

KEYS
    ``record_key = sha256(host_boot_id | container_id | window_end)`` is the
    projection key: a replayed event rewrites the same row.
    ``alert_key = sha256(host_boot_id | container_id | oom_kill_total | max_total)``
    is the alert dedup key: the same counter totals never alert twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOPIC = "onex.evt.omnibase-infra.lane-container-memory.v1"
SCHEMA_VERSION = "1.0.0"
EVENT_TYPE = "lane-container-memory-observation"

#: Envelope keys and the JSON type of each value. Task 5b's wire model reads
#: exactly this set; a change here is a schema version bump.
ENVELOPE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "schema_version": str,
    "event_type": str,
    "host": str,
    "host_boot_id": str,
    "window_start": str,
    "window_end": str,
    "ci_runs": list,
    "records": list,
}

#: One record per lane container.
RECORD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "lane": str,
    "container_id": str,
    "container_name": str,
    "container_started_at": str,
    "limit_bytes": (int, type(None)),
    "peak_bytes": int,
    "peak_window_start": str,
    "peak_window_end": str,
    "max_total": int,
    "max_delta": int,
    "oom_kill_total": int,
    "oom_kill_delta": int,
    "record_key": str,
    "alert_key": str,
}

#: One entry per CI job that overlapped the window.
CI_RUN_FIELDS: dict[str, type | tuple[type, ...]] = {
    "repo": str,
    "run_id": str,
    "runner_name": str,
    "job_started_at": str,
    "job_completed_at": (str, type(None)),
}

# ``Worker_20260928-185104-utc.log``: the runner names each worker log after the
# UTC second the job's worker process started.
_WORKER_LOG_NAME = re.compile(r"^Worker_(\d{8})-(\d{6})-utc\.log$")
# ``[2026-09-28 18:51:33Z INFO Worker] Job completed.``
_LOG_LINE_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})Z ")
_JOB_COMPLETED = "INFO Worker] Job completed."
# The job's GitHub context is serialized as ``"k": "<name>"`` followed on the
# next line by ``"v": "<value>"``. Only string values are read; the first
# occurrence is the job's own context.
_CONTEXT_KEY = re.compile(r'^\s*"k":\s*"(repository|run_id)",?\s*$')
_CONTEXT_STR_VALUE = re.compile(r'^\s*"v":\s*"([^"]*)"')


class MemoryObservationError(ValueError):
    """Raised when an observation cannot be turned into a truthful event."""


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def parse_ts(value: str) -> datetime:
    """Parse an RFC 3339 timestamp, including Docker's nanosecond form.

    ``datetime.fromisoformat`` accepts at most six fractional digits, and
    Docker writes nine (``2026-09-28T12:11:26.55947631Z``), so the fraction is
    truncated to microseconds before parsing. A naive value is refused.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    match = re.match(r"^(.*?\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{2}:\d{2})$", text)
    if not match:
        raise MemoryObservationError(f"not an RFC 3339 timestamp: {value!r}")
    base, frac, offset = match.groups()
    frac = (frac or "")[:7]
    parsed = datetime.fromisoformat(f"{base}{frac}{offset}")
    return parsed.astimezone(UTC)


def format_ts(value: datetime) -> str:
    """One wire form for every timestamp: UTC, microseconds, ``Z``."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# cgroup v2 files
# ---------------------------------------------------------------------------


def parse_memory_events(text: str) -> dict[str, int]:
    """Parse cgroup v2 ``memory.events``. ``max`` and ``oom_kill`` are required."""
    counters: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            counters[parts[0]] = int(parts[1])
        except ValueError as exc:
            raise MemoryObservationError(
                f"memory.events line is not '<name> <int>': {line!r}"
            ) from exc
    for required in ("max", "oom_kill"):
        if required not in counters:
            raise MemoryObservationError(
                f"memory.events carries no {required!r} counter: {text!r}"
            )
    return counters


def parse_memory_max(text: str) -> int | None:
    """Parse ``memory.max``: bytes, or ``None`` when it reads ``max`` (no limit)."""
    value = text.strip()
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise MemoryObservationError(
            f"memory.max is not an int or 'max': {text!r}"
        ) from exc


def parse_memory_peak(text: str) -> int:
    """Parse ``memory.peak`` (bytes)."""
    try:
        return int(text.strip())
    except ValueError as exc:
        raise MemoryObservationError(f"memory.peak is not an int: {text!r}") from exc


# ---------------------------------------------------------------------------
# Runner worker logs
# ---------------------------------------------------------------------------


def worker_log_started_at(log_name: str) -> datetime | None:
    """The job start encoded in a worker log's file name, or ``None`` if not one."""
    match = _WORKER_LOG_NAME.match(log_name)
    if not match:
        return None
    day, clock = match.groups()
    return datetime.strptime(f"{day}{clock}", "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def parse_worker_log(
    text: str, *, runner_name: str, log_name: str
) -> dict[str, str | None] | None:
    """Turn one runner worker log into a CI run entry.

    Returns ``None`` only for a log that is still starting: no job context yet
    and no ``Job completed.`` line. It will be read again on the next pass, since
    an unfinished job overlaps every later window. A COMPLETED log with no job
    context is refused: it ran a job this module cannot name, and naming it
    wrong or dropping it would both be false.
    """
    started = worker_log_started_at(log_name)
    if started is None:
        raise MemoryObservationError(f"not a runner worker log name: {log_name!r}")

    context: dict[str, str] = {}
    completed: datetime | None = None
    pending_key: str | None = None
    for line in text.splitlines():
        if pending_key is not None:
            value = _CONTEXT_STR_VALUE.match(line)
            if value and pending_key not in context:
                context[pending_key] = value.group(1)
            pending_key = None
            continue
        key = _CONTEXT_KEY.match(line)
        if key:
            pending_key = key.group(1)
            continue
        if _JOB_COMPLETED in line:
            stamp = _LOG_LINE_TS.match(line)
            if stamp:
                completed = datetime.strptime(
                    f"{stamp.group(1)} {stamp.group(2)}", "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=UTC)

    if "repository" not in context or "run_id" not in context:
        if completed is None:
            return None
        raise MemoryObservationError(
            f"worker log {runner_name}:{log_name} completed a job but names no "
            f"repository and run_id (found {sorted(context)})"
        )
    return {
        "repo": context["repository"],
        "run_id": context["run_id"],
        "runner_name": runner_name,
        "job_started_at": format_ts(started),
        "job_completed_at": format_ts(completed) if completed else None,
    }


def select_ci_runs(
    runs: Sequence[Mapping[str, str | None]],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, str | None]]:
    """Keep the runs that overlap ``[window_start, window_end]``, in a stable order."""
    kept: list[dict[str, str | None]] = []
    for run in runs:
        started = parse_ts(str(run["job_started_at"]))
        completed_raw = run.get("job_completed_at")
        completed = parse_ts(str(completed_raw)) if completed_raw else None
        if started > window_end:
            continue
        if completed is not None and completed < window_start:
            continue
        kept.append(dict(run))
    return sorted(
        kept,
        key=lambda r: (
            str(r["job_started_at"]),
            str(r["runner_name"]),
            str(r["repo"]),
            str(r["run_id"]),
        ),
    )


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _sha256(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def record_key(host_boot_id: str, container_id: str, window_end: str) -> str:
    return _sha256(host_boot_id, container_id, window_end)


def alert_key(
    host_boot_id: str, container_id: str, oom_kill_total: int, max_total: int
) -> str:
    return _sha256(host_boot_id, container_id, oom_kill_total, max_total)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise MemoryObservationError(f"{where} is missing {key!r}")
    return mapping[key]


def _delta(
    total: int,
    previous_total: int | None,
    *,
    started_at: datetime,
    window_start: datetime,
) -> int:
    """Counter rise inside this window.

    A container seen last pass: the difference. A counter lower than last pass
    means the cgroup was recreated under the same id, so all of it is new. A
    container first seen now that STARTED inside the window: everything it
    counted happened in the window. One first seen now that started BEFORE the
    window: its counts cannot be placed in this window, so it is a baseline (0).
    """
    if previous_total is not None:
        return total - previous_total if total >= previous_total else total
    return total if started_at >= window_start else 0


def build_event(
    *,
    host: str,
    observation: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Build ``(event, next_state, alerts)`` from one observation.

    ``observation`` is the collector's document: ``host_boot_id``,
    ``boot_time``, ``read_at``, ``containers`` (each with ``container_id``,
    ``container_name``, ``lane``, ``started_at``, ``memory_max``,
    ``memory_peak``, ``memory_events`` as the raw file text) and
    ``worker_logs`` (each with ``runner_name``, ``log_name``, ``text``).

    ``previous_state`` is the last published pass's ``next_state``, or ``None``.
    """
    boot_id = str(_require(observation, "host_boot_id", "observation"))
    boot_time = parse_ts(str(_require(observation, "boot_time", "observation")))
    window_end_dt = parse_ts(str(_require(observation, "read_at", "observation")))

    same_boot = bool(previous_state) and (
        previous_state is not None and previous_state.get("host_boot_id") == boot_id
    )
    if same_boot and previous_state is not None:
        window_start_dt = parse_ts(str(previous_state["window_end"]))
        previous_containers: Mapping[str, Any] = previous_state.get("containers") or {}
    else:
        window_start_dt = boot_time
        previous_containers = {}
    if window_start_dt > window_end_dt:
        raise MemoryObservationError(
            f"window_start {format_ts(window_start_dt)} is after window_end "
            f"{format_ts(window_end_dt)}"
        )
    window_start = format_ts(window_start_dt)
    window_end = format_ts(window_end_dt)

    records: list[dict[str, Any]] = []
    next_containers: dict[str, dict[str, int]] = {}
    alerts: list[str] = []
    for raw in _require(observation, "containers", "observation"):
        where = f"container {raw.get('container_name') or raw.get('container_id')!r}"
        cid = str(_require(raw, "container_id", where))
        name = str(_require(raw, "container_name", where))
        lane = str(_require(raw, "lane", where))
        started_dt = parse_ts(str(_require(raw, "started_at", where)))
        started = format_ts(started_dt)
        limit = parse_memory_max(str(_require(raw, "memory_max", where)))
        peak = parse_memory_peak(str(_require(raw, "memory_peak", where)))
        events = parse_memory_events(str(_require(raw, "memory_events", where)))
        max_total = events["max"]
        oom_total = events["oom_kill"]

        previous = previous_containers.get(cid) or {}
        prev_max = previous.get("max_total")
        prev_oom = previous.get("oom_kill_total")
        prev_max_delta = int(previous.get("max_delta") or 0)
        max_delta = _delta(
            max_total,
            int(prev_max) if prev_max is not None else None,
            started_at=started_dt,
            window_start=window_start_dt,
        )
        oom_delta = _delta(
            oom_total,
            int(prev_oom) if prev_oom is not None else None,
            started_at=started_dt,
            window_start=window_start_dt,
        )

        records.append(
            {
                "lane": lane,
                "container_id": cid,
                "container_name": name,
                "container_started_at": started,
                "limit_bytes": limit,
                "peak_bytes": peak,
                "peak_window_start": started,
                "peak_window_end": window_end,
                "max_total": max_total,
                "max_delta": max_delta,
                "oom_kill_total": oom_total,
                "oom_kill_delta": oom_delta,
                "record_key": record_key(boot_id, cid, window_end),
                "alert_key": alert_key(boot_id, cid, oom_total, max_total),
            }
        )
        next_containers[cid] = {
            "max_total": max_total,
            "oom_kill_total": oom_total,
            "max_delta": max_delta,
        }

        limit_text = str(limit) if limit is not None else "max"
        if oom_delta > 0:
            alerts.append(
                f"OOM_KILL lane={lane} container={name} delta={oom_delta} "
                f"total={oom_total} peak/limit={peak}/{limit_text}"
            )
        if max_delta > 0 and prev_max_delta > 0:
            alerts.append(
                f"LIMIT_HIT lane={lane} container={name} max_delta={max_delta} "
                f"previous_max_delta={prev_max_delta} peak/limit={peak}/{limit_text}"
            )

    runs: list[dict[str, str | None]] = []
    for log in observation.get("worker_logs") or []:
        parsed = parse_worker_log(
            str(_require(log, "text", "worker log")),
            runner_name=str(_require(log, "runner_name", "worker log")),
            log_name=str(_require(log, "log_name", "worker log")),
        )
        if parsed is not None:
            runs.append(parsed)

    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "host": host,
        "host_boot_id": boot_id,
        "window_start": window_start,
        "window_end": window_end,
        "ci_runs": select_ci_runs(
            runs, window_start=window_start_dt, window_end=window_end_dt
        ),
        "records": sorted(records, key=lambda r: (r["lane"], r["container_name"])),
    }
    validate_event(event)
    next_state = {
        "host": host,
        "host_boot_id": boot_id,
        "window_end": window_end,
        "containers": dict(sorted(next_containers.items())),
    }
    return event, next_state, sorted(alerts)


def _check_fields(
    doc: Mapping[str, Any],
    fields: Mapping[str, type | tuple[type, ...]],
    where: str,
) -> None:
    missing = sorted(set(fields) - set(doc))
    extra = sorted(set(doc) - set(fields))
    if missing or extra:
        raise MemoryObservationError(f"{where}: missing {missing}, unexpected {extra}")
    for key, expected in fields.items():
        value = doc[key]
        # bool is an int subclass; a counter must never be a bool.
        if isinstance(value, bool) or not isinstance(value, expected):
            raise MemoryObservationError(
                f"{where}: {key!r} is {type(value).__name__}, expected {expected}"
            )


def validate_event(event: Mapping[str, Any]) -> None:
    """Refuse an event whose shape differs from schema 1.0.0."""
    _check_fields(event, ENVELOPE_FIELDS, "event")
    if event["schema_version"] != SCHEMA_VERSION:
        raise MemoryObservationError(f"schema_version {event['schema_version']!r}")
    if event["event_type"] != EVENT_TYPE:
        raise MemoryObservationError(f"event_type {event['event_type']!r}")
    for index, record in enumerate(event["records"]):
        _check_fields(record, RECORD_FIELDS, f"records[{index}]")
    for index, run in enumerate(event["ci_runs"]):
        _check_fields(run, CI_RUN_FIELDS, f"ci_runs[{index}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument(
        "--state",
        required=True,
        type=Path,
        help="the previous pass's state file; absent means no previous pass",
    )
    parser.add_argument("--event-out", required=True, type=Path)
    parser.add_argument("--state-out", required=True, type=Path)
    parser.add_argument("--alerts-out", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        observation = _read_json(args.observation)
        previous = _read_json(args.state) if args.state.exists() else None
        event, next_state, alerts = build_event(
            host=args.host, observation=observation, previous_state=previous
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"lane container memory event: {exc}", file=sys.stderr)
        return 2

    args.event_out.parent.mkdir(parents=True, exist_ok=True)
    # One line: rpk produce reads newline-delimited records off stdin.
    args.event_out.write_text(
        json.dumps(event, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(args.state_out, next_state)
    args.alerts_out.write_text("".join(f"{a}\n" for a in alerts), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
