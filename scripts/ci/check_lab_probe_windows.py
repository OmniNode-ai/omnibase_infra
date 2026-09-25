#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lab probe-window file and its drift check (OMN-19412, seam L0.3).

WHAT THE FILE IS
    ``config/lab_probe_windows.yaml`` lists every scheduled probe that reads a
    lab lane: the workflow, its repository, its cron, the lane it reads and the
    longest it can run. The reconcile dispatcher (OMN-19420) and the
    ``runtime-train`` skill (OMN-19424) read it so that a lane mutation does not
    land in the gap before a probe. Until this file, those windows existed only
    as prose in one ledger row (2026-09-23T20:23:28Z, lane merge-drain-7f).

WHY A CHECK
    A window list that a person keeps by hand drifts from the workflows the
    first time someone moves a cron, and a stale list is worse than none: the
    dispatcher would trust it. So nothing in an entry is taken on trust. This
    script re-derives each field from the workflow file and refuses:

    * a cron that differs between the file and the workflow (both values named);
    * a lane that differs from what the workflow itself says it reads;
    * a maximum duration that differs from the workflow's job timeouts;
    * a workflow the file names that does not exist or has no schedule;
    * a scheduled probe workflow that has no entry at all ("unlisted").

WHAT COUNTS AS A PROBE (the unlisted direction)
    A scheduled workflow is a probe when it names the lab lane it reads, in one
    of the ways the probes in this fleet already do: a job named
    ``... (<lane> lane)``, a ``run-name`` carrying ``lane=``, or a job pinned to
    the customer machine's runner label. The rule is structural and carries no
    exclusion list. A listed entry need not match the rule (the release train
    and C17 do not name a lane) but must still agree with its workflow.

WHERE THE WORKFLOWS COME FROM
    ``--root <repo>=<path>`` for each of omnibase_infra, omninode_infra and
    omnimarket. The CI job passes the pull request's own checkout for
    omnibase_infra, so a PR that adds a probe cron is judged on its own head,
    and sparse checkouts of the other two repositories at their default branch.
    Every root is required; a missing root or an empty workflow directory is a
    usage error (exit 2), never a clean pass.

THE CONSUMER SIDE
    :func:`load_windows` returns typed :class:`ProbeWindow` records, and
    :meth:`ProbeWindow.occurrences` / :func:`busy_intervals` expand them into
    UTC intervals. Those are the seam the dispatcher and ``runtime-train``
    build against; the blackout policy itself (how long before a window a
    mutation must end) belongs to them, not to this file.

Exit codes: 0 clean, 1 drift or an unlisted probe, 2 usage error or an
unreadable window file.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOW_FILE = REPO_ROOT / "config" / "lab_probe_windows.yaml"

SCHEMA_VERSION = 1
REPOS: tuple[str, ...] = ("omnibase_infra", "omninode_infra", "omnimarket")
LANE_SOURCES: tuple[str, ...] = ("job_name", "run_name", "runs_on", "none")
NO_LANE = "none"

# The runner label of the no-checkout customer machine (omnipc2, .202). A job
# pinned to it is a customer-path probe by construction: the runner group
# admits only those producers (see omninode_infra c13-customer-local-delegation.yml).
CUSTOMER_MACHINE_LABELS: frozenset[str] = frozenset({"omnipc2-customer"})

# GitHub Actions' job timeout when a job declares none.
GITHUB_DEFAULT_TIMEOUT_MINUTES = 360

_ENTRY_KEYS_REQUIRED: tuple[str, ...] = (
    "id",
    "name",
    "repo",
    "workflow",
    "cron",
    "lane",
    "lane_source",
    "max_duration_minutes",
)
_ENTRY_KEYS_OPTIONAL: tuple[str, ...] = ("reads",)
_TOP_KEYS: frozenset[str] = frozenset({"schema_version", "probes"})

_JOB_LANE_RE = re.compile(r"\(([a-z0-9][a-z0-9-]*) lane\)")
_RUN_NAME_LANE_RE = re.compile(
    r"lane=(?:\$\{\{(?P<expr>.*?)\}\}|(?P<lit>[A-Za-z0-9][A-Za-z0-9_.-]*))"
)
_EXPR_DEFAULT_RE = re.compile(r"\|\|\s*'([^']+)'")
_CRON_FIELD_RE = re.compile(r"^(\*|\d+(-\d+)?)(/\d+)?$")

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2


class WindowFileError(ValueError):
    """The window file is unreadable or malformed. Never a clean pass."""


class WorkflowError(ValueError):
    """A workflow file cannot be read or a field cannot be derived from it."""


# --------------------------------------------------------------------------- #
# Cron: the minute and hour fields in full; day, month and weekday must be `*`.
# Every probe the plan names is daily-shaped. A cron outside that shape is
# refused rather than approximated, so the dispatcher never under-counts one.
# --------------------------------------------------------------------------- #


def _expand_field(field: str, low: int, high: int, cron: str) -> tuple[int, ...]:
    values: set[int] = set()
    for part in field.split(","):
        if not _CRON_FIELD_RE.match(part):
            raise WindowFileError(
                f"cron {cron!r}: field {field!r} is not a supported cron field"
            )
        base, _, step_text = part.partition("/")
        step = int(step_text) if step_text else 1
        if step < 1:
            raise WindowFileError(f"cron {cron!r}: step must be at least 1")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            a, b = base.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = int(base)
            end = high if step_text else start
        if not (low <= start <= end <= high):
            raise WindowFileError(f"cron {cron!r}: {part!r} is outside {low}-{high}")
        values.update(range(start, end + 1, step))
    return tuple(sorted(values))


def parse_cron(cron: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (minutes, hours) for a daily-shaped five-field cron, in UTC."""
    fields = cron.split()
    if len(fields) != 5:
        raise WindowFileError(
            f"cron {cron!r}: expected five fields, found {len(fields)}"
        )
    minute, hour, dom, month, dow = fields
    if (dom, month, dow) != ("*", "*", "*"):
        raise WindowFileError(
            f"cron {cron!r}: day-of-month, month and day-of-week must be '*' "
            "(only daily-shaped probe crons are supported; extend parse_cron before listing this one)"
        )
    return _expand_field(minute, 0, 59, cron), _expand_field(hour, 0, 23, cron)


# --------------------------------------------------------------------------- #
# The typed record (the seam's contract).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbeWindow:
    """One scheduled probe. ``occurrences`` and ``busy_intervals`` are UTC."""

    id: str
    name: str
    repo: str
    workflow: str
    cron: tuple[str, ...]
    lane: str
    lane_source: str
    max_duration_minutes: int
    reads: str | None = None

    @property
    def key(self) -> str:
        return f"{self.repo}:{self.workflow}"

    def occurrences(self, start: datetime, end: datetime) -> list[datetime]:
        """Every scheduled start in ``[start, end)``, sorted."""
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("occurrences needs timezone-aware datetimes")
        start_utc, end_utc = start.astimezone(UTC), end.astimezone(UTC)
        found: set[datetime] = set()
        for expression in self.cron:
            minutes, hours = parse_cron(expression)
            day = datetime(start_utc.year, start_utc.month, start_utc.day, tzinfo=UTC)
            while day < end_utc:
                for hour in hours:
                    for minute in minutes:
                        at = day.replace(hour=hour, minute=minute)
                        if start_utc <= at < end_utc:
                            found.add(at)
                day += timedelta(days=1)
        return sorted(found)


def busy_intervals(
    windows: Iterable[ProbeWindow], start: datetime, end: datetime
) -> list[tuple[str, datetime, datetime]]:
    """``(probe id, run start, run start + max duration)`` for every run that
    starts in ``[start, end)``, sorted by start."""
    busy = [
        (w.id, at, at + timedelta(minutes=w.max_duration_minutes))
        for w in windows
        for at in w.occurrences(start, end)
    ]
    return sorted(busy, key=lambda row: (row[1], row[0]))


def _require_str(entry: Mapping[str, Any], key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WindowFileError(f"{where}: {key!r} must be a non-empty string")
    return value


def _parse_entry(raw: object, index: int) -> ProbeWindow:
    where = f"probes[{index}]"
    if not isinstance(raw, dict):
        raise WindowFileError(f"{where}: an entry must be a mapping")
    missing = [k for k in _ENTRY_KEYS_REQUIRED if k not in raw]
    if missing:
        raise WindowFileError(f"{where}: missing required key(s) {missing}")
    extra = sorted(set(raw) - set(_ENTRY_KEYS_REQUIRED) - set(_ENTRY_KEYS_OPTIONAL))
    if extra:
        raise WindowFileError(f"{where}: unknown key(s) {extra}")
    probe_id = _require_str(raw, "id", where)
    where = f"probes[{index}] ({probe_id})"
    repo = _require_str(raw, "repo", where)
    if repo not in REPOS:
        raise WindowFileError(f"{where}: repo {repo!r} is not one of {list(REPOS)}")
    workflow = _require_str(raw, "workflow", where)
    if not workflow.startswith(".github/workflows/") or not workflow.endswith(
        (".yml", ".yaml")
    ):
        raise WindowFileError(
            f"{where}: workflow {workflow!r} must be a .github/workflows/*.yml path"
        )
    cron_raw = raw["cron"]
    if (
        not isinstance(cron_raw, list)
        or not cron_raw
        or not all(isinstance(c, str) for c in cron_raw)
    ):
        raise WindowFileError(f"{where}: 'cron' must be a non-empty list of strings")
    for expression in cron_raw:
        try:
            parse_cron(expression)
        except WindowFileError as exc:
            raise WindowFileError(f"{where}: {exc}") from exc
    lane = _require_str(raw, "lane", where)
    lane_source = _require_str(raw, "lane_source", where)
    if lane_source not in LANE_SOURCES:
        raise WindowFileError(
            f"{where}: lane_source {lane_source!r} is not one of {list(LANE_SOURCES)}"
        )
    if (lane_source == NO_LANE) != (lane == NO_LANE):
        raise WindowFileError(
            f"{where}: lane {lane!r} and lane_source {lane_source!r} disagree about 'none'"
        )
    reads = raw.get("reads")
    if lane_source == NO_LANE and (not isinstance(reads, str) or not reads.strip()):
        raise WindowFileError(
            f"{where}: lane_source 'none' requires a 'reads' statement saying what it reads"
        )
    if reads is not None and not isinstance(reads, str):
        raise WindowFileError(f"{where}: 'reads' must be a string")
    duration = raw["max_duration_minutes"]
    if not isinstance(duration, int) or isinstance(duration, bool) or duration < 1:
        raise WindowFileError(
            f"{where}: max_duration_minutes must be a positive integer, got {duration!r}"
        )
    return ProbeWindow(
        id=probe_id,
        name=_require_str(raw, "name", where),
        repo=repo,
        workflow=workflow,
        cron=tuple(cron_raw),
        lane=lane,
        lane_source=lane_source,
        max_duration_minutes=duration,
        reads=reads,
    )


def load_windows(path: Path) -> list[ProbeWindow]:
    """Load and validate the window file. Refuses rather than defaults."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WindowFileError(f"{path}: unreadable: {exc}") from exc
    if not isinstance(document, dict):
        raise WindowFileError(f"{path}: the top level must be a mapping")
    extra = sorted(set(document) - _TOP_KEYS)
    if extra:
        raise WindowFileError(f"{path}: unknown top-level key(s) {extra}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise WindowFileError(
            f"{path}: schema_version must be {SCHEMA_VERSION}, got {document.get('schema_version')!r}"
        )
    probes = document.get("probes")
    if not isinstance(probes, list) or not probes:
        raise WindowFileError(f"{path}: 'probes' must be a non-empty list")
    windows = [_parse_entry(raw, i) for i, raw in enumerate(probes)]
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for window in windows:
        if window.id in seen_ids:
            raise WindowFileError(f"{path}: id {window.id!r} has more than one entry")
        if window.key in seen_keys:
            raise WindowFileError(
                f"{path}: workflow {window.key} has more than one entry"
            )
        seen_ids.add(window.id)
        seen_keys.add(window.key)
    return windows


# --------------------------------------------------------------------------- #
# Derivation from a workflow file.
# --------------------------------------------------------------------------- #


def read_workflow(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkflowError(f"unreadable: {exc}") from exc
    if not isinstance(document, dict):
        raise WorkflowError("the top level is not a mapping")
    return document


def _triggers(workflow: Mapping[Any, Any]) -> Any:
    # PyYAML (YAML 1.1) reads the bare key `on` as boolean True.
    return workflow.get("on", workflow.get(True))


def workflow_crons(workflow: Mapping[Any, Any]) -> list[str]:
    triggers = _triggers(workflow)
    if not isinstance(triggers, dict):
        return []
    schedule = triggers.get("schedule")
    if not isinstance(schedule, list):
        return []
    return [
        str(item["cron"])
        for item in schedule
        if isinstance(item, dict) and "cron" in item
    ]


def _jobs(workflow: Mapping[Any, Any]) -> dict[str, dict[str, Any]]:
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise WorkflowError("declares no jobs")
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in jobs.items()}


def job_name_lanes(workflow: Mapping[Any, Any]) -> set[str]:
    lanes: set[str] = set()
    for job in _jobs(workflow).values():
        name = job.get("name")
        if isinstance(name, str):
            lanes.update(_JOB_LANE_RE.findall(name))
    return lanes


def run_name_lane(workflow: Mapping[Any, Any]) -> str | None:
    """The lane a scheduled run's title names, or None when it names none."""
    run_name = workflow.get("run-name")
    if not isinstance(run_name, str):
        return None
    match = _RUN_NAME_LANE_RE.search(run_name)
    if match is None:
        return None
    if match.group("lit"):
        return match.group("lit")
    default = _EXPR_DEFAULT_RE.search(match.group("expr") or "")
    if default is None:
        raise WorkflowError(
            "run-name carries lane= with no scheduled default ('... || '<lane>'')"
        )
    return default.group(1)


def _literal_labels(runs_on: object) -> set[str] | None:
    if isinstance(runs_on, str):
        return None if "${{" in runs_on else {runs_on}
    if isinstance(runs_on, list) and all(
        isinstance(x, str) and "${{" not in x for x in runs_on
    ):
        return set(runs_on)
    return None


def runs_on_labels(workflow: Mapping[Any, Any]) -> list[set[str] | None]:
    """Literal runner labels per job; None for a job placed by an expression."""
    return [_literal_labels(job.get("runs-on")) for job in _jobs(workflow).values()]


def lane_markers(workflow: Mapping[Any, Any]) -> dict[str, set[str]]:
    """Every way the workflow names a lane it reads, keyed by lane_source."""
    markers: dict[str, set[str]] = {}
    job_lanes = job_name_lanes(workflow)
    if job_lanes:
        markers["job_name"] = job_lanes
    run_lane = run_name_lane(workflow)
    if run_lane is not None:
        markers["run_name"] = {run_lane}
    customer = {
        label
        for labels in runs_on_labels(workflow)
        if labels
        for label in labels & CUSTOMER_MACHINE_LABELS
    }
    if customer:
        markers["runs_on"] = customer
    return markers


def derive_lane(workflow: Mapping[Any, Any], lane_source: str, declared: str) -> str:
    """The lane the workflow itself says it reads, through ``lane_source``."""
    if lane_source == "job_name":
        lanes = job_name_lanes(workflow)
        if len(lanes) != 1:
            raise WorkflowError(
                f"job names name {sorted(lanes) or 'no'} lane(s); exactly one is required"
            )
        return next(iter(lanes))
    if lane_source == "run_name":
        lane = run_name_lane(workflow)
        if lane is None:
            raise WorkflowError("run-name carries no lane= token")
        return lane
    if lane_source == "runs_on":
        per_job = runs_on_labels(workflow)
        if any(labels is None for labels in per_job):
            raise WorkflowError(
                "a job is placed by an expression, so its runner label cannot be read"
            )
        if all(labels is not None and declared in labels for labels in per_job):
            return declared
        found = sorted({label for labels in per_job if labels for label in labels})
        return ",".join(found)
    markers = lane_markers(workflow)
    if markers:
        named = sorted({lane for lanes in markers.values() for lane in lanes})
        return ",".join(named)
    return NO_LANE


def max_duration_minutes(workflow: Mapping[Any, Any]) -> int:
    """The longest ``needs`` chain of job timeouts (GitHub's 360 when absent)."""
    jobs = _jobs(workflow)
    own: dict[str, int] = {}
    needs: dict[str, list[str]] = {}
    for job_id, job in jobs.items():
        timeout = job.get("timeout-minutes", GITHUB_DEFAULT_TIMEOUT_MINUTES)
        if not isinstance(timeout, int) or isinstance(timeout, bool):
            raise WorkflowError(
                f"job {job_id!r} timeout-minutes {timeout!r} is not a literal integer"
            )
        own[job_id] = timeout
        raw_needs = job.get("needs", [])
        needs[job_id] = (
            [raw_needs] if isinstance(raw_needs, str) else [str(n) for n in raw_needs]
        )
        for need in needs[job_id]:
            if need not in jobs:
                raise WorkflowError(f"job {job_id!r} needs unknown job {need!r}")
    memo: dict[str, int] = {}

    def finish(job_id: str, trail: tuple[str, ...]) -> int:
        if job_id in trail:
            raise WorkflowError(f"needs cycle through {job_id!r}")
        if job_id not in memo:
            memo[job_id] = own[job_id] + max(
                (finish(n, (*trail, job_id)) for n in needs[job_id]), default=0
            )
        return memo[job_id]

    return max(finish(job_id, ()) for job_id in jobs)


# --------------------------------------------------------------------------- #
# The check.
# --------------------------------------------------------------------------- #


def _workflow_files(root: Path) -> list[Path]:
    directory = root / ".github" / "workflows"
    return sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])


def _check_entry(window: ProbeWindow, root: Path) -> list[str]:
    label = f"{window.id} ({window.key})"
    path = root / window.workflow
    if not path.is_file():
        return [f"{label}: workflow not found under {root}"]
    try:
        workflow = read_workflow(path)
        errors: list[str] = []
        crons = workflow_crons(workflow)
        if not crons:
            errors.append(
                f"{label}: the workflow has no schedule, so it is not a scheduled probe"
            )
        elif sorted(crons) != sorted(window.cron):
            errors.append(
                f"{label}: cron mismatch: the window file has {sorted(window.cron)!r}, "
                f"the workflow has {sorted(crons)!r}"
            )
        lane = derive_lane(workflow, window.lane_source, window.lane)
        if lane != window.lane:
            errors.append(
                f"{label}: lane mismatch via {window.lane_source}: the window file has {window.lane!r}, "
                f"the workflow says {lane!r}"
            )
        duration = max_duration_minutes(workflow)
        if duration != window.max_duration_minutes:
            errors.append(
                f"{label}: max_duration_minutes mismatch: the window file has {window.max_duration_minutes}, "
                f"the workflow's job timeouts give {duration}"
            )
    except WorkflowError as exc:
        return [f"{label}: {exc}"]
    return errors


def check(windows: Sequence[ProbeWindow], roots: Mapping[str, Path]) -> list[str]:
    """Every disagreement between ``windows`` and the workflows under ``roots``.

    Entries are checked against their repository's root; every scheduled probe
    under every given root must have an entry. An entry whose repository has
    no root is an error, never skipped.
    """
    errors: list[str] = []
    listed = {w.key for w in windows}
    for window in windows:
        root = roots.get(window.repo)
        if root is None:
            errors.append(
                f"{window.id} ({window.key}): no workflow root was given for {window.repo}"
            )
            continue
        errors.extend(_check_entry(window, root))
    for repo, root in sorted(roots.items()):
        for path in _workflow_files(root):
            key = f"{repo}:{path.relative_to(root).as_posix()}"
            if key in listed:
                continue
            try:
                workflow = read_workflow(path)
                crons = workflow_crons(workflow)
                markers = lane_markers(workflow) if crons else {}
            except WorkflowError as exc:
                errors.append(f"{key}: cannot be classified: {exc}")
                continue
            if crons and markers:
                named = "; ".join(
                    f"{src}={sorted(lanes)}" for src, lanes in sorted(markers.items())
                )
                errors.append(
                    f"unlisted probe {key}: scheduled {sorted(crons)!r} and names the lane it reads "
                    f"({named}), but config/lab_probe_windows.yaml has no entry for it"
                )
    return errors


def _parse_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise WindowFileError(f"--root {value!r}: expected <repo>=<path>")
        if name not in REPOS:
            raise WindowFileError(
                f"--root {value!r}: {name!r} is not one of {list(REPOS)}"
            )
        roots[name] = Path(path)
    missing = [repo for repo in REPOS if repo not in roots]
    if missing:
        raise WindowFileError(
            f"--root is required for every probe repository; missing {missing}"
        )
    for name, root in roots.items():
        if not _workflow_files(root):
            raise WindowFileError(
                f"--root {name}={root}: no workflow files found (an empty root proves nothing)"
            )
    return roots


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOW_FILE)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="REPO=PATH",
        help="a repository checkout holding .github/workflows; required for each of "
        + ", ".join(REPOS),
    )
    args = parser.parse_args(argv)
    try:
        roots = _parse_roots(args.root)
        windows = load_windows(args.windows)
    except WindowFileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    errors = check(windows, roots)
    scanned = ", ".join(
        f"{repo}={len(_workflow_files(root))}" for repo, root in sorted(roots.items())
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"lab probe windows: {len(errors)} error(s); {len(windows)} entries; workflows scanned: {scanned}"
        )
        return EXIT_DRIFT
    print(
        f"lab probe windows: OK; {len(windows)} entries agree with their workflows; workflows scanned: {scanned}"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
