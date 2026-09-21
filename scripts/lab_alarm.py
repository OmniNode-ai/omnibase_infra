#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Watch three lab conditions on a durable timer and raise once each (OMN-18867).

THE DEFECT THIS EXISTS TO CLOSE

Three lab conditions ran unobserved in one window:

* the savings writer crash-looping on an oversize snapshot, frozen **nine
  days** at lag 498 (OMN-18851);
* ``omninode-runtime`` crash-looping on a projection grant from a stale pin
  (omnibase_infra#3857);
* compose-dev lab-pass receipts FAILing for hours while a release lane waited.

Every one was found by a person looking, and the first nine days late.
**Detection time is the defect, not the conditions.**

WHY A LAUNCH AGENT AND NOT A SESSION TICK

Session-scoped scheduling is in-memory and dies with the session; this has to
survive one. ``StartCalendarInterval`` under launchd is the durable local
trigger, installed by ``scripts/launchd/install-lab-alarm.sh``.

The OMN-17173 correction applies and is the reason the installer prints a
readback: **launchd DOES fire on this Mac** -- the earlier ticks had been
bootout'd, not failed -- so before believing this runs, check BOTH a non-zero
PID column in ``launchctl list`` AND a fresh entry in its own log. A
loaded-but-bootout'd job looks installed and does nothing.

EDGE-TRIGGERED, NOT LEVEL-TRIGGERED

An alarm is raised when a subject ENTERS a bad state and not again until it
leaves. A condition that re-alarms every tick is one nobody reads by the
second day, and an alarm nobody reads is the state this ticket is trying to
get out of. The subjects currently alarming live in the state file; a subject
that returns OK is cleared from it and can alarm again later.

EVERY RUN IS RECORDED, INCLUDING THE QUIET ONES

A silent run and a run that never happened are indistinguishable from outside,
and telling them apart is half the point of a durable timer. So every run
appends one record naming **all three** conditions, their outcome and their
evidence, whether or not anything fired.

THREE OUTCOMES, NOT TWO

``OK``, ``ALARM`` and ``INDETERMINATE``. The third is load bearing: a docker
socket that will not answer, a broker that refuses, an artifact surface that
errors, or a lag condition with no previous sample to compare against are all
conditions the run COULD NOT EVALUATE. Folding those into ``OK`` is how a
monitor reports green over an outage -- Operating Rule 16, an empty result is
not evidence of absence.

WHAT THIS ALARM DOES NOT CLAIM

**A zero or a green here is not a liveness statement about the node behind
it.** Measured on this lane on 2026-09-20 (OMN-18881): a consumer group can
read Stable with assigned partitions and zero lag while the node behind it
refuses every message, and both dev-lane runtimes were docker-unhealthy on a
single dimension while ``/ready`` and ``/health`` returned 200. So the lag
condition measures LAG AND ITS GROWTH and nothing more, which is exactly the
failure it was built for, and the evidence string on every reading names which
fact was read so a green cannot be quoted as something it is not.

**A container's STATUS is recorded but not graded.** Measured on this lane on
2026-09-21: ``omnimarket-projection-api`` read ``exited`` with a restart count
of 0, which this condition correctly reports as inside its bound, because the
bound is on RESTARTS. Grading status would need its own per-container
declaration -- the migration gate exits on purpose every boot and a writer does
not -- so the status travels in the evidence where a reader can see it, and
alarming on it is a separate condition nobody has declared bounds for yet.

**No dead-letter condition is built here**, deliberately. The boundary writes
to a DERIVED per-class topic rather than the ``dlq_topics`` a contract
declares, so a probe reading the declared topic sees a permanent zero on a
healthy and an unhealthy lane alike -- two lanes concluded "messages are being
lost" from that zero on 2026-09-20. Adding one means reading the derived topic
AND shipping a positive control proving the reader can see a row at all.

NOTHING IS SENT ANYWHERE

**There is no posting arm in this module and no network client of any kind.**
No agent message is consent, the ticket is not consent and the plan is not
consent. :func:`resolve_posting_consent` is the gate a future sender must pass:
a durable OPERATOR-CONSENT row in the rolling work ledger carrying the
operator's verbatim words, a destination channel in APPROVED SCOPE and an OUT
OF SCOPE list. Until such a row exists it returns ``None``, every run records
``posting: disabled``, and the alarm is still useful because it writes a
durable local artifact a person and a sweep can both read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    EnumLabPassResult,
    ModelBrokerAccess,
    ModelLabPassReceipt,
    ReceiptLookupError,
    artifact_name,
    download_receipt,
    list_artifacts,
    read_group_total_lag,
)

RUN_RECORD_VERSION = "lab_alarm_run.v1"

#: How many entries of a subject's own history the evidence quotes. Two, because
#: the growth condition is a statement about a PAIR of readings and evidence
#: that does not carry both cannot be checked by the person reading it.
_EVIDENCE_SAMPLES = 2


class EnumAlarmCondition(StrEnum):
    """The three conditions, each with a declared bound rather than a judgement."""

    LAB_PASS_RECEIPT = "lab_pass_receipt"
    CONTAINER_RESTARTS = "container_restarts"
    CONSUMER_GROUP_LAG = "consumer_group_lag"


class EnumConditionOutcome(StrEnum):
    OK = "OK"
    ALARM = "ALARM"
    INDETERMINATE = "INDETERMINATE"


class CommandRunner(Protocol):
    """A ``subprocess.run``-shaped callable, so every condition is testable.

    Declared rather than reaching for ``subprocess.run`` implicitly because
    each condition below is exercised by a NEGATIVE control -- a test driving
    it against a known-bad reading. A condition no test has ever made fire is
    a condition never proven capable of firing.
    """

    def __call__(
        self, argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]: ...


def _run_read_only(
    argv: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a fixed-argv, no-shell, read-only command."""
    return subprocess.run(
        list(argv), capture_output=True, text=True, check=False, timeout=timeout
    )


def make_runner(docker_command: Sequence[str]) -> CommandRunner:
    """Return a runner that reaches docker wherever the lane actually is.

    **The lab lane does not run on the host this timer runs on.** The
    containers live on the lab box and the launching Mac's docker daemon
    cannot see them at all -- measured: a local ``docker ps`` filtered to the
    lane's compose project fails to connect to any socket. A timer that shelled
    out to a bare ``docker`` would therefore report every container unreadable,
    forever, on a healthy lane and an unhealthy one alike.

    The transport belongs here rather than inside each condition because
    :class:`CommandRunner` is already the seam every condition is tested
    through. Rewriting ``docker`` at the seam means the conditions, and the
    ``read_group_total_lag`` helper this module reuses unchanged from the
    lab-pass receipt, need no knowledge of where the lane is.

    Remote arguments are shell-quoted because ssh concatenates them and hands
    the result to a remote shell, so an unquoted argument is a remote shell
    injection waiting for a group name with a space in it.
    """
    prefix = list(docker_command)
    if prefix == ["docker"]:
        return _run_read_only

    def run(argv: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        rest = list(argv)
        if rest and rest[0] == "docker":
            rest = prefix + [shlex.quote(part) for part in rest[1:]]
        return _run_read_only(rest, timeout=timeout)

    return run


@dataclass(frozen=True)
class ModelAlarm:
    """One subject that has ENTERED a bad state."""

    condition: EnumAlarmCondition
    subject: str
    detail: str

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("an alarm must name its subject")
        if not self.detail:
            raise ValueError("an alarm must carry a detail a reader can act on")

    @property
    def key(self) -> str:
        """Identity for edge-triggering: one alarm per subject per condition."""
        return f"{self.condition.value}:{self.subject}"

    def to_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition.value,
            "subject": self.subject,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ModelConditionReport:
    """What one condition evaluated to, and the reading it evaluated it from."""

    condition: EnumAlarmCondition
    outcome: EnumConditionOutcome
    evidence: str
    alarms: tuple[ModelAlarm, ...] = ()

    def __post_init__(self) -> None:
        # Evidence is required on EVERY outcome, passing ones included: an OK
        # with an empty evidence string is indistinguishable from a condition
        # that was never evaluated, which is the distinction this whole module
        # exists to preserve.
        if not self.evidence.strip():
            raise ValueError(f"{self.condition.value} reported no evidence")
        if self.outcome is EnumConditionOutcome.ALARM and not self.alarms:
            raise ValueError(f"{self.condition.value} is ALARM with no alarm")
        if self.outcome is not EnumConditionOutcome.ALARM and self.alarms:
            raise ValueError(f"{self.condition.value} carries alarms but is not ALARM")

    def to_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition.value,
            "outcome": self.outcome.value,
            "evidence": self.evidence,
            "alarms": [alarm.to_json() for alarm in self.alarms],
        }


@dataclass(frozen=True)
class ModelAlarmRun:
    """One tick. Carries all three conditions whether or not anything fired."""

    started_at: str
    finished_at: str
    reports: tuple[ModelConditionReport, ...]
    raised: tuple[ModelAlarm, ...]
    posting: str
    run_version: str = RUN_RECORD_VERSION

    def __post_init__(self) -> None:
        evaluated = {report.condition for report in self.reports}
        missing = sorted(c.value for c in EnumAlarmCondition if c not in evaluated)
        if missing:
            # A run that skipped a condition must not be recordable. Otherwise
            # "the alarm ran and found nothing" silently becomes "the alarm ran
            # two of three conditions", which reads identically in the log.
            raise ValueError(f"run evaluated no {', '.join(missing)} condition")
        if len(evaluated) != len(self.reports):
            raise ValueError("a condition is reported twice in one run")

    def to_json(self) -> dict[str, Any]:
        return {
            "run_version": self.run_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "conditions_evaluated": sorted(c.value for c in EnumAlarmCondition),
            "reports": [report.to_json() for report in self.reports],
            "raised": [alarm.to_json() for alarm in self.raised],
            "posting": self.posting,
        }


# ---------------------------------------------------------------------------
# Condition 1 — a FAIL lab-pass receipt for the sha the lane is running
# ---------------------------------------------------------------------------


class ReceiptReader(Protocol):
    def __call__(self, repo: str, lane: EnumLabLane, sha: str) -> ModelLabPassReceipt:
        """Return the newest receipt for this exact sha, or raise."""


def read_latest_receipt(repo: str, lane: EnumLabLane, sha: str) -> ModelLabPassReceipt:
    """Resolve one receipt off the artifact surface, by exact name.

    Raises:
        ReceiptLookupError: the surface could not be read, or carries no
            receipt for this sha. Never returns a placeholder -- the caller
            grades an unreadable surface as INDETERMINATE, and a placeholder
            would arrive there as a verdict instead.
    """
    name = artifact_name(lane, sha)
    artifacts = list_artifacts(repo, name)
    if not artifacts:
        raise ReceiptLookupError(f"no unexpired artifact named {name}")
    newest = sorted(artifacts, key=lambda a: str(a.get("created_at", "")), reverse=True)
    artifact_id = newest[0].get("id")
    if not isinstance(artifact_id, int):
        raise ReceiptLookupError(f"artifact {name} carries no integer id")
    return download_receipt(repo, artifact_id)


def resolve_delivered_sha(*, agent_url: str, ready_url: str) -> tuple[str, str]:
    """Resolve the sha the lane is running, naming the surface that answered.

    Two surfaces, in descending directness, mirroring the order
    ``resolve_lane_binding`` already establishes for the same question:

    1. the deploy agent's recorded ``loaded_code_sha`` -- an actual commit;
    2. the runtime's ``/ready`` revision.

    **The ``/ready`` surface alone is not enough, measured rather than
    assumed.** On this lane on 2026-09-21 ``/ready`` answered healthy and
    carried a package version and no commit anywhere in its payload, so a
    resolver built on it alone reports "unresolved" on a perfectly healthy
    lane, forever -- which would make the receipt condition permanently
    INDETERMINATE and the alarm permanently half-blind. The agent answered
    with a 40-hex sha on the same host at the same moment.

    Returns:
        ``(sha, surface)``. An empty sha with the surfaces tried named, so the
        condition's evidence can say which surfaces were silent rather than
        reporting a bare failure.
    """
    from scripts.ci.lab_pass_receipt import (
        read_agent_loaded_code_sha,
        read_ready_revision,
    )

    for surface, reader, url in (
        ("deploy-agent loaded_code_sha", read_agent_loaded_code_sha, agent_url),
        ("runtime /ready revision", read_ready_revision, ready_url),
    ):
        if not url:
            continue
        resolved = reader(url)
        if resolved:
            return resolved, surface
    return "", "no surface answered (deploy agent, runtime /ready)"


def evaluate_lab_pass_receipt(
    *,
    repo: str,
    lane: EnumLabLane,
    sha: str,
    reader: ReceiptReader,
) -> ModelConditionReport:
    """A FAIL receipt for the delivered sha is an alarm naming that sha."""
    condition = EnumAlarmCondition.LAB_PASS_RECEIPT
    if not sha:
        return ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.INDETERMINATE,
            evidence=(
                "the lane's delivered sha could not be resolved, so no receipt "
                "could be looked up; an unresolved sha is not a passing lane"
            ),
        )
    try:
        receipt = reader(repo, lane, sha)
    except ReceiptLookupError as exc:
        return ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.INDETERMINATE,
            evidence=(
                f"the receipt surface for {lane.value} at {sha} could not be "
                f"read: {exc}. Unread is not passed"
            ),
        )

    failed = [check.name for check in receipt.checks if not check.ok]
    if receipt.result is EnumLabPassResult.FAIL:
        detail = (
            f"compose lane {receipt.lane.value} receipt for {receipt.sha} is FAIL "
            f"on {', '.join(failed) or 'an unnamed check'} "
            f"(finished {receipt.finished_at})"
        )
        return ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.ALARM,
            evidence=detail,
            alarms=(ModelAlarm(condition=condition, subject=sha, detail=detail),),
        )
    return ModelConditionReport(
        condition=condition,
        outcome=EnumConditionOutcome.OK,
        evidence=(
            f"lane {receipt.lane.value} receipt for {receipt.sha} is "
            f"{receipt.result.value} over {len(receipt.checks)} checks"
        ),
    )


# ---------------------------------------------------------------------------
# Condition 2 — a lane container past its declared restart bound
# ---------------------------------------------------------------------------

_RESTART_FORMAT = "{{.RestartCount}} {{.State.Status}}"


def read_container_restarts(
    container: str, *, runner: CommandRunner, timeout_seconds: float = 30.0
) -> tuple[int, str]:
    """Return ``(restart count, status)`` for one container.

    Raises:
        ValueError: docker could not answer. An absent or unreadable container
            is NOT "inside its bound" -- a container that is gone is a finding
            of its own shape, and calling it healthy is the quiet green.
    """
    result = runner(
        ["docker", "inspect", "-f", _RESTART_FORMAT, container],
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise ValueError(
            f"docker inspect {container} exited {result.returncode}: "
            f"{(result.stderr or '').strip()[:200]}"
        )
    fields = (result.stdout or "").split()
    if len(fields) < 2:
        raise ValueError(f"docker inspect {container} printed {result.stdout!r}")
    try:
        return int(fields[0]), fields[1]
    except ValueError as exc:
        raise ValueError(
            f"restart count for {container} is not an integer: {fields[0]!r}"
        ) from exc


def evaluate_container_restarts(
    bounds: Mapping[str, int],
    *,
    runner: CommandRunner,
    timeout_seconds: float = 30.0,
) -> ModelConditionReport:
    """Each declared container is at or under its own declared restart bound.

    The bound is per container and declared in configuration, not a single
    fleet-wide number: a broker that has never restarted and a runtime that
    restarts on every deploy do not share a threshold, and one number for both
    is either noise or blindness.
    """
    condition = EnumAlarmCondition.CONTAINER_RESTARTS
    if not bounds:
        return ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.INDETERMINATE,
            evidence=(
                "no container carries a declared restart bound, so nothing was "
                "checked; an empty subject list is not a healthy lane"
            ),
        )

    alarms: list[ModelAlarm] = []
    unreadable: list[str] = []
    readings: list[str] = []
    for container in sorted(bounds):
        bound = bounds[container]
        try:
            restarts, status = read_container_restarts(
                container, runner=runner, timeout_seconds=timeout_seconds
            )
        except ValueError as exc:
            unreadable.append(f"{container} ({exc})")
            continue
        readings.append(f"{container}={restarts}/{bound} {status}")
        if restarts > bound:
            alarms.append(
                ModelAlarm(
                    condition=condition,
                    subject=container,
                    detail=(
                        f"container {container} has restarted {restarts} times, "
                        f"past its declared bound of {bound}; status {status}"
                    ),
                )
            )

    evidence = f"restart counts read: {', '.join(readings) or 'none'}"
    if unreadable:
        # Unreadable dominates: a partial read that found nothing wrong says
        # nothing about the containers it could not reach.
        return ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.INDETERMINATE,
            evidence=f"{evidence}; UNREADABLE: {'; '.join(unreadable)}",
        )
    if alarms:
        return ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.ALARM,
            evidence=evidence,
            alarms=tuple(alarms),
        )
    return ModelConditionReport(
        condition=condition,
        outcome=EnumConditionOutcome.OK,
        evidence=f"{evidence}; every container is inside its declared bound",
    )


# ---------------------------------------------------------------------------
# Condition 3 — a declared consumer group whose lag GROWS across two samples
# ---------------------------------------------------------------------------


def evaluate_consumer_group_lag(
    access: ModelBrokerAccess,
    groups: Sequence[str],
    *,
    previous: Mapping[str, int] | None,
    runner: CommandRunner,
    timeout_seconds: float = 60.0,
) -> tuple[ModelConditionReport, dict[str, int]]:
    """Growth across two consecutive samples, and ONLY growth.

    **A high but FLAT lag raises nothing, deliberately.** A bound alone cannot
    see the failure this exists for and cannot avoid the noise that would get
    it muted: the savings writer sat at 498 for nine days (OMN-18851), a number
    a generous bound admits and a tight bound would flag on every healthy busy
    group as well. Growth separates a backlog being worked from a consumer that
    has stopped, and it is the only one of the two that is scale-free. Flat-lag
    alarming is how that nine-day freeze would have been ignored a second time.

    The two samples straddle two TICKS rather than two seconds within one run,
    because a consumer that has stopped and a consumer mid-poll are
    indistinguishable seconds apart.

    Returns:
        The report, and this run's sample for the caller to persist. The sample
        is returned even on an INDETERMINATE reading, so that a run which could
        not compare still leaves a baseline for the next one.

    What this does NOT measure: whether the node behind the group is alive. A
    group can read Stable with zero lag while its node refuses every message
    (OMN-18881). The evidence string names the fact read, so a green reading
    cannot be quoted as liveness.
    """
    condition = EnumAlarmCondition.CONSUMER_GROUP_LAG
    if not groups:
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.INDETERMINATE,
                evidence=(
                    "no consumer group is declared, so nothing was sampled; an "
                    "empty group list is not a lane with no lag"
                ),
            ),
            {},
        )

    sample: dict[str, int] = {}
    unreadable: list[str] = []
    for group in sorted(groups):
        try:
            sample[group] = read_group_total_lag(
                access, group, runner=runner, timeout_seconds=timeout_seconds
            )
        except ValueError as exc:
            unreadable.append(f"{group} ({exc})")

    readings = ", ".join(f"{g}={sample[g]}" for g in sorted(sample)) or "none"
    read_fact = (
        f"TOTAL-LAG read from rpk group describe via {access.container}"
        f"{' with SASL' if access.authenticated else ' unauthenticated'}"
    )

    if unreadable:
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.INDETERMINATE,
                evidence=(
                    f"{read_fact}; sampled {readings}; UNREADABLE: "
                    f"{'; '.join(unreadable)}. An unreadable group is not a "
                    "group at zero"
                ),
            ),
            sample,
        )

    if previous is None:
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.INDETERMINATE,
                evidence=(
                    f"{read_fact}; sampled {readings}; no previous sample, so "
                    "growth could not be evaluated and this run is the baseline"
                ),
            ),
            sample,
        )

    alarms: list[ModelAlarm] = []
    compared: list[str] = []
    for group in sorted(sample):
        before = previous.get(group)
        now = sample[group]
        if before is None:
            compared.append(f"{group}: new, {now}")
            continue
        compared.append(f"{group}: {before}->{now}")
        if now > before:
            alarms.append(
                ModelAlarm(
                    condition=condition,
                    subject=group,
                    detail=(
                        f"consumer group {group} lag GREW from {before} to {now} "
                        f"across two consecutive samples. {read_fact}. This is a "
                        "lag statement, not a liveness statement about the node "
                        "behind the group"
                    ),
                )
            )

    evidence = f"{read_fact}; {'; '.join(compared[: _EVIDENCE_SAMPLES * 8]) or 'none'}"
    if alarms:
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.ALARM,
                evidence=evidence,
                alarms=tuple(alarms),
            ),
            sample,
        )
    return (
        ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.OK,
            evidence=(
                f"{evidence}; no declared group grew between the two samples "
                "(a high but flat lag is deliberately not an alarm)"
            ),
        ),
        sample,
    )


# ---------------------------------------------------------------------------
# The consent gate a future sender must pass
# ---------------------------------------------------------------------------

_CONSENT_ROW = re.compile(
    r"\|\s*OPERATOR-CONSENT\s*\|.*?APPROVED SCOPE:(?P<approved>.*?)\|"
    r"\s*OUT OF SCOPE:(?P<out>.*?)(\||$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelPostingConsent:
    """A resolved, citable authorization to send this alarm somewhere."""

    channel: str
    ledger_path: str
    line: int

    @property
    def citation(self) -> str:
        return f"{self.ledger_path}:{self.line}"


def resolve_posting_consent(
    ledger_path: Path, *, channel: str
) -> ModelPostingConsent | None:
    """Return the consent row authorizing a send to *channel*, or ``None``.

    **There is no sender in this module and this function does not create
    one.** It is the gate a sender must pass when one is written, and it is
    shipped now rather than later so the refusal is a tested behaviour instead
    of an intention.

    A row qualifies only when it is an OPERATOR-CONSENT row carrying BOTH scope
    lists and naming *channel* in APPROVED SCOPE. Both lists are required
    because the OUT OF SCOPE half is what bounds the grant, and a row missing
    it looks identical to a valid one to the next reader. An unreadable ledger
    returns ``None``: a grant that cannot be read is not a grant.
    """
    try:
        lines = ledger_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for number, line in enumerate(lines, start=1):
        match = _CONSENT_ROW.search(line)
        if match is None:
            continue
        approved = match.group("approved")
        if not match.group("out").strip():
            continue
        if channel.lower() in approved.lower():
            return ModelPostingConsent(
                channel=channel, ledger_path=str(ledger_path), line=number
            )
    return None


# ---------------------------------------------------------------------------
# State: edge-triggering and the previous lag sample
# ---------------------------------------------------------------------------


@dataclass
class ModelAlarmState:
    """What the previous tick left behind."""

    active: dict[str, str] = field(default_factory=dict)
    lag_sample: dict[str, int] | None = None

    @classmethod
    def load(cls, path: Path) -> ModelAlarmState:
        """Read the state file, treating an unreadable one as empty.

        An unreadable state file makes the next run re-alarm every active
        subject and re-baseline the lag comparison. That is the SAFE direction:
        a duplicate alarm is noise, a suppressed one is the nine-day freeze.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(payload, dict):
            return cls()
        active = payload.get("active")
        sample = payload.get("lag_sample")
        return cls(
            active={
                str(k): str(v) for k, v in active.items() if isinstance(active, dict)
            }
            if isinstance(active, dict)
            else {},
            lag_sample={str(k): int(v) for k, v in sample.items()}
            if isinstance(sample, dict)
            else None,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"active": self.active, "lag_sample": self.lag_sample}, indent=2
            ),
            encoding="utf-8",
        )


def select_new_alarms(
    reports: Sequence[ModelConditionReport], state: ModelAlarmState, *, now: str
) -> tuple[ModelAlarm, ...]:
    """Raise a subject only as it ENTERS a bad state; clear it when it leaves.

    Level-triggering was the alternative and is rejected: a condition that
    re-fires every tick is muted by the second day, and a muted alarm is worse
    than no alarm because it reads as coverage.

    A subject is cleared only by an ``OK`` on its own condition. An
    INDETERMINATE deliberately does NOT clear it -- "I could not tell" is not
    "it recovered", and clearing on it would re-raise the same alarm on the
    next readable tick, which is the duplicate this function exists to prevent.
    """
    fresh: list[ModelAlarm] = []
    for report in reports:
        prefix = f"{report.condition.value}:"
        if report.outcome is EnumConditionOutcome.OK:
            for key in [k for k in state.active if k.startswith(prefix)]:
                del state.active[key]
            continue
        for alarm in report.alarms:
            if alarm.key in state.active:
                continue
            state.active[alarm.key] = now
            fresh.append(alarm)
    return tuple(fresh)


# ---------------------------------------------------------------------------
# Configuration and the run
# ---------------------------------------------------------------------------


_ENV_REF = re.compile(r"\$\{(?P<name>[A-Z_][A-Z0-9_]*)\}")


def expand_env(value: str, *, source: Path) -> str:
    """Substitute ``${VAR}`` references, RAISING on an unset variable.

    The config names the lab host as ``${ONEX_INFRA_HOST}`` rather than
    carrying its address, because a literal machine address in source is what
    Operating Rule 6 forbids and a config that only works on one laptop is the
    cross-machine breakage that rule exists to prevent.

    Unset raises rather than expanding to an empty string (Operating Rule 8):
    a silently empty host produces a URL that fails to connect, which this
    alarm would correctly grade INDETERMINATE and a reader would then spend an
    afternoon debugging as a lane outage.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        resolved = os.environ.get(name)
        if not resolved:
            raise ValueError(
                f"{source}: ${{{name}}} is unset, so this config cannot be "
                "resolved; refusing rather than probing an empty address"
            )
        return resolved

    return _ENV_REF.sub(replace, value)


@dataclass(frozen=True)
class ModelAlarmConfig:
    """Declared subjects and bounds. JSON, not YAML, on purpose.

    This module runs under launchd on the brew interpreter with no virtual
    environment -- the same reason ``scripts/ci/lab_pass_receipt.py`` is
    stdlib-only. A YAML config would put a third-party import on the path of a
    timer whose whole value is that it keeps running unattended.
    """

    repo: str
    lane: EnumLabLane
    ready_url: str
    agent_url: str
    broker_container: str
    broker_address: str
    docker_command: tuple[str, ...]
    container_restart_bounds: dict[str, int]
    consumer_groups: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> ModelAlarmConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not a JSON object")
        bounds = payload.get("container_restart_bounds") or {}
        if not isinstance(bounds, dict):
            raise ValueError(f"{path}: container_restart_bounds must be an object")
        groups = payload.get("consumer_groups") or []
        if not isinstance(groups, list):
            raise ValueError(f"{path}: consumer_groups must be a list")
        docker = payload.get("docker_command") or ["docker"]
        if not isinstance(docker, list) or not docker:
            raise ValueError(f"{path}: docker_command must be a non-empty list")
        return cls(
            repo=expand_env(str(payload["repo"]), source=path),
            lane=EnumLabLane(str(payload.get("lane", EnumLabLane.COMPOSE_DEV.value))),
            ready_url=expand_env(str(payload["ready_url"]), source=path),
            agent_url=expand_env(str(payload.get("agent_url", "")), source=path),
            broker_container=str(payload["broker_container"]),
            broker_address=expand_env(str(payload["broker_address"]), source=path),
            docker_command=tuple(expand_env(str(p), source=path) for p in docker),
            container_restart_bounds={str(k): int(v) for k, v in bounds.items()},
            consumer_groups=tuple(str(g) for g in groups),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_once(
    config: ModelAlarmConfig,
    *,
    state_dir: Path,
    ledger_path: Path,
    sha: str,
    receipt_reader: ReceiptReader,
    runner: CommandRunner,
    posting_channel: str,
) -> ModelAlarmRun:
    """Evaluate all three conditions, record the run, return it."""
    started = _now()
    state = ModelAlarmState.load(state_dir / "state.json")

    receipt_report = evaluate_lab_pass_receipt(
        repo=config.repo, lane=config.lane, sha=sha, reader=receipt_reader
    )
    restart_report = evaluate_container_restarts(
        config.container_restart_bounds, runner=runner
    )
    access = ModelBrokerAccess(
        container=config.broker_container,
        brokers=config.broker_address,
        sasl_mechanism=os.environ.get("KAFKA_SASL_MECHANISM", ""),
        sasl_username=os.environ.get("KAFKA_SASL_USERNAME", ""),
        sasl_password=os.environ.get("KAFKA_SASL_PASSWORD", ""),
    )
    lag_report, lag_sample = evaluate_consumer_group_lag(
        access, config.consumer_groups, previous=state.lag_sample, runner=runner
    )

    reports = (receipt_report, restart_report, lag_report)
    raised = select_new_alarms(reports, state, now=started)
    state.lag_sample = lag_sample
    state.save(state_dir / "state.json")

    consent = resolve_posting_consent(ledger_path, channel=posting_channel)
    posting = (
        f"disabled: no OPERATOR-CONSENT row in {ledger_path} names "
        f"{posting_channel} in APPROVED SCOPE"
        if consent is None
        else f"authorized by {consent.citation}; no sender is wired in this module"
    )

    run = ModelAlarmRun(
        started_at=started,
        finished_at=_now(),
        reports=reports,
        raised=raised,
        posting=posting,
    )
    _append_jsonl(state_dir / "alarm-runs.jsonl", run.to_json())
    for alarm in raised:
        _append_jsonl(
            state_dir / "alarms.jsonl", {"raised_at": run.started_at, **alarm.to_json()}
        )
    return run


def render(run: ModelAlarmRun) -> str:
    lines = [f"lab alarm {run.started_at} -> {run.finished_at}"]
    for report in run.reports:
        lines.append(f"  [{report.outcome.value:<13}] {report.condition.value}")
        lines.append(f"      {report.evidence}")
    if run.raised:
        lines.append("  RAISED THIS RUN:")
        lines += [f"      {alarm.subject}: {alarm.detail}" for alarm in run.raised]
    else:
        lines.append("  raised nothing new this run")
    lines.append(f"  posting: {run.posting}")
    return "\n".join(lines)


#: Exit codes. The durable JSONL is the record a person reads; the exit code is
#: what every OTHER consumer reads -- launchd's own status accounting, a
#: wrapper, a future sweep. A monitor that printed a finding and exited 0 would
#: be the "prints and passes" shape this whole epic is about, so the verdict is
#: in the status as well as in the artifact.
EXIT_OK = 0
EXIT_ALARM = 1
EXIT_INDETERMINATE = 2


def exit_code(run: ModelAlarmRun) -> int:
    """Alarm beats indeterminate beats OK.

    INDETERMINATE is deliberately NOT 0: a run that could not evaluate a
    condition has not found the lane healthy, and collapsing it into success
    is exactly how a monitor reports green over an outage. It ranks below
    ALARM only because a known failure is the more actionable of the two.
    """
    if run.raised:
        return EXIT_ALARM
    if any(
        report.outcome is EnumConditionOutcome.INDETERMINATE for report in run.reports
    ):
        return EXIT_INDETERMINATE
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument(
        "--sha",
        default="",
        help="Delivered sha; resolved from the lane's own /ready when omitted.",
    )
    parser.add_argument(
        "--posting-channel",
        default="#onex-lab-alarms",
        help="The channel a consent row would have to name. Nothing is sent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ModelAlarmConfig.load(args.config)

    sha, surface = (
        (args.sha, "argv")
        if args.sha
        else resolve_delivered_sha(
            agent_url=config.agent_url, ready_url=config.ready_url
        )
    )
    sys.stderr.write(f"delivered sha: {sha or '(unresolved)'} via {surface}\n")

    run = run_once(
        config,
        state_dir=args.state_dir,
        ledger_path=args.ledger,
        sha=sha,
        receipt_reader=read_latest_receipt,
        runner=make_runner(config.docker_command),
        posting_channel=args.posting_channel,
    )
    sys.stdout.write(render(run) + "\n")
    return exit_code(run)


if __name__ == "__main__":
    raise SystemExit(main())
