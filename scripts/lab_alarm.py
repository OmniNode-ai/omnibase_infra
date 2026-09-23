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
appends one record naming **all five** conditions (OMN-19091 added
EFFECTS_HELD_BEHIND_RUNTIME and STALE_INDETERMINATE), their outcome and their
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

DELIVERY IS CONSENTED, CITED, AND EDGE-TRIGGERED

**No agent message is consent.** Not a dispatch brief, not the ticket, not the
plan. The only thing that authorizes a send is a durable row in the rolling
work ledger, resolved at run time by :func:`resolve_posting_consent`, and every
run records the citation it resolved or the fact that it resolved nothing.

The live authorization is an operator ruling of 2026-09-21 permitting this
alarm to post to one named channel using the lab's existing bot token **by
reference**. The token is read out of an env file by NAME at send time, held
for one request, and never logged, never written to the run record, never put
on a command line and never included in an error message -- a failure names
the variable and the file instead.

Delivery is per NEWLY RAISED alarm, so the edge-triggering above is also the
channel's deduplication: a condition that stays bad posts once, not once an
hour. A failed send is recorded as a failure rather than swallowed, because an
alarm that believes it delivered and did not is this ticket's own defect in a
new place.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    EnumLabPassResult,
    ModelLabPassReceipt,
    ReceiptLookupError,
    artifact_name,
    download_receipt,
    list_artifacts,
)

#: OMN-18867/OMN-19091: the consumer-lag condition reads the dev broker
#: in-process via aiokafka, using the same ~/.onex client store the hook edge
#: and ``onex delegate --lane dev`` already read (OPERATOR-CONSENT
#: ``docs/tracking/ROLLING_WORK_LEDGER.md:3853`` item 7). This is a
#: DELIBERATE departure from this module's otherwise stdlib-only design: a
#: ruling of 2026-09-21 ("a secret on argv does not satisfy the consent,
#: transient or not") ruled out the rpk-over-ssh path this condition used to
#: take, because rpk has no credential-input mechanism that keeps a value off
#: argv. Both ``yaml`` and ``aiokafka`` are confirmed present in the bare brew
#: interpreter's global site-packages on this Mac (no venv, nothing installed
#: by this change) -- see ``test_the_alarm_runs_on_the_brew_interpreter_with_no_virtualenv``.
#: ``pydantic`` stays forbidden: nothing here needs a model, only two files
#: and a socket.
_ONEX_LANE = "dev"

RUN_RECORD_VERSION = "lab_alarm_run.v1"

#: The chat endpoint and the NAME of the credential, never its value.
#:
#: Named here rather than taken on argv so the token cannot reach a process
#: listing, a shell history or a log line. The alarm reads it out of an env
#: file at send time and holds it only for the duration of one request.
# fmt: off
SLACK_POST_URL = "https://slack.com/api/chat.postMessage"  # url-authority-ok: fixed public Slack Web API method, no ONEX routing authority -- same contract as the existing chat.postMessage calls in scripts/ci/runner_saturation_record.py and scripts/ci/nonrequired_check_failure_rate.py
# fmt: on
SLACK_TOKEN_VAR = "SLACK_BOT_TOKEN"

#: How many entries of a subject's own history the evidence quotes. Two, because
#: the growth condition is a statement about a PAIR of readings and evidence
#: that does not carry both cannot be checked by the person reading it.
_EVIDENCE_SAMPLES = 2


class EnumAlarmCondition(StrEnum):
    """Five conditions, each with a declared bound rather than a judgement.

    The first four watch the lab. The fifth, STALE_INDETERMINATE
    (OMN-19091), watches the other four: an alarm that reads INDETERMINATE
    on every sample forever is installed but not watching, and today that
    state was silent. It is a real condition, not a side channel -- evaluated
    every run, carrying its own evidence, subject to the same edge-triggered
    delivery -- because a condition that only watches state this module
    already persists needs no new transport to stay honest about its own
    blind spots.

    EFFECTS_HELD_BEHIND_RUNTIME (OMN-19091) watches a narrower, more direct
    fact than a container's docker status: whether the delegate-skill
    runtime-effects consumer group has any live (Stable, member-bearing)
    instance at all. Five measured occurrences on 2026-09-21 (18:47:53,
    19:55:09, 20:27:38, 20:57:50, 21:29:04, roughly six minutes each) were
    each detected only by a lane tripping over a delegation refusal, never by
    a monitor -- ``omninode-runtime-effects`` sits ``State=created`` for
    minutes behind a slow runtime health gate while its docker CONTAINER
    status alone gives no signal that delegation itself cannot proceed.
    """

    LAB_PASS_RECEIPT = "lab_pass_receipt"
    CONTAINER_RESTARTS = "container_restarts"
    CONSUMER_GROUP_LAG = "consumer_group_lag"
    EFFECTS_HELD_BEHIND_RUNTIME = "effects_held_behind_runtime"
    STALE_INDETERMINATE = "stale_indeterminate"


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

    As of OMN-19091 this is used by ``container_restarts`` alone --
    ``consumer_group_lag`` reads the broker in-process via aiokafka and needs
    no docker transport at all. The transport still belongs here rather than
    inside the one condition that uses it, so :class:`CommandRunner` stays the
    seam that condition is tested through.

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
    """One tick. Carries all five conditions whether or not anything fired."""

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
            # four of five conditions", which reads identically in the log.
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
#
# In-process via aiokafka, not ssh + docker exec + rpk (OMN-19091). rpk has
# exactly two credential-input mechanisms -- CLI ``-X user=/-X pass=`` flags
# (argv) or a persisted profile (disk) -- and a ruling of 2026-09-21 held
# that neither satisfies "no value on argv, disk, plist or log", transient or
# not. The identity is the SAME one the hook edge and ``onex delegate --lane
# dev`` already present (``~/.onex``, OPERATOR-CONSENT
# ``docs/tracking/ROLLING_WORK_LEDGER.md:3853`` item 7, satisfied by a live
# phase-2a proof rather than an ACL listing: all six declared groups read
# with zero ``GroupAuthorizationFailedError``). This also drops ssh and
# docker exec for this one condition entirely -- a moving part removed, not
# relocated.


class LaneCredentialError(Exception):
    """The dev lane identity in ``~/.onex`` could not be resolved."""


class GroupAuthorizationError(Exception):
    """The identity is connected but lacks DESCRIBE on this group."""


class GroupNeverCommittedError(Exception):
    """The group holds no committed offset on any partition.

    Not the same as zero lag: Operating Rule 16, an empty result is not
    evidence of absence. A group that has never committed could be freshly
    declared, or could be a name that no longer matches anything live --
    either way this is not a reading of "caught up".
    """


class GroupLagReader(Protocol):
    def __call__(self, group: str) -> int:
        """Return TOTAL-LAG for *group*, or raise one of the three above."""


def read_onex_lane_credential(onex_home: Path, lane: str) -> tuple[str, str]:
    """Resolve ``(sasl_username, sasl_password)`` for *lane* from ``~/.onex``.

    Same two files, same shape, same 0600-on-READ enforcement as
    ``omnibase_infra.cli.store_lane_credential.StoreLaneCredential`` --
    re-implemented here rather than imported, because importing anything
    under the ``omnibase_infra`` package from the bare brew interpreter this
    module runs under triggers that package's full ``__init__`` chain, which
    fails on this Mac today: the globally-installed ``omnibase_core`` in
    brew's site-packages is stale against what the local ``omnibase_infra``
    source tree imports (``ModuleNotFoundError`` on a model only the newer
    core carries). That skew is a pre-existing environment fact, not
    something this module fixes by reaching into global site-packages.

    ``config.yaml``'s ``lanes.<lane>`` block carries a principal NAME and a
    REFERENCE (``sasl_password_ref``), never a value -- the file is
    world-readable by default. The value lives only in ``credentials.json``,
    keyed by that reference, mode 0600.

    Raises:
        LaneCredentialError: either file is unreadable, malformed, the
            secrets file is not mode 0600, the lane is not declared, or the
            reference does not resolve. Never returns a partial credential.
    """
    config_path = onex_home / "config.yaml"
    creds_path = onex_home / "credentials.json"

    try:
        mode = stat.S_IMODE(creds_path.stat().st_mode)
    except OSError as exc:
        raise LaneCredentialError(f"{creds_path} could not be read: {exc}") from exc
    if mode != 0o600:
        raise LaneCredentialError(
            f"{creds_path} is mode {oct(mode)}, not 0600; refusing to read a "
            "secret file whose permissions have drifted"
        )

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LaneCredentialError(f"{config_path} could not be read: {exc}") from exc
    if not isinstance(config, dict):
        raise LaneCredentialError(f"{config_path} is not a mapping")
    lanes = config.get("lanes")
    entry = lanes.get(lane) if isinstance(lanes, dict) else None
    if not isinstance(entry, dict):
        held = ", ".join(sorted(lanes)) if isinstance(lanes, dict) else "none"
        raise LaneCredentialError(
            f"{config_path} declares no identity for lane {lane!r}; held: {held}"
        )
    username = entry.get("sasl_username")
    password_ref = entry.get("sasl_password_ref")
    if not isinstance(username, str) or not username:
        raise LaneCredentialError(f"{config_path} lane {lane!r} has no sasl_username")
    if not isinstance(password_ref, str) or not password_ref:
        raise LaneCredentialError(
            f"{config_path} lane {lane!r} has no sasl_password_ref"
        )

    try:
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaneCredentialError(f"{creds_path} could not be read: {exc}") from exc
    if not isinstance(creds, dict) or password_ref not in creds:
        raise LaneCredentialError(
            f"{creds_path} carries no value for reference {password_ref!r}"
        )
    password = creds[password_ref]
    if not isinstance(password, str) or not password:
        raise LaneCredentialError(
            f"{creds_path} reference {password_ref!r} is not a non-empty string"
        )
    return username, password


def make_kafka_lag_reader(
    *,
    bootstrap_servers: str,
    sasl_username: str,
    sasl_password: str,
    sasl_mechanism: str = "SCRAM-SHA-256",
    security_protocol: str = "SASL_PLAINTEXT",
    timeout_seconds: float = 30.0,
) -> GroupLagReader:
    """An in-process reader: committed offsets vs. partition end offsets.

    One connection per group rather than one shared session, deliberately --
    this alarm fires hourly against six groups, so the connection overhead is
    immaterial, and one group's failure staying fully isolated from the
    other five (the existing per-group resilience the ssh/rpk path already
    had) matters more than the extra round trips.

    The returned callable never raises a raw ``aiokafka`` exception: every
    failure is reclassified into one of :class:`GroupAuthorizationError`,
    :class:`GroupNeverCommittedError`, or a plain :class:`ValueError` naming
    only the exception's TYPE, never ``str(exc)`` -- an aiokafka error can
    carry the broker address and, on some paths, connection-string-shaped
    detail, and this alarm's whole discipline is that nothing it reads ever
    reaches a log, evidence string, or Slack message unfiltered.
    """
    from aiokafka import AIOKafkaConsumer
    from aiokafka.admin import AIOKafkaAdminClient
    from aiokafka.errors import GroupAuthorizationFailedError, KafkaError

    client_kwargs: dict[str, Any] = {
        "bootstrap_servers": bootstrap_servers,
        "security_protocol": security_protocol,
        "sasl_mechanism": sasl_mechanism,
        "sasl_plain_username": sasl_username,
        "sasl_plain_password": sasl_password,
        "request_timeout_ms": int(timeout_seconds * 1000),
    }

    async def _read(group: str) -> int:
        admin = AIOKafkaAdminClient(
            client_id="omninode-lab-alarm-admin", **client_kwargs
        )
        await admin.start()
        try:
            offsets = await admin.list_consumer_group_offsets(group)
        finally:
            await admin.close()

        if not offsets:
            raise GroupNeverCommittedError(
                f"group {group} holds no committed offset on any partition"
            )

        consumer = AIOKafkaConsumer(
            client_id="omninode-lab-alarm-consumer", **client_kwargs
        )
        await consumer.start()
        try:
            ends = await consumer.end_offsets(list(offsets.keys()))
        finally:
            await consumer.stop()

        total = 0
        for topic_partition, metadata in offsets.items():
            end = ends.get(topic_partition)
            if end is None:
                raise ValueError(
                    f"group {group} has no end offset for "
                    f"{topic_partition.topic}[{topic_partition.partition}]"
                )
            total += max(0, end - metadata.offset)
        return total

    def read(group: str) -> int:
        try:
            return asyncio.run(_read(group))
        except GroupNeverCommittedError:
            raise
        except GroupAuthorizationFailedError as exc:
            raise GroupAuthorizationError(
                f"group {group} refused DESCRIBE ({type(exc).__name__})"
            ) from None
        except KafkaError as exc:
            raise ValueError(
                f"group {group} unreadable ({type(exc).__name__})"
            ) from None
        except Exception as exc:  # noqa: BLE001 -- classify by type, never echo str(exc)
            raise ValueError(
                f"group {group} unreadable ({type(exc).__name__})"
            ) from None

    return read


def evaluate_consumer_group_lag(
    reader: GroupLagReader,
    groups: Sequence[str],
    *,
    previous: Mapping[str, int] | None,
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

    **A never-committed group is a distinct per-group STATE, not an unreadable
    one (corrected 2026-09-21 against live evidence).** Measured live on the
    dev lane: three of six declared groups (registration-writer,
    tenant-credentials-writer, tenant-registry-writer) have never committed
    and, as far as this alarm can tell, never will. Folding that into
    "unreadable" made the condition permanently INDETERMINATE on this lane's
    own steady state -- an alarm whose normal reading is "I can't tell" is
    read by nobody. A never-committed group is now named in the evidence and
    does NOT, by itself, block OK or force INDETERMINATE. Operating Rule 16
    still applies in full to a group this alarm genuinely cannot read at all
    -- an authorization refusal or a connection failure -- those keep forcing
    INDETERMINATE, unchanged. The one case a never-committed reading DOES
    raise: a group that held a real committed offset in the PREVIOUS sample
    and now reads never-committed has not gone quiet, its offsets have
    vanished -- that is an ALARM, not a shrug.

    Returns:
        The report, and this run's sample for the caller to persist. Only
        groups read with a real numeric offset this run are in the sample --
        a never-committed group is never persisted with a value, so "this
        group held a value in the previous sample" is exactly the
        already-vanished comparison above and needs no extra state field.

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
    never_committed: list[tuple[str, str]] = []
    for group in sorted(groups):
        try:
            sample[group] = reader(group)
        except GroupAuthorizationError as exc:
            unreadable.append(f"{group} ({exc})")
        except GroupNeverCommittedError as exc:
            never_committed.append((group, str(exc)))
        except ValueError as exc:
            unreadable.append(f"{group} ({exc})")

    readings = ", ".join(f"{g}={sample[g]}" for g in sorted(sample)) or "none"
    read_fact = (
        "TOTAL-LAG read in-process via aiokafka (committed offsets vs. "
        "partition end offsets), no ssh, no docker exec, no rpk"
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

    never_committed_note = (
        "; never-committed (informational, does not block OK): "
        + "; ".join(f"{g} ({r})" for g, r in sorted(never_committed))
        if never_committed
        else ""
    )

    if previous is None:
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.INDETERMINATE,
                evidence=(
                    f"{read_fact}; sampled {readings}; no previous sample, so "
                    "growth could not be evaluated and this run is the "
                    f"baseline{never_committed_note}"
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

    vanished: list[str] = []
    for group, _reason in sorted(never_committed):
        before = previous.get(group)
        if before is not None:
            vanished.append(f"{group}: held {before}, now never-committed")
            alarms.append(
                ModelAlarm(
                    condition=condition,
                    subject=group,
                    detail=(
                        f"consumer group {group} held a committed offset of "
                        f"{before} in the previous sample and now reads "
                        "never-committed -- its offsets appear to have been "
                        "wiped or the group deleted, not merely quiet"
                    ),
                )
            )

    evidence = (
        f"{read_fact}; {'; '.join(compared[: _EVIDENCE_SAMPLES * 8]) or 'none'}"
        f"{never_committed_note}"
        + (f"; VANISHED: {'; '.join(vanished)}" if vanished else "")
    )
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
# Condition 4 — the delegate-skill runtime-effects group held with zero
# live members (OMN-19091)
# ---------------------------------------------------------------------------
#
# Also in-process via aiokafka, same identity, same broker. The docker-status
# signal team-lead's own brief named first -- omninode-runtime-effects sitting
# in status=created with StartedAt at the zero time -- needs ssh and docker
# exec; the CONSUMER GROUP signal below needs neither and is the better one
# regardless, because it is what delegation actually depends on: a container
# that docker calls "created" and a container that has zero live members in
# its own consumer group are two different facts, and only the second one is
# "delegation cannot be processed right now".
#
# THE GROUP NAME IS VERSION-BEARING, DELIBERATELY NOT PINNED LITERALLY.
# ``local.omnimarket.node_delegate_skill_orchestrator.consume.{version}.__i.
# runtime-effects.__t.onex.cmd.omnimarket.delegate-skill.v1`` embeds the
# node's own contract_version, which this session watched drift from 1.2.0 to
# 1.3.0 in the ordinary course of a redeploy -- scripts/runtime_build/
# declared_consumer_groups.py's own docstring names exactly this failure mode
# ("a stale name is what happens the moment a contract version is bumped").
# This condition therefore matches by the STABLE prefix/suffix around that
# version segment and asks "does ANY matching group currently have a live
# member", not "does group X specifically". Both the retired 1.2.0 group
# (state Empty, zero members) and the live 1.3.0 one (state Stable, members)
# coexisted on the broker while this was built and tested live.
#
# THE DESCRIBE-GROUPS BATCHING BUG, MEASURED LIVE, 2026-09-21. aiokafka
# 0.14.0's ``describe_consumer_groups`` against this Redpanda broker decodes
# correctly for exactly one group_id per call and raises
# ``ValueError: Buffer underrun decoding string`` (surfacing as
# ``KafkaConnectionError``) the moment more than one group_id is batched into
# a single call -- reproduced three times, deterministic on batch size, not on
# which groups. The per-group entries in a successful response are also PLAIN
# POSITIONAL TUPLES, not an attribute-bearing struct, despite what their own
# ``__repr__`` prints. The reader below therefore calls
# ``describe_consumer_groups`` once per candidate group and unpacks each
# result by position; batching the discovered candidates into one call is the
# exact defect this comment exists to keep out.


class EffectsGroupReadError(Exception):
    """The consumer-group listing or per-group describe could not be read."""


class EffectsHeldReader(Protocol):
    def __call__(self) -> tuple[tuple[str, str, int], ...]:
        """Return ``(group_name, state, member_count)`` for every group
        matching the declared prefix/suffix, resolved live. Raises
        :class:`EffectsGroupReadError` on any failure -- never returns a
        partial or placeholder reading.
        """


def make_effects_held_reader(
    *,
    bootstrap_servers: str,
    sasl_username: str,
    sasl_password: str,
    group_prefix: str,
    group_suffix: str,
    sasl_mechanism: str = "SCRAM-SHA-256",
    security_protocol: str = "SASL_PLAINTEXT",
    timeout_seconds: float = 30.0,
) -> EffectsHeldReader:
    """An in-process reader: every group matching prefix/suffix, with its
    live state and member count.

    Never raises a raw ``aiokafka`` exception: every failure is reclassified
    into :class:`EffectsGroupReadError` naming only the exception's TYPE,
    never ``str(exc)`` -- the same discipline the lag reader above follows,
    for the same reason.
    """
    from aiokafka.admin import AIOKafkaAdminClient
    from aiokafka.errors import KafkaError

    client_kwargs: dict[str, Any] = {
        "bootstrap_servers": bootstrap_servers,
        "security_protocol": security_protocol,
        "sasl_mechanism": sasl_mechanism,
        "sasl_plain_username": sasl_username,
        "sasl_plain_password": sasl_password,
        "request_timeout_ms": int(timeout_seconds * 1000),
    }

    async def _read() -> tuple[tuple[str, str, int], ...]:
        admin = AIOKafkaAdminClient(
            client_id="omninode-lab-alarm-effects-admin", **client_kwargs
        )
        await admin.start()
        try:
            groups = await admin.list_consumer_groups()
            names = (g[0] if isinstance(g, tuple) else g for g in groups)
            candidates = [
                name
                for name in names
                if name.startswith(group_prefix) and name.endswith(group_suffix)
            ]
            results: list[tuple[str, str, int]] = []
            for name in candidates:
                # ONE group per describe call -- see the module comment above
                # this section: batching more than one group_id here corrupts
                # the decode on this broker, measured live and reproduced.
                responses = await admin.describe_consumer_groups([name])
                for response in responses:
                    for entry in response.groups:
                        # Positional tuple, not an attribute-bearing struct:
                        # (error_code, group, state, protocol_type, protocol,
                        # members). Indexed, not unpacked by name, because
                        # aiokafka's own repr prints field names without
                        # making them real attributes.
                        results.append((entry[1], entry[2], len(entry[5])))
            return tuple(results)
        finally:
            await admin.close()

    def read() -> tuple[tuple[str, str, int], ...]:
        try:
            return asyncio.run(_read())
        except KafkaError as exc:
            raise EffectsGroupReadError(f"unreadable ({type(exc).__name__})") from None
        except Exception as exc:  # noqa: BLE001 -- classify by type, never echo str(exc)
            raise EffectsGroupReadError(f"unreadable ({type(exc).__name__})") from None

    return read


#: How long the group may hold at zero live members before this alarms.
#: Team-lead's own bound: "for more than 60 s". Sixty seconds is well inside
#: even the shortest measured occurrence (roughly six minutes) and well
#: outside the couple of seconds a normal rebalance takes, so it does not fire
#: on the transition itself.
EFFECTS_HELD_AFTER = timedelta(seconds=60)


@dataclass(frozen=True)
class ModelRecoveryNotice:
    """A subject LEFT a bad state, and how long it was in it.

    Not a :class:`ModelAlarm`: recovery is not itself a finding a person must
    act on, but the window it names is durable evidence for sizing a blast
    radius after the fact -- exactly what the OMN-18843 occurrences were
    missing, each one found only by a lane tripping over a live refusal.
    """

    condition: EnumAlarmCondition
    subject: str
    detail: str

    def __post_init__(self) -> None:
        if not self.subject:
            raise ValueError("a recovery notice must name its subject")
        if not self.detail:
            raise ValueError(
                "a recovery notice must carry a detail a reader can act on"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition.value,
            "subject": self.subject,
            "detail": self.detail,
        }


def evaluate_effects_held_behind_runtime(
    reader: EffectsHeldReader,
    *,
    state: ModelAlarmState,
    now: str,
    threshold: timedelta = EFFECTS_HELD_AFTER,
) -> tuple[ModelConditionReport, ModelRecoveryNotice | None]:
    """ALARM when NO group matching the declared prefix/suffix is Stable
    with a live member, for at least *threshold*.

    Edge-triggered like every other condition here, tracked via
    ``state.effects_held_since`` (one subject: ``"runtime-effects"``). Three
    outcomes exactly as team-lead specified: OK while live or while held under
    the threshold, ALARM once held at or past it, INDETERMINATE ONLY when the
    read itself is refused -- unlike the lag condition, a short hold is not
    itself indeterminate, because the bound IS the definition of "held" here,
    not a comparison this run cannot yet make.
    """
    condition = EnumAlarmCondition.EFFECTS_HELD_BEHIND_RUNTIME
    key = "runtime-effects"

    try:
        candidates = reader()
    except EffectsGroupReadError as exc:
        state.effects_held_since.pop(key, None)
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.INDETERMINATE,
                evidence=(
                    f"the delegate-skill runtime-effects group could not be read: {exc}"
                ),
            ),
            None,
        )

    if not candidates:
        state.effects_held_since.pop(key, None)
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.INDETERMINATE,
                evidence=(
                    "no group matched the declared delegate-skill "
                    "runtime-effects prefix/suffix; a renamed or undeclared "
                    "group is not a live one"
                ),
            ),
            None,
        )

    readings = "; ".join(
        f"{name}: {group_state}/{members}" for name, group_state, members in candidates
    )
    live = any(
        group_state == "Stable" and members > 0
        for _name, group_state, members in candidates
    )

    if live:
        recovery = None
        since = state.effects_held_since.pop(key, None)
        if since is not None:
            held_for = datetime.fromisoformat(now) - datetime.fromisoformat(since)
            recovery = ModelRecoveryNotice(
                condition=condition,
                subject=key,
                detail=(
                    "the delegate-skill runtime-effects group is Stable with "
                    f"a live member again, after {held_for} held with none "
                    f"(since {since})"
                ),
            )
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.OK,
                evidence=f"{readings}; at least one candidate is Stable with a live member",
            ),
            recovery,
        )

    since = state.effects_held_since.get(key, now)
    state.effects_held_since[key] = since
    held_for = datetime.fromisoformat(now) - datetime.fromisoformat(since)

    if held_for < threshold:
        return (
            ModelConditionReport(
                condition=condition,
                outcome=EnumConditionOutcome.OK,
                evidence=(
                    f"{readings}; zero live members since {since} "
                    f"({held_for}), under the {threshold} bound"
                ),
            ),
            None,
        )

    detail = (
        f"the delegate-skill runtime-effects group has had zero live "
        f"(Stable, member-bearing) instances for {held_for} (since {since}) "
        "-- delegation cannot be processed while this holds, whatever the "
        "container's own docker status says"
    )
    return (
        ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.ALARM,
            evidence=f"{readings}; {detail}",
            alarms=(ModelAlarm(condition=condition, subject=key, detail=detail),),
        ),
        None,
    )


# ---------------------------------------------------------------------------
# The consent gate a future sender must pass
# ---------------------------------------------------------------------------

#: A consent row in the CANONICAL shape Operating Rule 18 specifies.
_CONSENT_ROW_LABELLED = re.compile(
    r"OPERATOR-CONSENT.*?APPROVED SCOPE:(?P<approved>.*?)\|"
    r"\s*OUT OF SCOPE:(?P<out>.*?)(\||$)",
    re.IGNORECASE,
)

#: A consent row recorded as a dated RULING instead.
#:
#: Rule 18's canonical row carries two labelled lists. The row this alarm is
#: actually authorized by does not -- it is a RULING carrying the literal
#: OPERATOR-CONSENT token, an ``approved_by=`` naming the operator with the
#: timestamp and their verbatim words, the destination channel, the credential
#: by reference, and its exclusions in prose.
#:
#: **The resolver requires the SUBSTANCE and not the two label strings**, and
#: that is a deliberate call rather than an oversight. What the OUT OF SCOPE
#: half exists to do is bound the grant; here the named channel IS the bound,
#: and the exclusion clause states the rest. A gate that refused a real,
#: dated, operator-attributed, channel-naming authorization over two missing
#: labels would be a spelling test rather than an authorization control, and
#: the next lane would route around it.
#:
#: What still refuses, and is proven by its own control: a row with no
#: approver, a row naming no channel, a row naming a DIFFERENT channel, a row
#: with no exclusion clause at all, and an unreadable ledger.
_CONSENT_ROW_RULING = re.compile(r"OPERATOR-CONSENT(?P<body>.*)", re.IGNORECASE)
_APPROVED_BY = re.compile(
    r"approved_by\s*=\s*(?P<who>[A-Za-z0-9_@.:-]+)", re.IGNORECASE
)
_EXCLUSION = re.compile(r"\bno\s+new\b|\bno\s+webhook\b|OUT OF SCOPE:", re.IGNORECASE)


@dataclass(frozen=True)
class ModelPostingConsent:
    """A resolved, citable authorization to send this alarm somewhere."""

    channel: str
    ledger_path: str
    line: int
    approved_by: str

    @property
    def citation(self) -> str:
        return f"{self.ledger_path}:{self.line}"


def resolve_posting_consent(
    ledger_path: Path, *, channel: str
) -> ModelPostingConsent | None:
    """Return the consent row authorizing a send to *channel*, or ``None``.

    **No agent message is consent.** Not a dispatch brief, not a peer lane's
    claim, not this ticket and not the plan. The only thing that authorizes a
    send is a durable row in the ledger, and this function is the only place
    that decides whether one is present.

    Two accepted shapes, both requiring the same four substantive facts: the
    OPERATOR-CONSENT token, an ``approved_by``, the destination channel, and an
    explicit exclusion. See :data:`_CONSENT_ROW_RULING` for why the second
    shape is accepted.

    An unreadable ledger returns ``None``: a grant that cannot be read is not
    a grant.
    """
    try:
        lines = ledger_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    for number, line in enumerate(lines, start=1):
        labelled = _CONSENT_ROW_LABELLED.search(line)
        if labelled is not None and labelled.group("out").strip():
            if channel.lower() in labelled.group("approved").lower():
                who = _APPROVED_BY.search(line)
                return ModelPostingConsent(
                    channel=channel,
                    ledger_path=str(ledger_path),
                    line=number,
                    approved_by=who.group("who") if who else "operator",
                )
            continue

        ruling = _CONSENT_ROW_RULING.search(line)
        if ruling is None:
            continue
        body = ruling.group("body")
        who = _APPROVED_BY.search(body)
        if who is None:
            continue
        if channel.lower() not in body.lower():
            continue
        if _EXCLUSION.search(body) is None:
            continue
        return ModelPostingConsent(
            channel=channel,
            ledger_path=str(ledger_path),
            line=number,
            approved_by=who.group("who"),
        )
    return None


class PostingError(RuntimeError):
    """Delivery was attempted and did not succeed."""


def read_secret(env_file: Path, name: str) -> str:
    """Read ONE named secret out of an env file, by NAME.

    The value is returned and never logged, never written to the run record,
    never put on a command line and never included in an error message -- the
    caller's failures name the VARIABLE and the file, which is the part a
    reader can act on. This mirrors how the lab's existing reporter reaches the
    same credential.

    An env file rather than the process environment because this runs under
    launchd, which inherits no shell, so the token is not in the agent's
    environment at all and a design that assumed it would be silently sends
    nothing.
    """
    try:
        for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, _, value = stripped.partition("=")
            if key.strip() == name:
                return value.strip().strip("'\"")
    except OSError as exc:
        raise PostingError(f"could not read {env_file} for {name}") from exc
    raise PostingError(f"{env_file} declares no {name}")


def _post_text_to_slack(
    text: str,
    *,
    consent: ModelPostingConsent,
    env_file: Path,
    timeout_seconds: float,
) -> str:
    """Deliver ONE already-rendered message to the consented channel.

    Shared by :func:`post_alarm` and :func:`post_recovery` -- the token
    handling, the channel, and the failure discipline are identical for both;
    only the rendered text differs.

    Raises:
        PostingError: the send did not succeed. A failed send is never
            swallowed: a message that believes it delivered and did not is
            the silent failure this whole ticket is about.
    """
    import json as _json
    import urllib.request

    token = read_secret(env_file, SLACK_TOKEN_VAR)
    payload = _json.dumps({"channel": consent.channel, "text": text}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed https endpoint
        SLACK_POST_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - SLACK_POST_URL is a fixed https constant
            request, timeout=timeout_seconds
        ) as response:
            body = _json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        raise PostingError(f"delivery to {consent.channel} failed: {exc}") from exc
    if not body.get("ok"):
        raise PostingError(
            f"delivery to {consent.channel} refused: {body.get('error', 'unknown')}"
        )
    return str(body.get("ts", ""))


def post_alarm(
    alarm: ModelAlarm,
    *,
    consent: ModelPostingConsent,
    env_file: Path,
    timeout_seconds: float = 15.0,
) -> str:
    """Deliver ONE alarm to the consented channel. Returns the message ts."""
    text = (
        f":rotating_light: lab alarm — {alarm.condition.value}\n"
        f"*{alarm.subject}*\n{alarm.detail}\n"
        f"_authorized by {consent.citation}_"
    )
    return _post_text_to_slack(
        text, consent=consent, env_file=env_file, timeout_seconds=timeout_seconds
    )


def post_recovery(
    notice: ModelRecoveryNotice,
    *,
    consent: ModelPostingConsent,
    env_file: Path,
    timeout_seconds: float = 15.0,
) -> str:
    """Deliver ONE recovery notice to the consented channel. Returns the ts.

    OMN-19091: the window length lives in ``notice.detail`` already, so a
    reader does not have to reconstruct "how long was this actually down"
    from two separate messages by hand.
    """
    text = (
        f":white_check_mark: lab alarm cleared — {notice.condition.value}\n"
        f"*{notice.subject}*\n{notice.detail}\n"
        f"_authorized by {consent.citation}_"
    )
    return _post_text_to_slack(
        text, consent=consent, env_file=env_file, timeout_seconds=timeout_seconds
    )


# ---------------------------------------------------------------------------
# State: edge-triggering and the previous lag sample
# ---------------------------------------------------------------------------


@dataclass
class ModelAlarmState:
    """What the previous tick left behind."""

    active: dict[str, str] = field(default_factory=dict)
    lag_sample: dict[str, int] | None = None
    #: OMN-19091: per-condition-key -> the ISO timestamp it FIRST read
    #: INDETERMINATE, contiguously. Absent the moment that condition reads a
    #: real verdict (OK or ALARM) again -- see evaluate_stale_indeterminate.
    indeterminate_since: dict[str, str] = field(default_factory=dict)
    #: OMN-19091: per-subject -> the ISO timestamp the effects-held-behind-
    #: runtime condition FIRST read zero live members, contiguously. Absent
    #: the moment a live member is read again -- see
    #: evaluate_effects_held_behind_runtime.
    effects_held_since: dict[str, str] = field(default_factory=dict)

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
        since = payload.get("indeterminate_since")
        held_since = payload.get("effects_held_since")
        return cls(
            active={
                str(k): str(v) for k, v in active.items() if isinstance(active, dict)
            }
            if isinstance(active, dict)
            else {},
            lag_sample={str(k): int(v) for k, v in sample.items()}
            if isinstance(sample, dict)
            else None,
            indeterminate_since={str(k): str(v) for k, v in since.items()}
            if isinstance(since, dict)
            else {},
            effects_held_since={str(k): str(v) for k, v in held_since.items()}
            if isinstance(held_since, dict)
            else {},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "active": self.active,
                    "lag_sample": self.lag_sample,
                    "indeterminate_since": self.indeterminate_since,
                    "effects_held_since": self.effects_held_since,
                },
                indent=2,
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
# Condition 4 — a condition stuck INDETERMINATE for too long (OMN-19091)
# ---------------------------------------------------------------------------

#: A condition that has read INDETERMINATE on every sample for a full day has
#: not told anyone anything for a full day. Chosen to be well past the hourly
#: cadence's own noise floor (a single bad tick, a lab restart) while still
#: inside the window a person checks in on the lab.
STALE_INDETERMINATE_AFTER = timedelta(hours=24)


def evaluate_stale_indeterminate(
    reports: Sequence[ModelConditionReport],
    state: ModelAlarmState,
    *,
    now: str,
    threshold: timedelta = STALE_INDETERMINATE_AFTER,
) -> ModelConditionReport:
    """A condition INDETERMINATE on every sample for >= *threshold* is blind.

    Today, absent this condition, that state is silent forever: INDETERMINATE
    never posts (only a newly raised ALARM does), and nothing tracks how long
    a condition has been reading it. An alarm that cannot tell is not the same
    as an alarm that says nothing is wrong, and the difference is exactly the
    one Operating Rule 16 exists to preserve -- so it gets its own condition
    rather than a silent gap in the other three.

    This condition watches the OTHER THREE reports this same run produced; it
    never watches its own prior outcome, so it cannot go stale watching
    itself. Its own outcome is always OK or ALARM, never INDETERMINATE --
    "is a condition stuck" is always answerable from state plus the clock.

    Cleared per-condition the moment that condition reads a real verdict
    again, OK or ALARM either one: a real reading, of either shape, proves the
    alarm is watching, which is the whole thing this condition checks for.
    """
    condition = EnumAlarmCondition.STALE_INDETERMINATE
    now_dt = datetime.fromisoformat(now)
    updated: dict[str, str] = {}
    stale: list[str] = []
    readings: list[str] = []

    for report in reports:
        key = report.condition.value
        if report.outcome is not EnumConditionOutcome.INDETERMINATE:
            readings.append(f"{key}: {report.outcome.value}")
            continue
        since = state.indeterminate_since.get(key, now)
        updated[key] = since
        age = now_dt - datetime.fromisoformat(since)
        readings.append(f"{key}: INDETERMINATE since {since} (age {age})")
        if age >= threshold:
            stale.append(key)

    state.indeterminate_since = updated
    evidence = f"condition ages this run: {'; '.join(readings) or 'none'}"

    if stale:
        return ModelConditionReport(
            condition=condition,
            outcome=EnumConditionOutcome.ALARM,
            evidence=evidence,
            alarms=tuple(
                ModelAlarm(
                    condition=condition,
                    subject=key,
                    detail=(
                        f"condition {key} has read INDETERMINATE on every "
                        f"sample for at least {threshold}, with no OK and no "
                        "ALARM to clear it -- an alarm that cannot tell is "
                        "not the same as one that found nothing wrong"
                    ),
                )
                for key in stale
            ),
        )
    return ModelConditionReport(
        condition=condition, outcome=EnumConditionOutcome.OK, evidence=evidence
    )


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
    stdlib-only. A YAML config here would put a third-party import on a path
    that has to run before the alarm can even discover what it is bounded to
    load. This predates, and is unrelated to, the ``yaml``/``aiokafka``
    import OMN-19091 added for the consumer-lag identity read -- config
    PARSING stays JSON; nothing about how the config is spelled changed.
    """

    repo: str
    lane: EnumLabLane
    ready_url: str
    agent_url: str
    kafka_bootstrap_servers: str
    docker_command: tuple[str, ...]
    container_restart_bounds: dict[str, int]
    consumer_groups: tuple[str, ...]
    #: OMN-19091: the stable prefix/suffix bracketing the delegate-skill
    #: runtime-effects group's volatile contract-version segment. See the
    #: module comment above evaluate_effects_held_behind_runtime for why this
    #: is a prefix/suffix match rather than one pinned literal.
    effects_group_prefix: str
    effects_group_suffix: str

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
            kafka_bootstrap_servers=expand_env(
                str(payload["kafka_bootstrap_servers"]), source=path
            ),
            docker_command=tuple(expand_env(str(p), source=path) for p in docker),
            container_restart_bounds={str(k): int(v) for k, v in bounds.items()},
            consumer_groups=tuple(str(g) for g in groups),
            effects_group_prefix=str(payload["effects_group_prefix"]),
            effects_group_suffix=str(payload["effects_group_suffix"]),
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
    lag_reader: GroupLagReader,
    effects_reader: EffectsHeldReader,
    posting_channel: str,
    env_file: Path,
) -> ModelAlarmRun:
    """Evaluate all five conditions, record the run, return it."""
    started = _now()
    state = ModelAlarmState.load(state_dir / "state.json")

    receipt_report = evaluate_lab_pass_receipt(
        repo=config.repo, lane=config.lane, sha=sha, reader=receipt_reader
    )
    restart_report = evaluate_container_restarts(
        config.container_restart_bounds, runner=runner
    )
    lag_report, lag_sample = evaluate_consumer_group_lag(
        lag_reader, config.consumer_groups, previous=state.lag_sample
    )
    effects_report, recovery = evaluate_effects_held_behind_runtime(
        effects_reader, state=state, now=started
    )
    watched_reports = (receipt_report, restart_report, lag_report, effects_report)
    stale_report = evaluate_stale_indeterminate(watched_reports, state, now=started)

    reports = (*watched_reports, stale_report)
    raised = select_new_alarms(reports, state, now=started)
    state.lag_sample = lag_sample
    state.save(state_dir / "state.json")

    consent = resolve_posting_consent(ledger_path, channel=posting_channel)
    if consent is None:
        posting = (
            f"disabled: no OPERATOR-CONSENT row in {ledger_path} authorizes "
            f"{posting_channel}"
        )
    elif not raised and recovery is None:
        posting = (
            f"authorized by {consent.citation} (approved_by={consent.approved_by}); "
            "nothing new to deliver this run"
        )
    else:
        # Delivery is per NEWLY RAISED alarm, so the edge-triggering above is
        # also the deduplication of the channel: a condition that stays bad
        # posts once, not once an hour. A recovery notice rides the SAME
        # consent and the same channel, delivered alongside any alarms this
        # run also raised.
        delivered: list[str] = []
        failures: list[str] = []
        for alarm in raised:
            try:
                delivered.append(
                    f"{alarm.subject}@{post_alarm(alarm, consent=consent, env_file=env_file)}"
                )
            except PostingError as exc:
                failures.append(str(exc))
        recovered: list[str] = []
        if recovery is not None:
            try:
                recovered.append(
                    f"{recovery.subject}@{post_recovery(recovery, consent=consent, env_file=env_file)}"
                )
            except PostingError as exc:
                failures.append(str(exc))
        posting = (
            f"authorized by {consent.citation} (approved_by={consent.approved_by}); "
            f"delivered {len(delivered)}/{len(raised)} alarms to {consent.channel}"
            + (f", {len(recovered)}/1 recovery notice" if recovery is not None else "")
            + (
                f"; delivery ids {', '.join(delivered + recovered)}"
                if delivered or recovered
                else ""
            )
            + (f"; FAILED: {'; '.join(failures)}" if failures else "")
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
        default="#omninode-notifications",
        help="The channel a consent row must name. Nothing is sent without one.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".omnibase" / ".env",
        help=(
            f"File declaring {SLACK_TOKEN_VAR}. Read by NAME at send time; the "
            "value is never logged, recorded or passed on a command line."
        ),
    )
    parser.add_argument(
        "--onex-home",
        type=Path,
        default=Path.home() / ".onex",
        help=(
            "The ~/.onex client store the consumer-lag condition reads its "
            f"dev-lane identity from (lane {_ONEX_LANE!r}) -- the same store "
            "the hook edge and 'onex delegate --lane dev' already read."
        ),
    )
    return parser


def _resolve_onex_credential_or_reason(
    onex_home: Path,
) -> tuple[str, str] | str:
    """Resolve the dev-lane identity ONCE for both aiokafka readers below.

    Returns ``(username, password)`` on success, or the failure REASON as a
    plain string on failure -- never raises, so a credential problem never
    crashes the timer before the other conditions are evaluated.
    """
    try:
        return read_onex_lane_credential(onex_home, _ONEX_LANE)
    except LaneCredentialError as exc:
        return str(exc)


def _build_lag_reader(
    credential: tuple[str, str] | str, *, bootstrap_servers: str
) -> GroupLagReader:
    """Bind a lag reader to an already-resolved credential (or its failure).

    A credential resolution failure does not crash the timer: it returns a
    reader that fails EVERY group with the same named reason, so the
    consumer-lag condition reads INDETERMINATE with a reason a person can act
    on -- 'fix ~/.onex' -- instead of the whole run dying before the other
    conditions are evaluated.
    """
    if isinstance(credential, str):
        reason = credential

        def failing_reader(group: str) -> int:
            raise ValueError(
                f"group {group} unreadable: identity unresolved ({reason})"
            )

        return failing_reader

    username, password = credential
    return make_kafka_lag_reader(
        bootstrap_servers=bootstrap_servers,
        sasl_username=username,
        sasl_password=password,
    )


def _build_effects_reader(
    credential: tuple[str, str] | str,
    *,
    bootstrap_servers: str,
    group_prefix: str,
    group_suffix: str,
) -> EffectsHeldReader:
    """Bind an effects-held reader to an already-resolved credential.

    Same fail-soft shape as :func:`_build_lag_reader`: a credential failure
    returns a reader that refuses with a named reason rather than raising.
    """
    if isinstance(credential, str):
        reason = credential

        def failing_reader() -> tuple[tuple[str, str, int], ...]:
            raise EffectsGroupReadError(f"identity unresolved ({reason})")

        return failing_reader

    username, password = credential
    return make_effects_held_reader(
        bootstrap_servers=bootstrap_servers,
        sasl_username=username,
        sasl_password=password,
        group_prefix=group_prefix,
        group_suffix=group_suffix,
    )


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

    credential = _resolve_onex_credential_or_reason(args.onex_home)
    run = run_once(
        config,
        state_dir=args.state_dir,
        ledger_path=args.ledger,
        sha=sha,
        receipt_reader=read_latest_receipt,
        runner=make_runner(config.docker_command),
        lag_reader=_build_lag_reader(
            credential, bootstrap_servers=config.kafka_bootstrap_servers
        ),
        effects_reader=_build_effects_reader(
            credential,
            bootstrap_servers=config.kafka_bootstrap_servers,
            group_prefix=config.effects_group_prefix,
            group_suffix=config.effects_group_suffix,
        ),
        posting_channel=args.posting_channel,
        env_file=args.env_file,
    )
    sys.stdout.write(render(run) + "\n")
    return exit_code(run)


if __name__ == "__main__":
    raise SystemExit(main())
