# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The .201 dev instance converges on the omnimarket dev head when idle (OMN-19509).

WHY THIS EXISTS
---------------
Task B7 of the second-deploy-slot plan (knowledge-base-internal
``beta/plans/2026-09-24-second-deploy-slot-plan.md``, epic OMN-19500). Once the
routing table (``config/deploy_lane_routing.yaml``, OMN-19506) sends
omnimarket-requested rebuilds to dev-202, the .201 lane vendors a newer
omnimarket only when an omnibase_infra rebuild stages omnimarket at dev head.
The chain canary (C15) and the dev-lane corpus probe only .201, and staging
delivery requires a fresh green C15, so in an omnimarket-heavy period C15 would
grade older omnimarket code than the code being delivered. So the .201 instance
starts ONE rebuild at dev head on its own, as ``agent/idle-converge``, when it
has run no job for 16 minutes and its running omnimarket revision is not the
omnimarket dev head.

WHAT THE MODEL FORCED (DeployRoutingIdle.tla, TLC logs on OMN-19509)
--------------------------------------------------------------------
The design configuration holds the B3 routing invariants plus
SingleWriter201, NoConvergeAtProbe, FreshAtProbe, QueuedFirst and
OneConvergePerHead. Each of five mutants breaks exactly one of them:

* **A converge outside the poll loop** (a timer of its own) ran beside a routed
  job: SingleWriter201. So the agent calls :func:`decide` only from the loop's
  idle branch, and a job in flight is a refusal here as well.
* **No queue check** started a converge while a routed command was already
  waiting: QueuedFirst. The idle branch is reached only when the poll returned
  no command, which is that check.
* **The ticket's 10-minute margin alone** let a converge that started 11
  minutes before C15 still be running when C15 started: NoConvergeAtProbe. The
  start exclusion is the converge CEILING plus the margin
  (:data:`CONVERGE_CEILING` + :data:`PROBE_MARGIN`).
* **A guard that only looks forward** started a converge under a C15 that was
  already running: NoConvergeAtProbe again. The exclusion also covers each
  probe's own run, to its declared ``max_duration_minutes``.
* **No converge at all** left .201 behind an omnimarket merge carried by dev-202
  alone at the next C15: FreshAtProbe. That is the reason for this module.

The probe schedule is ``config/lab_probe_windows.yaml``, read through the one
parser the probe-window gate uses (``scripts/ci/check_lab_probe_windows.py``),
from the agent's own clone. An unreadable schedule is a refusal, never a clear
window.

INERT UNTIL A ROUTE EXISTS
--------------------------
The converge fires only when the loaded routing table sends omnimarket to
another instance. The committed table routes nothing to dev-202, so on .201
nothing changes until the (dev, omnimarket) -> dev-202 row lands (task B8).

THE SAME RULE, FOR omnibase_infra (OMN-20867)
---------------------------------------------
The rule is per repository, over :data:`FOLLOWED_REPOSITORIES`: an instance
follows the dev head of every followed repository whose rebuilds the table
routes to another instance. On the default instance that is omnimarket alone,
as before (omnibase_infra routes to the default itself). On an instance whose
receipt proves omnimarket only, it is omnibase_infra: its rebuilds go to the
default instance by operator ruling, so such an instance otherwise took an
omnibase_infra build only when an omnimarket merge happened to rebuild it, and
ran a revision without a merged runtime change for as long as omnimarket was
quiet. A converge is not a merge: it carries no
``gha/`` requester, no per-merge verify job runs for it and no lab-pass receipt
is emitted for it, and an instance receipt counts only for the repositories its
``proves`` names, so this changes no repository's lab-pass evidence.

A LANE WITH NO RUNNING RUNTIME IS RESTORED (OMN-20864)
-----------------------------------------------------
The converge rule above reads the running revision out of the runtime
container, so with no runtime running it could only refuse
(``running_unreadable``), and it did, every minute, from 14:14Z to 19:41Z on
2026-10-11 while the lab dev lane sat with ``omninode-runtime`` absent and
its replacement stuck in Created. A lane that is down is not "up to date" and
not "unreadable"; it is down. :func:`decide_restore` is a separate rule,
evaluated first in the same idle branch: when the lane's main runtime
container is absent or stopped, the instance runs ONE runtime rebuild
(:func:`restore_command`, requested as ``agent/idle-restore``) to bring it up.
It keeps every bound that protects a running lane or a waiting command -- the
idle branch only, no job in flight, the lane's FROZEN flag, the load gate --
and adds two of its own: the lane must have been quiet for
:data:`RESTORE_AFTER`, and a restore is tried at most once per
:data:`RESTORE_RETRY_AFTER`, so a lane that will not come up costs one rebuild
an hour, not one a minute. It ignores routing (a missing runtime is missing on
every instance) and the probe windows, which protect a probe from a lane being
rebuilt under it; a lane with no runtime fails the probe either way.

A LAGGING LANE HAS A DECLARED BOUND, AND IS LOUD (OMN-21146)
-----------------------------------------------------------
The probe windows above cost more than they protect. C15 runs every two hours
for a declared 65 minutes, so with the 40-minute lead each run excludes 105
minutes of every 120, and a merge that lands in one waits for the 15 minutes
between. On 2026-10-15 omnimarket#3964 merged at 17:07Z, inside the 17:41Z
run's exclusion; the default instance's agent accepted its converge at 18:46:52Z, the
exclusion's end, and dod-verify refused merged_pr fleet-wide on the old image
until the lane was rebuilt, with nothing saying the lane was behind.

So an instance may declare ``flags.max_dev_lag_minutes`` in the routing table
(``ModelLaneFlags.max_dev_lag``). The agent records when it first saw the lane
behind a followed repository's dev head (:func:`next_behind_since`); once that
is as old as the bound the lane is OVERDUE, and :func:`decide` no longer
refuses it for the idle period (``recently_active``), an unreadable probe
schedule or a probe window. A probe graded against code older than the code
being delivered is the failure FreshAtProbe names; past the declared bound, the
lane's freshness wins. Every other bound holds: no job in flight, the FROZEN
flag, the load gate, and one converge per head. And the lag is loud:
:func:`dev_lane_drift` is the ``ModelDevLaneDrift`` the agent publishes and
serves at ``GET /dev-lane-drift``. An instance with no bound is unchanged.

THE STABILITY LANE FOLLOWS THE TAG DEV RUNS (OMN-19998)
------------------------------------------------------
A dev agent whose fence also admits the stability lane
(``lane_policy.follows_stability``) keeps it on dev's tag. On the same idle
tick that follows the newest omnimarket dev tag for the dev lane, once the
dev lane runs that tag (or a commit that contains it) and the stability lane
does not, the agent runs one stability-lane rebuild named
:data:`STABILITY_FOLLOW_REQUESTER` (:func:`decide_stability_follow`,
:func:`stability_follow_command`). Stability therefore builds only a tag dev
already runs, and trails dev by at most the one tag in flight. One attempt per
tag; the instance's freeze flag and load gate hold it like any converge.

THE LAB FOLLOWS omnimarket's DEV TAGS, NOT ITS RELEASES (OMN-19998)
------------------------------------------------------------------
Operator RULING 2026-10-16T09:35:49Z: in the lab, omnimarket deploys from a
tag cut on each dev merge, not from a PyPI release, and that is part of the
process rather than something somebody remembers. omnimarket cuts a
lightweight ``dev-<UTC YYYYMMDDHHMMSS>-<sha12>`` on every push to dev. For
each repository in ``tracking_ref.DEV_TAG_FOLLOWED_REPOSITORIES`` the head
this module follows is the newest such tag, read with ``git ls-remote``
against the canonical clone's origin (:func:`read_dev_tags`), and a lab build
stages that tag (``deploy_source_ref.py --follow-dev-tag``). Until origin
carries a first tag the dev branch head is followed, as before.

Every instance also beats against that tag, whatever the routing table says:
:func:`dev_tag_heartbeat` is the ``ModelDevTagHeartbeat`` the agent publishes
every :data:`DEV_TAG_HEARTBEAT_INTERVAL`, and its ``alarm`` is raised once the
lane has run behind the newest tag for :data:`DEV_TAG_ALARM_AFTER`
(:func:`dev_tag_behind_since`).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Final
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelDevLaneDrift,
    ModelDevTagHeartbeat,
    ModelRebuildRequested,
    Scope,
)

logger = logging.getLogger(__name__)

#: ``requested_by`` of every idle converge, so its job record and its terminal
#: event are never mistaken for a merge's.
IDLE_CONVERGE_REQUESTER: Final = "agent/idle-converge"

#: ``requested_by`` of a restore of a lane with no running runtime (OMN-20864).
IDLE_RESTORE_REQUESTER: Final = "agent/idle-restore"

#: ``requested_by`` of a stability-lane rebuild that follows dev's newest
#: omnimarket tag (OMN-19998), so its job record and terminal event are that
#: lane's own receipt, never mistaken for a merge's or a dev converge's.
STABILITY_FOLLOW_REQUESTER: Final = "agent/stability-follow-tag"

#: OMN-20864: how long a lane with no running runtime must have run no job
#: before a restore starts. Long enough that a deploy's own recovery, or an
#: operator's hand refresh that just ended, is not raced; short next to the
#: 5h50m the 2026-10-11 outage cost.
RESTORE_AFTER: Final = timedelta(minutes=5)

#: OMN-20864: at most one restore per this long, so a lane that will not come up
#: costs one rebuild an hour instead of one every idle interval.
RESTORE_RETRY_AFTER: Final = timedelta(minutes=60)

#: ``docker inspect`` states of the main runtime that mean the lane is down.
DOWN_STATES: Final = frozenset({"created", "exited", "dead"})

#: OMN-19509 AC1: no job for this long.
IDLE_AFTER: Final = timedelta(minutes=16)

#: OMN-19509 AC2: the margin before a scheduled C15 or C16 run.
PROBE_MARGIN: Final = timedelta(minutes=10)

#: How long a converge may run: the plan's whole-job ceiling (task A2). A
#: converge must not START within this plus the margin before a probe, or it
#: is still running when the probe starts.
CONVERGE_CEILING: Final = timedelta(minutes=30)

#: The probes that grade .201 and bind staging delivery.
GUARDED_PROBES: Final = frozenset({"C15", "C16"})

#: The repositories whose dev heads an idle instance follows, in the order the
#: idle branch evaluates them (OMN-20867). Each is followed only on an instance
#: the routing table does not send its rebuilds to.
FOLLOWED_REPOSITORIES: Final = ("omnimarket", "omnibase_infra")

#: How often the idle branch re-evaluates. Each check reads a container file
#: and asks git for one ref.
CHECK_INTERVAL_SECONDS: Final = 60

PROBE_WINDOWS_RELPATH: Final = "config/lab_probe_windows.yaml"
PROBE_WINDOWS_PARSER_RELPATH: Final = "scripts/ci/check_lab_probe_windows.py"
BUILD_PROVENANCE_PATH: Final = "/app/build-provenance.json"
#: OMN-19998: how long a lab lane may run behind a tag-followed repository's
#: newest dev tag before the heartbeat raises its alarm.
DEV_TAG_ALARM_AFTER: Final = timedelta(hours=2)
#: OMN-19998: how often the heartbeat is published while its alarm holds still.
DEV_TAG_HEARTBEAT_INTERVAL: Final = timedelta(minutes=15)
#: The one parser of ``dev-*`` tag names, shared with the build's checkout.
DEPLOY_SOURCE_REF_RELPATH: Final = "scripts/runtime_build/deploy_source_ref.py"
READ_TIMEOUT_SECONDS: Final = 30

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class EnumIdleConvergeVerdict(StrEnum):
    CONVERGE = "converge"
    FROZEN = "frozen"
    NOT_ROUTED_ELSEWHERE = "not_routed_elsewhere"
    JOB_ACTIVE = "job_active"
    RECENTLY_ACTIVE = "recently_active"
    WINDOWS_UNREADABLE = "windows_unreadable"
    PROBE_WINDOW = "probe_window"
    HEAD_UNREADABLE = "head_unreadable"
    RUNNING_UNREADABLE = "running_unreadable"
    UP_TO_DATE = "up_to_date"
    ALREADY_ATTEMPTED = "already_attempted"


class ModelIdleConvergeInputs(BaseModel):
    """Everything one decision reads, gathered before it is made."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    now: datetime
    #: The followed repository this decision is about (OMN-20867).
    repository: str
    routed_elsewhere: bool
    job_active: bool
    #: The end of the last job, or the agent's start when it has run none.
    last_activity: datetime
    running_ref: str | None
    head_ref: str | None
    #: The probe run that excludes ``now``, or ``None`` when none does.
    probe_blocker: str | None
    #: Why the probe schedule could not be read, or ``None``.
    windows_error: str | None
    attempted_heads: frozenset[str]
    #: The instance's FROZEN reason in force (``deploy_agent.routing`` lane flags), or ``None``.
    frozen_reason: str | None = None
    #: OMN-21146: when the agent first saw the lane behind this repository's
    #: dev head (:func:`next_behind_since`), or ``None``.
    behind_since: datetime | None = None
    #: OMN-21146: the instance's declared lag bound, or ``None`` for none.
    max_lag: timedelta | None = None

    def overdue(self) -> bool:
        """Whether the lane has run behind the head for the declared bound."""
        return (
            self.max_lag is not None
            and self.behind_since is not None
            and self.now - self.behind_since >= self.max_lag
        )


class ModelIdleConvergeDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumIdleConvergeVerdict
    detail: str
    head_ref: str | None = None


def decide(inputs: ModelIdleConvergeInputs) -> ModelIdleConvergeDecision:
    """Converge, or the first reason not to. Pure: every read happened already."""

    def refuse(
        verdict: EnumIdleConvergeVerdict, detail: str
    ) -> ModelIdleConvergeDecision:
        return ModelIdleConvergeDecision(verdict=verdict, detail=detail)

    if inputs.frozen_reason is not None:
        return refuse(EnumIdleConvergeVerdict.FROZEN, inputs.frozen_reason)
    repo = inputs.repository
    if not inputs.routed_elsewhere:
        return refuse(
            EnumIdleConvergeVerdict.NOT_ROUTED_ELSEWHERE,
            f"the routing table sends {repo} rebuilds to this instance",
        )
    if inputs.job_active:
        return refuse(EnumIdleConvergeVerdict.JOB_ACTIVE, "a job is in flight")
    # OMN-21146: past the declared bound the idle wait and the probe windows
    # no longer hold the lane back; each overridden refusal is named.
    overdue = inputs.overdue()
    overridden: list[str] = []
    idle = inputs.now - inputs.last_activity
    if idle < IDLE_AFTER:
        detail = (
            f"idle {int(idle.total_seconds())}s of {int(IDLE_AFTER.total_seconds())}s"
        )
        if not overdue:
            return refuse(EnumIdleConvergeVerdict.RECENTLY_ACTIVE, detail)
        overridden.append(detail)
    if inputs.windows_error is not None:
        if not overdue:
            return refuse(
                EnumIdleConvergeVerdict.WINDOWS_UNREADABLE, inputs.windows_error
            )
        overridden.append(inputs.windows_error)
    if inputs.probe_blocker is not None:
        if not overdue:
            return refuse(EnumIdleConvergeVerdict.PROBE_WINDOW, inputs.probe_blocker)
        overridden.append(inputs.probe_blocker)
    if inputs.head_ref is None:
        return refuse(
            EnumIdleConvergeVerdict.HEAD_UNREADABLE,
            f"the {repo} dev head could not be read",
        )
    if inputs.running_ref is None:
        return refuse(
            EnumIdleConvergeVerdict.RUNNING_UNREADABLE,
            f"the lane's running {repo} revision could not be read",
        )
    if inputs.running_ref == inputs.head_ref:
        return refuse(
            EnumIdleConvergeVerdict.UP_TO_DATE,
            f"the lane runs {repo} {inputs.head_ref[:12]}, the dev head",
        )
    if inputs.head_ref in inputs.attempted_heads:
        return refuse(
            EnumIdleConvergeVerdict.ALREADY_ATTEMPTED,
            f"{repo} {inputs.head_ref[:12]} was already converged on once",
        )
    detail = (
        f"idle {int(idle.total_seconds())}s, {repo} "
        f"{inputs.running_ref[:12]} -> dev head {inputs.head_ref[:12]}"
    )
    if overdue:
        assert inputs.behind_since is not None and inputs.max_lag is not None
        lag = inputs.now - inputs.behind_since
        detail += (
            f"; overdue: behind {int(lag.total_seconds())}s of a "
            f"{int(inputs.max_lag.total_seconds())}s bound"
        )
        if overridden:
            detail += f", overriding {'; '.join(overridden)}"
    return ModelIdleConvergeDecision(
        verdict=EnumIdleConvergeVerdict.CONVERGE,
        detail=detail,
        head_ref=inputs.head_ref,
    )


def next_behind_since(
    previous: datetime | None,
    running_ref: str | None,
    head_ref: str | None,
    now: datetime,
) -> datetime | None:
    """When the lane started running behind the head, after one more read.

    Starts at the first read that sees the lane behind, keeps the earliest while
    the head moves on, and clears on a read that sees the lane on the head. A
    read that left either ref unknown proves nothing, so it changes nothing.
    """
    if running_ref is None or head_ref is None:
        return previous
    if running_ref == head_ref:
        return None
    return previous if previous is not None else now


def dev_lane_drift(
    *,
    instance: str,
    repository: str,
    runtime_container: str,
    running_image: str,
    running_ref: str | None,
    head_ref: str | None,
    behind_since: datetime | None,
    max_lag: timedelta | None,
    now: datetime,
) -> ModelDevLaneDrift | None:
    """The lane's drift behind ``repository``'s dev head, or ``None``.

    ``None`` when the instance declares no bound, the lane is not behind, or a
    ref is unread: there is nothing to say that the converge verdict does not
    already log.
    """
    if (
        max_lag is None
        or behind_since is None
        or running_ref is None
        or head_ref is None
    ):
        return None
    lag = now - behind_since
    return ModelDevLaneDrift(
        instance=instance,
        repository=repository,
        runtime_container=runtime_container,
        running_image=running_image,
        running_ref=running_ref,
        head_ref=head_ref,
        behind_since=behind_since,
        observed_at=now,
        lag_seconds=max(0, int(lag.total_seconds())),
        max_lag_seconds=int(max_lag.total_seconds()),
        overdue=lag >= max_lag,
    )


def converge_command() -> ModelRebuildRequested:
    """The one rebuild a converge runs: the dev lane, at the tracking ref.

    Workspace-sourced, like every CI-published dev command, so the build stages
    omnimarket from origin/dev (stage_workspace.sh), which is the revision the
    decision compared against; and it builds omnibase_infra at the tracking
    ref, the dev head an omnibase_infra decision compared against (OMN-20867).
    """
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by=IDLE_CONVERGE_REQUESTER,
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
        build_source=BuildSource.WORKSPACE,
    )


def _probe_windows_module(clone_root: Path) -> ModuleType:
    path = clone_root / PROBE_WINDOWS_PARSER_RELPATH
    spec = importlib.util.spec_from_file_location("_lab_probe_windows", path)
    if spec is None or spec.loader is None:
        raise OSError(f"cannot load the probe-window parser at {path}")
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its own module through sys.modules while the class
    # body executes, so the module is registered before it runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def load_probe_windows(clone_root: Path) -> list[Any]:
    """The probe schedule, through the probe-window gate's own parser.

    Raises when the file or the parser cannot be read; the caller turns that
    into a refusal.
    """
    module = _probe_windows_module(clone_root)
    windows: list[Any] = module.load_windows(clone_root / PROBE_WINDOWS_RELPATH)
    return windows


def probe_blocking(
    now: datetime,
    windows: Iterable[Any],
    *,
    guarded: frozenset[str] = GUARDED_PROBES,
    lead: timedelta = CONVERGE_CEILING + PROBE_MARGIN,
) -> str | None:
    """The guarded probe run whose exclusion contains ``now``, or ``None``.

    A run starting at ``s`` and declared to take ``d`` excludes
    ``[s - lead, s + d]``: nothing may start that would still be running at
    ``s``, and nothing may start under a run in progress.
    """
    runs: list[tuple[datetime, str, timedelta]] = []
    for window in windows:
        if window.id not in guarded:
            continue
        duration = timedelta(minutes=int(window.max_duration_minutes))
        for start in window.occurrences(
            now - duration, now + lead + timedelta(minutes=1)
        ):
            runs.append((start, str(window.id), duration))
    for start, probe_id, duration in sorted(runs):
        if start - lead <= now <= start + duration:
            return f"{probe_id} run at {start:%Y-%m-%dT%H:%MZ}"
    return None


def _default_run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, capture_output=True, text=True, check=False, timeout=timeout
    )


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(c in "0123456789abcdef" for c in value)
    )


def read_running_ref(
    container: str, repository: str, *, run: Runner = _default_run
) -> str | None:
    """The ``repository`` commit the lane's runtime image carries, or ``None``.

    Read from the image's own build-provenance manifest, the same file the
    lineage fence and the sibling-revision guard read: ``infra_vcs_ref`` for
    omnibase_infra, which the image is built from, and the vendored sibling's
    ``vcs_ref`` for any other repository.
    """
    try:
        result = run(
            ["docker", "exec", container, "cat", BUILD_PROVENANCE_PATH],
            timeout=READ_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        manifest = json.loads(result.stdout)
        ref = (
            manifest["infra_vcs_ref"]
            if repository == "omnibase_infra"
            else manifest["per_repo_vcs_provenance"]["siblings"][repository]["vcs_ref"]
        )
    except (ValueError, KeyError, TypeError):
        return None
    return ref if _is_sha(ref) else None


class EnumRuntimePresence(StrEnum):
    """Whether the lane's main runtime container is running (OMN-20864)."""

    RUNNING = "running"
    DOWN = "down"
    UNKNOWN = "unknown"


class EnumIdleRestoreVerdict(StrEnum):
    RESTORE = "restore"
    NOT_DOWN = "not_down"
    PRESENCE_UNREADABLE = "presence_unreadable"
    FROZEN = "frozen"
    JOB_ACTIVE = "job_active"
    RECENTLY_ACTIVE = "recently_active"
    RECENTLY_ATTEMPTED = "recently_attempted"


class ModelIdleRestoreInputs(BaseModel):
    """Everything one restore decision reads, gathered before it is made."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    now: datetime
    presence: EnumRuntimePresence
    #: What the presence read saw, for the log line.
    presence_detail: str
    job_active: bool
    #: The end of the last job, or the agent's start when it has run none.
    last_activity: datetime
    #: When the last restore started, or ``None`` when none has.
    last_attempt: datetime | None
    frozen_reason: str | None = None


class ModelIdleRestoreDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumIdleRestoreVerdict
    detail: str


def decide_restore(inputs: ModelIdleRestoreInputs) -> ModelIdleRestoreDecision:
    """Restore a lane whose main runtime is not running, or the first reason not to.

    Pure: every read happened already. See the module's OMN-20864 section.
    """

    def refuse(
        verdict: EnumIdleRestoreVerdict, detail: str
    ) -> ModelIdleRestoreDecision:
        return ModelIdleRestoreDecision(verdict=verdict, detail=detail)

    if inputs.presence is EnumRuntimePresence.RUNNING:
        return refuse(EnumIdleRestoreVerdict.NOT_DOWN, inputs.presence_detail)
    if inputs.presence is EnumRuntimePresence.UNKNOWN:
        return refuse(
            EnumIdleRestoreVerdict.PRESENCE_UNREADABLE, inputs.presence_detail
        )
    if inputs.frozen_reason is not None:
        return refuse(EnumIdleRestoreVerdict.FROZEN, inputs.frozen_reason)
    if inputs.job_active:
        return refuse(EnumIdleRestoreVerdict.JOB_ACTIVE, "a job is in flight")
    idle = inputs.now - inputs.last_activity
    if idle < RESTORE_AFTER:
        return refuse(
            EnumIdleRestoreVerdict.RECENTLY_ACTIVE,
            f"lane down ({inputs.presence_detail}); idle "
            f"{int(idle.total_seconds())}s of {int(RESTORE_AFTER.total_seconds())}s",
        )
    if (
        inputs.last_attempt is not None
        and inputs.now - inputs.last_attempt < RESTORE_RETRY_AFTER
    ):
        return refuse(
            EnumIdleRestoreVerdict.RECENTLY_ATTEMPTED,
            f"lane down ({inputs.presence_detail}); last restore started "
            f"{inputs.last_attempt:%Y-%m-%dT%H:%M:%SZ}, next allowed after "
            f"{int(RESTORE_RETRY_AFTER.total_seconds())}s",
        )
    return ModelIdleRestoreDecision(
        verdict=EnumIdleRestoreVerdict.RESTORE,
        detail=(
            f"lane down ({inputs.presence_detail}), idle {int(idle.total_seconds())}s"
        ),
    )


def restore_command() -> ModelRebuildRequested:
    """The one rebuild a restore runs: the converge's command, named apart.

    A runtime-scope rebuild at the tracking ref brings every runtime service
    up through the agent's ordinary path (compose up, per-container recovery,
    verification), so a restore is a deploy with a job record and a terminal
    event like any other.
    """
    return converge_command().model_copy(
        update={"correlation_id": uuid4(), "requested_by": IDLE_RESTORE_REQUESTER}
    )


def read_runtime_presence(
    container: str, *, run: Runner = _default_run
) -> tuple[EnumRuntimePresence, str]:
    """Whether ``container`` is running, down, or could not be read.

    DOWN covers both shapes of 2026-10-11: no container under the name at all
    (docker answers "No such object"), and one there but created, exited or
    dead. Any other failure -- docker not answering, a timeout -- is UNKNOWN,
    which never restores.
    """
    try:
        result = run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container],
            timeout=READ_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return EnumRuntimePresence.UNKNOWN, f"docker inspect {container}: {exc}"
    if result.returncode != 0:
        said = (result.stderr or result.stdout).strip()
        if "no such object" in said.lower() or "no such container" in said.lower():
            return EnumRuntimePresence.DOWN, f"no container named {container}"
        return (
            EnumRuntimePresence.UNKNOWN,
            f"docker inspect {container} exited {result.returncode}: {said[:200]}",
        )
    state = result.stdout.strip().lower()
    if state in DOWN_STATES:
        return EnumRuntimePresence.DOWN, f"{container} is {state}"
    return EnumRuntimePresence.RUNNING, f"{container} is {state}"


def read_running_image(container: str, *, run: Runner = _default_run) -> str:
    """``<image name>@<image id>`` of ``container``, or ``""`` when unread.

    OMN-21146: the drift event names the image the lane runs, so a reader can
    tell an old build from a build of the wrong ref.
    """
    try:
        result = run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Config.Image}}@{{.Image}}",
                container,
            ],
            timeout=READ_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def read_dev_head(clone: Path, *, run: Runner = _default_run) -> str | None:
    """The ``dev`` head on ``clone``'s remote, read from the local clone."""
    try:
        result = run(
            ["git", "-C", str(clone), "ls-remote", "origin", "refs/heads/dev"],
            timeout=READ_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first = (result.stdout.split() or [""])[0]
    return first if _is_sha(first) else None


class ModelDevTag(BaseModel):
    """One ``dev-<UTC YYYYMMDDHHMMSS>-<sha12>`` tag on a repository's origin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    sha: str
    cut_at: datetime


def _deploy_source_ref_module(clone_root: Path) -> ModuleType:
    path = clone_root / DEPLOY_SOURCE_REF_RELPATH
    spec = importlib.util.spec_from_file_location("_deploy_source_ref", path)
    if spec is None or spec.loader is None:
        raise OSError(f"cannot load the dev-tag parser at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def parse_dev_tags(ls_remote_output: str, clone_root: Path) -> list[ModelDevTag]:
    """The well-formed dev tags in ``git ls-remote --tags`` output, oldest first.

    Parsed by ``deploy_source_ref.parse_dev_tags`` from ``clone_root``, the
    same function the build's checkout uses, so what the agent follows and what
    a build stages can never read the tags differently.
    """
    module = _deploy_source_ref_module(clone_root)
    return [
        ModelDevTag(name=tag.name, sha=tag.sha, cut_at=tag.cut_at)
        for tag in module.parse_dev_tags(ls_remote_output)
    ]


def read_dev_tags(
    clone: Path, clone_root: Path, *, run: Runner = _default_run
) -> list[ModelDevTag] | None:
    """Every dev tag on ``clone``'s origin, oldest first, or ``None`` unread."""
    try:
        result = run(
            [
                "git",
                "-C",
                str(clone),
                "ls-remote",
                "--tags",
                "origin",
                "refs/tags/dev-*",
            ],
            timeout=READ_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return parse_dev_tags(result.stdout, clone_root)
    except (OSError, ImportError, AttributeError):
        return None


def dev_tag_behind_since(
    tags: list[ModelDevTag],
    running_ref: str | None,
    previous: datetime | None,
    now: datetime,
    *,
    running_contains_newest: bool = False,
) -> datetime | None:
    """When the lane started running behind the newest dev tag, or ``None``.

    ``None`` when it runs the newest tag, or a commit that contains it
    (``running_contains_newest``: a build of the branch ahead of its newest
    tag is not behind). When it runs an older tag, the cut time of the next
    tag: the first one it missed. When its commit is no tag (a build from
    before the tags, or a hand build), the first read that saw it off the
    newest tag, kept across reads. An unread running ref, or no tag at all,
    changes nothing.
    """
    if running_ref is None or not tags:
        return previous
    if running_ref == tags[-1].sha or running_contains_newest:
        return None
    for index, tag in enumerate(tags[:-1]):
        if tag.sha == running_ref:
            return tags[index + 1].cut_at
    return previous if previous is not None else now


def contains_commit(
    clone: Path, ancestor: str, descendant: str, *, run: Runner = _default_run
) -> bool:
    """Whether ``descendant`` contains ``ancestor`` in ``clone``; False unread."""
    try:
        result = run(
            [
                "git",
                "-C",
                str(clone),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            timeout=READ_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def dev_tag_heartbeat(
    *,
    instance: str,
    repository: str,
    runtime_container: str,
    running_ref: str | None,
    tags: list[ModelDevTag] | None,
    behind_since: datetime | None,
    now: datetime,
    alarm_after: timedelta = DEV_TAG_ALARM_AFTER,
) -> ModelDevTagHeartbeat:
    """The lane's reading against ``repository``'s newest dev tag."""
    newest = tags[-1] if tags else None
    lag = (
        max(timedelta(0), now - behind_since)
        if behind_since is not None
        else timedelta(0)
    )
    return ModelDevTagHeartbeat(
        instance=instance,
        repository=repository,
        runtime_container=runtime_container,
        running_ref=running_ref,
        newest_tag=newest.name if newest else None,
        newest_tag_sha=newest.sha if newest else None,
        newest_tag_cut_at=newest.cut_at if newest else None,
        behind_since=behind_since,
        observed_at=now,
        lag_seconds=int(lag.total_seconds()),
        alarm_after_seconds=int(alarm_after.total_seconds()),
        alarm=(
            running_ref is not None
            and newest is not None
            and running_ref != newest.sha
            and lag > alarm_after
        ),
    )


class EnumStabilityFollowVerdict(StrEnum):
    FOLLOW = "follow"
    FROZEN = "frozen"
    JOB_ACTIVE = "job_active"
    NO_TAG = "no_tag"
    DEV_BEHIND = "dev_behind"
    RUNNING_UNREADABLE = "running_unreadable"
    UP_TO_DATE = "up_to_date"
    ALREADY_ATTEMPTED = "already_attempted"


class ModelStabilityFollowInputs(BaseModel):
    """Every read the stability follow rule takes, taken before it decides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tags: tuple[ModelDevTag, ...] | None
    dev_running_ref: str | None
    dev_contains_newest: bool = False
    stability_running_ref: str | None
    stability_contains_newest: bool = False
    job_active: bool
    frozen_reason: str | None = None
    attempted: frozenset[str] = frozenset()


class ModelStabilityFollowDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumStabilityFollowVerdict
    newest_tag: str | None = None
    newest_tag_sha: str | None = None
    detail: str


def decide_stability_follow(
    inputs: ModelStabilityFollowInputs,
) -> ModelStabilityFollowDecision:
    """Whether the stability lane rebuilds now onto the tag dev runs (OMN-19998).

    FOLLOW only when the newest dev tag is read, the dev lane runs it (or a
    commit containing it), the stability lane reads as running something else, no
    job is in flight, the instance is not frozen, and this tag was not already
    attempted. Every unread input refuses: an unread lane is never rebuilt on
    a guess.
    """
    if inputs.frozen_reason:
        return ModelStabilityFollowDecision(
            verdict=EnumStabilityFollowVerdict.FROZEN, detail=inputs.frozen_reason
        )
    if inputs.job_active:
        return ModelStabilityFollowDecision(
            verdict=EnumStabilityFollowVerdict.JOB_ACTIVE, detail="a job is in flight"
        )
    if not inputs.tags:
        return ModelStabilityFollowDecision(
            verdict=EnumStabilityFollowVerdict.NO_TAG,
            detail="no dev tag read on omnimarket's origin",
        )
    newest = inputs.tags[-1]
    named = {"newest_tag": newest.name, "newest_tag_sha": newest.sha}
    if not (inputs.dev_running_ref == newest.sha or inputs.dev_contains_newest):
        return ModelStabilityFollowDecision(
            verdict=EnumStabilityFollowVerdict.DEV_BEHIND,
            detail=(
                f"dev runs {(inputs.dev_running_ref or 'unread')[:12]}, not "
                f"{newest.name}; stability takes a tag only once dev runs it"
            ),
            **named,
        )
    if inputs.stability_running_ref is None:
        return ModelStabilityFollowDecision(
            verdict=EnumStabilityFollowVerdict.RUNNING_UNREADABLE,
            detail="the stability lane's running omnimarket revision is unread",
            **named,
        )
    if inputs.stability_running_ref == newest.sha or inputs.stability_contains_newest:
        return ModelStabilityFollowDecision(
            verdict=EnumStabilityFollowVerdict.UP_TO_DATE,
            detail=f"the stability lane runs {newest.name}",
            **named,
        )
    if newest.sha in inputs.attempted:
        return ModelStabilityFollowDecision(
            verdict=EnumStabilityFollowVerdict.ALREADY_ATTEMPTED,
            detail=f"{newest.name} was already attempted on the stability lane",
            **named,
        )
    return ModelStabilityFollowDecision(
        verdict=EnumStabilityFollowVerdict.FOLLOW,
        detail=(
            f"the stability lane runs {inputs.stability_running_ref[:12]}; dev runs "
            f"{newest.name}"
        ),
        **named,
    )


def stability_follow_command() -> ModelRebuildRequested:
    """The one rebuild a stability follow runs: the stability lane, workspace-built.

    Workspace-sourced at the tracking ref, like a dev converge, so the build
    stages omnimarket at its newest dev tag (``DEPLOY_FOLLOW_DEV_TAG_REPOS``,
    set by the executor for every lab build).
    """
    return ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by=STABILITY_FOLLOW_REQUESTER,
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.STABILITY_TEST,
        build_source=BuildSource.WORKSPACE,
    )
