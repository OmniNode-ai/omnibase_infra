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
    ModelRebuildRequested,
    Scope,
)

logger = logging.getLogger(__name__)

#: ``requested_by`` of every idle converge, so its job record and its terminal
#: event are never mistaken for a merge's.
IDLE_CONVERGE_REQUESTER: Final = "agent/idle-converge"

#: ``requested_by`` of a restore of a lane with no running runtime (OMN-20864).
IDLE_RESTORE_REQUESTER: Final = "agent/idle-restore"

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
    idle = inputs.now - inputs.last_activity
    if idle < IDLE_AFTER:
        return refuse(
            EnumIdleConvergeVerdict.RECENTLY_ACTIVE,
            f"idle {int(idle.total_seconds())}s of {int(IDLE_AFTER.total_seconds())}s",
        )
    if inputs.windows_error is not None:
        return refuse(EnumIdleConvergeVerdict.WINDOWS_UNREADABLE, inputs.windows_error)
    if inputs.probe_blocker is not None:
        return refuse(EnumIdleConvergeVerdict.PROBE_WINDOW, inputs.probe_blocker)
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
    return ModelIdleConvergeDecision(
        verdict=EnumIdleConvergeVerdict.CONVERGE,
        detail=(
            f"idle {int(idle.total_seconds())}s, {repo} "
            f"{inputs.running_ref[:12]} -> dev head {inputs.head_ref[:12]}"
        ),
        head_ref=inputs.head_ref,
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
