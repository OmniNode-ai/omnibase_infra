#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Assert the .201 compose dev lane carries a specific SIBLING revision (OMN-18268).

WHY THIS EXISTS
---------------

``check_dev_lane_staleness.py`` proves the lane runs a given **omnibase_infra**
commit, read from ``org.opencontainers.image.revision`` on the running container.
That label answers the question the omnibase_infra trigger asks and nothing else.

The runtime image VENDORS its siblings: ``stage_workspace.sh`` stages
``omnibase_core``, ``omnibase_compat`` and ``omnimarket`` from the host clones
into the build context, resolving each one's ref at staging time
(``DEPLOY_SIBLING_FALLBACK_REF``, default ``origin/dev`` -- OMN-17135). So an
omnimarket merge changes what the lane RUNS while changing nothing the infra
revision label can show. Point the existing guard at a sibling-triggered rebuild
and it reports ``converged`` on its first poll, having proven nothing: a false
green by construction, the same shape OMN-18200 removed from the compose-dev
receipt.

WHAT IT READS
-------------

``/app/build-provenance.json`` inside the lane's container, written at build time
by ``scripts/runtime_build/compute_workspace_provenance.py``. Its
``per_repo_vcs_provenance.siblings.<repo>.vcs_ref`` is the commit RT-1 actually
resolved for that sibling -- the build's own record of what it staged, not a
label anybody could restamp. Measured live 2026-09-14 on
``omninode-runtime-effects``: ``omnimarket -> 65c237cd`` (omnimarket#2538) while
the infra revision label read ``732fd291``.

CONTAINMENT, NOT EQUALITY
-------------------------

The sibling ref comes from ``origin/dev`` at staging time, so a build that starts
after a later merge legitimately carries a DESCENDANT of the merge that fired the
trigger. Requiring equality would red on a lane that is MORE current than asked
and would flap whenever two merges land inside one rebuild window. The question
is containment: does the lane's revision have this merge in its history?
``GET /repos/{owner}/{repo}/compare/{expected}...{lane}`` answers it --
``identical`` and ``ahead`` mean yes, ``behind`` and ``diverged`` mean no, and
any other value is refused rather than guessed.

WHY THERE IS NO FALSE-GREEN RISK ON THE FIRST POLL
--------------------------------------------------

Unlike the infra revision, the expected SHA here is a commit that did not exist
before this merge. No image built before the merge can contain it, so a first-poll
pass is a real pass: some concurrent rebuild already carried the change. That is
the honest answer to "does the lane carry this merge", which is the only question
this guard asks.

THE BUDGET STARTS AT THE AGENT'S ACCEPTANCE (OMN-18685, porting OMN-18573)
--------------------------------------------------------------------------

``--wait-timeout`` is the budget the LANE gets, and it is measured from the
moment the deploy agent ACCEPTS the rebuild command -- read from the agent's own
``/job/<correlation_id>`` record -- not from the moment this guard starts.
``check_dev_lane_staleness.py`` made this move first (OMN-18573) and the sibling
path did not receive it, so this file kept a bare wall clock opened at
``main()`` while the twin it was written to match had stopped using one.

Measured on omnimarket#2641 (2026-09-18, receipt artifact ``10546422861``,
``lab-pass-receipt-compose-dev-8d52a7ccf563...``): the trigger published at
11:57:25Z, this guard opened a 25-minute clock at 11:58:01Z, and the agent did
not accept the command until 12:08:41Z -- it was queued behind a job that ran
42m07s. Accept-to-recreate on that lane is about 22m15s, so the lane needed
roughly 32m40s from the moment the guard started and had 25m00s. The guard wrote
``[SIBLING_NOT_CONVERGED]`` at 12:23:16Z while the agent's own build argv read
``OMNIMARKET_REF=8d52a7ccf563...`` verbatim. ``sibling_revision`` was the only
failing check of eight, and rule 24(b) then refused a good sha for delivery.

``--wall-clock-seconds`` bounds how long THIS run may watch, because the job
still has to probe the lane and write the receipt. It is a BOUND, never the
budget, and it is never widened to absorb a queue.

That gives this guard a THIRD verdict, and the distinction is the point:

* ``ok`` -- the lane vendors a revision carrying the merge;
* ``fail`` -- the lane was granted its whole budget FROM ACCEPTANCE and does not
  carry it. A statement about the lane;
* ``indeterminate`` -- the acceptance could not be established (no correlation
  id, an unreachable agent, a command the effect never handed over), or this run
  ran out of its own clock first. A statement about the RUN. It is still not a
  pass: the exit status is non-zero and the receipt stays non-PASS, so rule
  24(b) still refuses the sha. What changes is what the refusal SAYS.

FAIL-CLOSED CASES
-----------------

An unreadable manifest, a sibling absent from it, a ``vcs_dirty`` tree (the SHA
does not describe what was built), a non-SHA value, a compose project that is not
the dev lane, a compare the API cannot resolve, and a budget spent from
acceptance are all failures. None of them is evidence the lane is current.

An unreadable lane keeps its FAIL, deliberately and narrowly: it is unreadable
AFTER the lane was granted its whole budget, which is a fact about a lane that
should by then be answering. What moved to INDETERMINATE is only the case where
the budget was never established or never spent -- where nothing was shown about
the lane at all.

Exit codes: ``0`` the lane carries the merge, ``1`` it does not (or the read
failed closed), ``3`` convergence indeterminate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess  # fixed argv, no shell, trusted docker/gh binaries
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# OMN-18685. The acceptance primitives are IMPORTED from the guard OMN-18573
# landed them in, never re-implemented here. A second copy of the budget
# arithmetic is a second place for the anchor to drift, and the two guards must
# agree on what "the lane's budget" means or the same lane receipts differently
# depending on which repository's merge fired the rebuild.
from scripts.ci.check_dev_lane_staleness import (
    EXIT_INDETERMINATE,
    AcceptanceUnresolvedError,
    EnumConvergenceOutcome,
    ModelAcceptanceProbe,
    ModelConvergenceBudget,
    convergence_check_outcome,
    read_agent_acceptance,
)
from scripts.ci.lab_pass_receipt import read_lane_generation

#: The lane's effects container. Named AND fenced by its compose project below,
#: so this guard can never read a governed lane (prod, stability-test, judge, or
#: a collaborator lane) even if a container were renamed onto the same daemon.
DEFAULT_CONTAINER = "omninode-runtime-effects"
DEFAULT_COMPOSE_PROJECT = "omnibase-infra"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

#: Written into the image by the workspace build; the label pointing at it is
#: ``com.omninode.workspace_provenance_manifest``.
DEFAULT_PROVENANCE_PATH = "/app/build-provenance.json"

#: Matches the bound ``check_dev_lane_staleness.py`` uses for the same lane, for
#: the same reason: node_redeploy_orchestrator and node_redeploy_deploy_effect
#: each declare ``timeout_ms: 660000``, plus margin for the compose recreate.
DEFAULT_WAIT = timedelta(minutes=25)
DEFAULT_POLL_INTERVAL = timedelta(seconds=60)

#: Compare statuses that mean the lane's revision HAS the expected commit.
CARRIES_STATUSES = frozenset({"identical", "ahead"})
#: Compare statuses that mean it does not. Anything outside either set is
#: unrecognised and is refused rather than mapped onto the nearest guess.
MISSING_STATUSES = frozenset({"behind", "diverged"})

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class Finding:
    code: str
    detail: str

    def render(self) -> str:
        return f"[{self.code}] {self.detail}"


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings


def _format_age(delta: timedelta) -> str:
    total = int(max(delta.total_seconds(), 0))
    minutes, seconds = divmod(total, 60)
    return f"{minutes}m{seconds:02d}s"


def assert_lane_fence(
    compose_project: str, container: str, expected_project: str
) -> None:
    """Refuse to evaluate anything but the declared dev lane."""
    if compose_project != expected_project:
        raise ValueError(
            f"lane fence: container {container!r} reports "
            f"{COMPOSE_PROJECT_LABEL}={compose_project!r}, expected "
            f"{expected_project!r}. This guard evaluates the dev lane only and "
            "refuses to read any other compose project."
        )


def parse_sibling_revision(payload: Any, repo: str) -> str:
    """Project a build-provenance manifest onto one sibling's resolved commit.

    Split out from the docker read so the test drives it over the manifest bytes
    read from the RUNNING container, rather than a dict a test author rebuilt by
    hand -- a replay that reconstructs the shape proves the verdict logic and
    nothing about whether the guard can read what the build actually writes.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"build-provenance manifest is {type(payload).__name__}, not an object"
        )
    siblings = (payload.get("per_repo_vcs_provenance") or {}).get("siblings") or {}
    if not isinstance(siblings, dict):
        raise ValueError(
            "build-provenance manifest has no per_repo_vcs_provenance.siblings "
            "object; this image was not built in workspace mode, so which "
            "sibling revisions it carries is unknown."
        )
    entry = siblings.get(repo)
    if entry is None:
        listed = ", ".join(sorted(siblings)) or "(none)"
        raise ValueError(
            f"sibling {repo!r} is absent from the lane image's build-provenance "
            f"manifest (it lists: {listed}). Absent is UNKNOWN, never unchanged."
        )
    if not isinstance(entry, dict):
        raise ValueError(f"sibling {repo!r} entry is not an object: {entry!r}")
    if entry.get("vcs_dirty"):
        raise ValueError(
            f"sibling {repo!r} was staged from a DIRTY tree (vcs_dirty=true), so "
            f"its recorded vcs_ref {entry.get('vcs_ref')!r} does not describe "
            "what was built. Failing closed."
        )
    revision = str(entry.get("vcs_ref", "")).strip().lower()
    if not _SHA_RE.match(revision):
        raise ValueError(
            f"sibling {repo!r} vcs_ref={entry.get('vcs_ref')!r} is not a git SHA; "
            "refusing to compare an unrecognised identity."
        )
    return revision


def evaluate_sibling_convergence(
    *,
    repo: str,
    lane_revision: str,
    expected_revision: str,
    containment: str,
    waited: timedelta,
    wait_timeout: timedelta,
) -> Verdict:
    """Decide whether the lane's vendored ``repo`` carries ``expected_revision``."""
    verdict = Verdict()
    status = containment.strip().lower()
    if status in CARRIES_STATUSES:
        verdict.notes.append(
            f"lane vendors {repo} at {lane_revision[:12]}, which carries "
            f"{expected_revision[:12]} ({status}), after {_format_age(waited)}"
        )
        return verdict

    if status in MISSING_STATUSES:
        verdict.findings.append(
            Finding(
                "SIBLING_NOT_CONVERGED",
                f"after {_format_age(waited)} (bound {_format_age(wait_timeout)}) "
                f"the dev lane still vendors {repo} at {lane_revision[:12]}, "
                f"which is {status} relative to {expected_revision[:12]} -- the "
                "merge this run published a redeploy-start for. The lane does NOT "
                "carry that change. Check, in order: (1) whether a "
                "rebuild-requested exists for this correlation at all and what is "
                "queued ahead of it on onex.cmd.deploy.rebuild-requested.v1; (2) "
                "whether stage_workspace.sh resolved the sibling from a clone "
                "whose origin/dev was itself stale; (3) the runtime-effects DLQ. "
                "Do NOT widen the bound to clear this.",
            )
        )
        return verdict

    verdict.findings.append(
        Finding(
            "CONTAINMENT_UNKNOWN",
            f"the GitHub compare of {expected_revision[:12]}...{lane_revision[:12]} "
            f"in {repo} returned status {containment!r}, which this guard does not "
            "recognise. An unrecognised status is not a pass; failing closed.",
        )
    )
    return verdict


# The evidence line is written into a single-line GITHUB_OUTPUT record and then
# into a receipt check, so it carries nothing a shell would re-interpret and
# nothing that would split the record in two. Same table as
# ``check_dev_lane_staleness.py``'s, for the same reason.
_EVIDENCE_UNSAFE = str.maketrans({"\n": " ", "\r": " ", '"': "'", "`": "'", "$": "S"})


@dataclass(frozen=True)
class ModelSiblingObservation:
    """One look at the lane: what it vendors and how that relates, or why not.

    Carries EXACTLY one of a reading and an ``unreadable_reason``. Both, or
    neither, would make "the lane vendors an older sibling" and "the lane could
    not be read" the same value -- and those lead to different next actions, so
    collapsing them is the shape this port exists to refuse one layer up.
    """

    lane_revision: str
    containment: str
    unreadable_reason: str

    def __post_init__(self) -> None:
        readable = bool(self.lane_revision and self.containment)
        if readable == bool(self.unreadable_reason):
            msg = (
                "a sibling observation carries EXACTLY one of a reading "
                "(revision AND containment) and an unreadable reason"
            )
            raise ValueError(msg)

    @property
    def carries(self) -> bool:
        """Whether the lane's vendored revision HAS the expected commit."""
        return self.containment.strip().lower() in CARRIES_STATUSES

    @property
    def unrecognised(self) -> bool:
        """A compare status this guard does not map. Never guessed at."""
        if self.unreadable_reason:
            return False
        status = self.containment.strip().lower()
        return status not in CARRIES_STATUSES and status not in MISSING_STATUSES


@dataclass(frozen=True)
class ModelSiblingConvergenceResult:
    """Everything the emitting step needs, structured rather than re-derived."""

    outcome: EnumConvergenceOutcome
    reason: str
    observation: ModelSiblingObservation
    budget: ModelConvergenceBudget
    waited: timedelta
    finished_at: datetime


def run_sibling_convergence_wait(
    *,
    observe: Callable[[], ModelSiblingObservation],
    resolve_acceptance: Callable[[], ModelAcceptanceProbe],
    declared_budget: timedelta,
    wall_clock: timedelta,
    poll_interval: timedelta,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
) -> ModelSiblingConvergenceResult:
    """Wait for the lane to vendor the merge, on the LANE's clock (OMN-18685).

    The same loop ``run_convergence_wait`` runs for the infra revision, over the
    sibling's build-provenance reading instead of the container's revision
    label. It is a separate loop rather than a shared generic one because the
    two guards resolve convergence from different surfaces and the shared part
    -- the budget, its anchor and its three verdicts -- IS shared, by import.

    Three exits, which are the three verdicts:

    * the lane carries the merge -- ``OK``;
    * acceptance is known and its budget is spent -- ``FAIL``, about the lane;
    * this job's own wall clock expired first, or acceptance was never
      established -- ``INDETERMINATE``, about the run.

    Every moving part is injected so all three shapes are testable without a
    lane, an agent, or a wall clock that really passes.
    """
    started = clock()
    wall_deadline = started + wall_clock
    probe = resolve_acceptance()
    budget = ModelConvergenceBudget(
        declared_seconds=int(declared_budget.total_seconds()),
        wall_clock_seconds=int(wall_clock.total_seconds()),
        acceptance=probe.acceptance,
        unresolved_reason=probe.reason,
    )
    observation = observe()

    while not observation.carries:
        now = clock()
        if budget.established and budget.exhausted(now):
            break
        if now >= wall_deadline:
            break
        remaining = (wall_deadline - now).total_seconds()
        if budget.established:
            deadline = budget.deadline()
            assert deadline is not None
            remaining = min(remaining, (deadline - now).total_seconds())
        if remaining <= 0:
            break
        sleep(min(poll_interval.total_seconds(), remaining))
        if not budget.established:
            # Keep asking. The command is usually accepted DURING this wait,
            # and the moment it is, the lane's budget starts from the agent's
            # own timestamp rather than from when CI noticed.
            probe = resolve_acceptance()
            budget = ModelConvergenceBudget(
                declared_seconds=budget.declared_seconds,
                wall_clock_seconds=budget.wall_clock_seconds,
                acceptance=probe.acceptance,
                unresolved_reason=probe.reason,
            )
        observation = observe()

    now = clock()
    waited = now - started

    def _result(
        outcome: EnumConvergenceOutcome, reason: str
    ) -> ModelSiblingConvergenceResult:
        return ModelSiblingConvergenceResult(
            outcome=outcome,
            reason=reason,
            observation=observation,
            budget=budget,
            waited=waited,
            finished_at=now,
        )

    if observation.carries:
        return _result(EnumConvergenceOutcome.OK, "")
    # Ordered deliberately: an unestablished budget is checked BEFORE anything
    # that would describe the lane, because without a budget nothing about the
    # lane has been shown -- not even by an unreadable read.
    if not budget.established:
        return _result(EnumConvergenceOutcome.INDETERMINATE, budget.unresolved_reason)
    if observation.unrecognised:
        # An unrecognised compare status is a refusal, not a wait: polling
        # cannot turn it into one of the four this guard maps.
        return _result(EnumConvergenceOutcome.FAIL, "")
    if budget.exhausted(now):
        return _result(EnumConvergenceOutcome.FAIL, "")
    return _result(EnumConvergenceOutcome.INDETERMINATE, budget.shortfall_reason(now))


def sibling_convergence_evidence(
    *,
    repo: str,
    expected_revision: str,
    result: ModelSiblingConvergenceResult,
) -> str:
    """Render the one line the receipt's ``sibling_revision`` check carries.

    The string this replaces was assembled in the WORKFLOW from the step's
    exit status alone -- ``ok`` or ``fail``, with a static sentence naming the
    flags -- so a FAIL receipt said which command ran and nothing about what
    was read. A reader could not tell a lane vendoring an older sibling from a
    lane whose budget had never started, which is exactly the confusion
    OMN-18685 was filed on.

    On every verdict the line names both revisions. When acceptance is known it
    also carries the agent's timestamp and the elapsed time since it -- on a
    PASS too, because a passing receipt that does not say where the lane's
    clock started cannot be used to check that it started in the right place.
    """
    expected = expected_revision[:12]
    observed = result.observation.lane_revision[:12]
    waited_text = _format_age(result.waited)
    phrase = result.budget.acceptance_phrase(result.finished_at)
    acceptance_clause = f"; {phrase}" if phrase else ""

    if result.outcome is EnumConvergenceOutcome.INDETERMINATE:
        seen = (
            f"lane vendors {repo} at {observed} ({result.observation.containment})"
            if not result.observation.unreadable_reason
            else f"the lane's {repo} revision could not be read "
            f"({result.observation.unreadable_reason})"
        )
        line = (
            f"INDETERMINATE: {seen}; whether it carries merge sha {expected} was "
            f"not established after {waited_text}{acceptance_clause}. Reason: "
            f"{result.reason}. This asserts nothing about the lane; the receipt "
            "is still non-PASS."
        )
        return " ".join(line.translate(_EVIDENCE_UNSAFE).split())

    if result.observation.unreadable_reason:
        line = (
            f"the lane's {repo} revision could not be read after {waited_text}, "
            f"so whether it carries merge sha {expected} is unproven and this "
            f"fails closed: {result.observation.unreadable_reason}"
        )
    elif result.outcome is EnumConvergenceOutcome.OK:
        line = (
            f"lane vendors {repo} at {observed}, which carries merge sha "
            f"{expected} ({result.observation.containment}), converged after "
            f"{waited_text} (check_lane_sibling_revision.py --repo {repo} "
            f"--expect-revision {expected_revision} against the "
            f"{DEFAULT_COMPOSE_PROJECT} compose project)"
        )
    elif result.observation.unrecognised:
        line = (
            f"lane vendors {repo} at {observed}; the GitHub compare of "
            f"{expected}...{observed} returned status "
            f"{result.observation.containment!r}, which this guard does not "
            "recognise and will not map onto the nearest guess"
        )
    else:
        line = (
            f"lane vendors {repo} at {observed}, which is "
            f"{result.observation.containment} relative to merge sha "
            f"{expected}; not converged after {waited_text}"
        )
    return " ".join((line + acceptance_clause).translate(_EVIDENCE_UNSAFE).split())


def _run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} {' '.join(argv[1:])} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def read_compose_project(container: str) -> str:
    raw = _run(
        [
            "docker",
            "inspect",
            container,
            "--format",
            f'{{{{ index .Config.Labels "{COMPOSE_PROJECT_LABEL}" }}}}',
        ]
    )
    return raw.strip()


def read_lane_sibling_revision(container: str, repo: str, provenance_path: str) -> str:
    """Read one sibling's staged commit out of the running lane image."""
    raw = _run(["docker", "exec", container, "cat", provenance_path])
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{provenance_path} in {container} is not valid JSON: {exc}"
        ) from exc
    return parse_sibling_revision(payload, repo)


def read_containment(repo_slug: str, expected: str, actual: str) -> str:
    """Ask GitHub whether ``actual`` has ``expected`` in its history.

    Uses the compare API rather than a local ``git merge-base`` for the same
    reason ``check_dev_lane_staleness.py`` does: the job has a depth-1 checkout
    of one repository, and a guard that silently answered from history it does
    not have would be the same class of false green this ticket is about.
    """
    payload = json.loads(
        _run(["gh", "api", f"repos/{repo_slug}/compare/{expected}...{actual}"])
    )
    return str(payload["status"])


def _parse_duration(value: str) -> timedelta:
    """Parse ``90s`` / ``25m`` / ``2h`` into a timedelta."""
    match = re.fullmatch(r"(\d+)([smh])", value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a duration like 90s, 25m or 2h"
        )
    amount, unit = int(match.group(1)), match.group(2)
    return {
        "s": timedelta(seconds=amount),
        "m": timedelta(minutes=amount),
        "h": timedelta(hours=amount),
    }[unit]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", required=True, help="Sibling repo name, e.g. omnimarket"
    )
    parser.add_argument(
        "--repo-slug",
        required=True,
        help="owner/name of the sibling repo for the compare API",
    )
    parser.add_argument(
        "--expect-revision",
        required=True,
        help="The sibling commit the lane must carry (the merge that fired this run)",
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--compose-project", default=DEFAULT_COMPOSE_PROJECT)
    parser.add_argument("--provenance-path", default=DEFAULT_PROVENANCE_PATH)
    parser.add_argument(
        "--wait-timeout",
        type=_parse_duration,
        default=DEFAULT_WAIT,
        help=(
            "the LANE's convergence budget, measured from the moment the deploy "
            "agent accepts the rebuild command -- never from this step's start"
        ),
    )
    parser.add_argument(
        "--poll-interval", type=_parse_duration, default=DEFAULT_POLL_INTERVAL
    )
    # OMN-18685, porting OMN-18573. The three inputs that move the convergence
    # budget off this step's own clock and onto the lane's. None of them widens
    # the budget: --wait-timeout is still the lane's grant, and
    # --wall-clock-seconds only says how long THIS job may watch before it must
    # let the receipt be written.
    parser.add_argument(
        "--agent-url",
        default="",
        help=(
            "the deploy agent's HTTP surface, whose /job/<correlation_id> "
            "endpoint serves the acceptance timestamp the convergence budget is "
            "measured from. Empty means the budget cannot be established, which "
            "is INDETERMINATE and never a lane failure."
        ),
    )
    parser.add_argument(
        "--correlation-id",
        default="",
        help=(
            "the redeploy correlation id this run published, as written by "
            "scripts/trigger_rebuild_on_merge.py. Empty is INDETERMINATE."
        ),
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=float,
        default=10.0,
        help="per-request timeout for the deploy-agent acceptance read",
    )
    parser.add_argument(
        "--wall-clock-seconds",
        type=int,
        default=int(DEFAULT_WAIT.total_seconds()),
        help=(
            "how long THIS job may watch before it must stop and let the "
            "receipt be written. A BOUND on the wait, never the lane's budget: "
            "when it is what stopped the wait, the verdict is INDETERMINATE "
            "naming both numbers, not a FAIL blaming the lane."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    expected = args.expect_revision.strip().lower()
    if not _SHA_RE.match(expected):
        print(
            f"::error::--expect-revision={args.expect_revision!r} is not a git SHA",
            file=sys.stderr,
        )
        return 1

    def _observe() -> ModelSiblingObservation:
        try:
            assert_lane_fence(
                read_compose_project(args.container),
                args.container,
                args.compose_project,
            )
            lane_revision = read_lane_sibling_revision(
                args.container, args.repo, args.provenance_path
            )
            containment = read_containment(args.repo_slug, expected, lane_revision)
        except (RuntimeError, ValueError) as exc:
            # A read failure early in the window is usually a lane mid-recreate.
            # It is carried as a REASON rather than raised, so the loop can keep
            # polling and the final verdict can say which of the three it is.
            return ModelSiblingObservation(
                lane_revision="", containment="", unreadable_reason=str(exc)
            )
        return ModelSiblingObservation(
            lane_revision=lane_revision, containment=containment, unreadable_reason=""
        )

    def _resolve_acceptance() -> ModelAcceptanceProbe:
        if not args.correlation_id.strip():
            return ModelAcceptanceProbe(
                acceptance=None,
                reason=(
                    "the publishing job recorded no correlation id for this run, "
                    "so the deploy agent's acceptance cannot be located and the "
                    "lane's convergence budget has no start"
                ),
            )
        if not args.agent_url.strip():
            return ModelAcceptanceProbe(
                acceptance=None,
                reason="no deploy-agent URL was supplied to this guard",
            )
        try:
            acceptance = read_agent_acceptance(
                agent_url=args.agent_url,
                correlation_id=args.correlation_id,
                request_timeout_seconds=args.agent_timeout_seconds,
            )
        except AcceptanceUnresolvedError as exc:
            return ModelAcceptanceProbe(acceptance=None, reason=str(exc))
        return ModelAcceptanceProbe(acceptance=acceptance, reason="")

    result = run_sibling_convergence_wait(
        observe=_observe,
        resolve_acceptance=_resolve_acceptance,
        declared_budget=args.wait_timeout,
        wall_clock=timedelta(seconds=args.wall_clock_seconds),
        poll_interval=args.poll_interval,
        clock=lambda: datetime.now(UTC),
        sleep=time.sleep,
    )

    evidence = sibling_convergence_evidence(
        repo=args.repo, expected_revision=expected, result=result
    )
    # The emit step reads these instead of re-deriving a verdict from the exit
    # status: the status has two values and this has three, and a mapping
    # written in a `run:` block is a mapping nothing tests.
    _write_output("evidence", evidence)
    _write_output("verdict", convergence_check_outcome(result.outcome).value)

    # OMN-18436: publish the identity of the container this guard read, so the
    # lab-pass probe that runs next can prove its HTTP reads came from the SAME
    # generation. The probe runs under `if: always()`, so on a failed guard the
    # lane still answers -- from the PREVIOUS generation -- and without this the
    # receipt records green reads with nothing in it naming the container that
    # produced them. Written on all three outcomes, because "which container
    # answered" has an answer whichever verdict this reached.
    _write_output_generation(args.container)

    budget = result.budget
    _summary(
        [
            "## Sibling-revision convergence (OMN-18268, OMN-18685)",
            "",
            f"- verdict: `{result.outcome.value}`",
            f"- sibling: `{args.repo}` ({args.repo_slug})",
            f"- expected revision: `{expected}`",
            f"- lane vendors: "
            f"`{result.observation.lane_revision or 'UNREADABLE'}`"
            f" (`{result.observation.containment or result.observation.unreadable_reason}`)",
            f"- agent acceptance: "
            f"`{budget.acceptance.accepted_at.isoformat() if budget.acceptance else 'UNESTABLISHED'}`",
            f"- lane budget: {budget.declared_seconds}s from acceptance; this "
            f"job could watch for {budget.wall_clock_seconds}s",
            f"- watched: {_format_age(result.waited)}",
            *([f"- reason: {result.reason}"] if result.reason else []),
            "",
            "Convergence is CONTAINMENT: a lane vendoring a DESCENDANT of the "
            "merge has exercised the change. A lane vendoring an ancestor has "
            "not, and is still a failure.",
            "",
            "The budget is measured from the moment the deploy agent ACCEPTED "
            "the rebuild command, never from this step. A verdict of "
            "`indeterminate` means the lane was never shown to have had its "
            "budget -- it is not a finding about the lane, and it is still not "
            "a pass.",
        ]
    )

    if result.outcome is EnumConvergenceOutcome.OK:
        print(f"[ok] {evidence}")
        return 0
    if result.outcome is EnumConvergenceOutcome.INDETERMINATE:
        print(f"::warning::{evidence}")
        return EXIT_INDETERMINATE

    # The operator-facing FAIL text -- the "check, in order:" triage list -- is
    # rendered by the same evaluator the OMN-18268 tests pin, so the message a
    # reader has learnt to act on is unchanged by this port.
    if result.observation.unreadable_reason:
        print(f"::error::{result.observation.unreadable_reason}", file=sys.stderr)
        return 1
    verdict = evaluate_sibling_convergence(
        repo=args.repo,
        lane_revision=result.observation.lane_revision,
        expected_revision=expected,
        containment=result.observation.containment,
        waited=result.waited,
        wait_timeout=args.wait_timeout,
    )
    for note in verdict.notes:
        print(note)
    for finding in verdict.findings:
        print(f"::error::{finding.render()}", file=sys.stderr)
    return 0 if verdict.ok else 1


def _summary(lines: list[str]) -> None:
    """Append one block to this step's job summary, when there is one."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_output(name: str, value: str) -> None:
    """Publish one single-line value to the calling step's outputs."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _write_output_generation(container: str) -> None:
    """Publish one single-line generation record to the calling step's outputs.

    An unreadable container warns and writes nothing; the probe treats an absent
    record as a binding FAILURE rather than as permission to skip the check, so
    a silent gap here cannot become a silent pass there.
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    try:
        generation = read_lane_generation(container)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"::warning::lane generation unreadable: {exc}")
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"generation={generation.to_json()}\n")


if __name__ == "__main__":
    raise SystemExit(main())
