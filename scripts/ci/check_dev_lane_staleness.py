# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-lane deployed-SHA staleness guard (OMN-17888 AC4).

Why this exists
---------------
OMN-17888's finding was that ``runtime-rebuild-trigger.yml`` logged a line that
read like success and then failed to deliver, so the dev lane ran code older
than ``dev`` while the mechanism that was supposed to say so reported the
opposite. AC1-AC3 fixed the publisher. They do not make the NEXT silent break
loud, and the ticket says so in AC4:

    "A check that the dev lane's deployed SHA tracks ``dev``, so a silent
    delivery failure surfaces as lane staleness rather than only as an
    ignorable red post-merge check."

This is that check. It is deliberately independent of the delivery mechanism it
watches: it reads what is RUNNING and compares it to ``dev``. Every layer
between the two — the CI publisher, the control bus, the orchestrator, the FSM
reducer, the deploy effect, the deploy agent, the compose recreate — can fail in
its own way, and this guard reports the same fact regardless of which one did:
the lane is not running ``dev``.

That property is the point. The measured 2026-09-06 failure was NOT the one
OMN-17888 originally described. The publisher fix landed and worked — five
redeploy-start commands were delivered and consumed (topic watermark 63 -> 68,
orchestrator group LAG 0) — and the lane still did not move, because
``omninode-runtime-effects`` dead-lettered the downstream
``prod-promotion-gate-evaluated`` event with ``failure_class=no_dispatcher``.
A guard wired to "did the publish succeed" would have been green through that.
This one is red.

What it reads, and why by ``docker inspect``
--------------------------------------------
The fact is the ``org.opencontainers.image.revision`` label on the RUNNING
container, stamped from the git SHA at build time by ``docker/Dockerfile.runtime``
(see :mod:`scripts.check_runtime_image_identity`). It is the same label the prod
promotion lineage guard treats as authoritative.

There is no HTTP surface carrying it. Probed live 2026-09-06T18:4xZ against the
dev lane: ``GET :8085/health`` returns ``version`` (the *package* version,
``0.38.19``) and ``GET :8085/v1/introspection/manifest`` returns
``package_version`` per contract. Neither exposes a git revision, so an HTTP
read could only compare package versions — which do not move on most merges and
would report "fresh" through exactly the outage this guard exists for.

Lane fence
----------
The container is resolved by name AND its ``com.docker.compose.project`` label
is asserted to be the dev lane's project. A guard that could silently read
``omninode-prod-runtime`` — a container one label away on the same daemon —
would be an unreviewed read of a governed lane. Any mismatch is a hard failure,
not a fallback.

Thresholds
----------
AC4 states no bound. The defaults here are therefore a stated policy choice, not
a measurement: ``--max-commits-behind 3`` and ``--max-age 2h``. The rationale is
that the dev lane is the fully mutable test platform whose whole purpose is to
carry merged ``dev`` code; three merges or two hours of divergence is already
enough for someone to test against the wrong runtime. Both bounds are flags;
neither may be widened to turn a genuinely stale lane green (AC5: "Not satisfied
by removing or muting the failing step").

Convergence mode
----------------
``--expect-revision <sha>`` polls until the lane reports a revision that CONTAINS
that sha, and fails when it does not within ``--wait-timeout``. This is what the
post-merge job uses, so a delivered-but-not-applied redeploy surfaces on the SAME
run that published it rather than an hour later. The default 25-minute wait is
derived from checked-in contract values, not invented: ``node_redeploy_orchestrator``
and ``node_redeploy_deploy_effect`` each declare ``timeout_ms: 660000`` (11 min),
so the declared chain bound is 22 minutes, plus a 3-minute margin for the bus
hops and the container recreate.

Convergence is containment, not byte equality (OMN-18388)
---------------------------------------------------------
This mode originally demanded that the label equal the merge sha byte for byte.
Measured 2026-09-15 on run 34934096166 (merge sha ``18b539f0``): the deploy agent
had rebuilt the lane at 08:15Z from a LATER dev head, ``0d0250a6``, which CONTAINS
``18b539f0``. Byte equality reported that lane as never having applied the change,
the job failed NOT_CONVERGED at 08:40:32Z, and the compose-dev lab-pass receipt
for ``18b539f0`` was emitted FAIL on ``deployed_revision``. With dev taking several
runtime-affecting merges an hour and a rebuild costing 30 minutes to two hours
under lab load, most merge shas could not obtain a PASS receipt at all, because by
the time the lane converged it was already on a newer head — and rule 24(b) then
refuses to deliver any sha that was not the last merge before a quiet period.

The two directions are NOT symmetric and the guard must keep them apart:

* a lane on a **descendant** of the merge sha, **on the tracked branch**, contains
  the change and has exercised it — converged;
* a lane on an **ancestor** is running code older than the merge — stale, and the
  same delivered-but-not-applied finding as before (this is the OMN-18284
  "stale lane serving happily" class, which must not be widened away);
* a lane on a revision the branch does not contain — an unrelated commit or a
  branch build — is NOT converged for its own reason, whether or not it happens to
  contain the merge sha. Measured fixture: PR 3569's head ``3ad9b3af`` is seven
  commits ahead of ``18b539f0`` and dev does not contain it.

Both facts come from GitHub's compare API rather than local ``git`` because the
job's checkout is depth 1: ``compare/{expected}...{observed}`` gives the relation,
``compare/{observed}...{branch}`` gives containment. An ancestry that cannot be
resolved is its own finding and never a pass.

The receipt KEY is untouched by any of this. Rule 24(b) keys the artifact by the
exact 40-hex merge sha and ``ModelLabPassReceipt`` enforces that; the descendant
window lives in the CHECK and its evidence, which name the observed revision and
the ancestry relation.

Positive control
----------------
``--positive-control`` (with ``--deployed-revision``) inverts the exit contract:
it exits 0 only if the supplied revision is reported STALE. An empty or green
result from a verification surface is not evidence of health unless the surface
has been shown to fire, so the control is part of the guard rather than a thing
someone is supposed to remember to do by hand.

Fail-closed
-----------
Every unreadable state is a non-pass: the container missing or not running, an
empty or sentinel revision label, a revision GitHub cannot resolve, a compare
whose status is ``diverged`` or ``behind``, a docker or API error. A staleness
guard that reports fresh when it could not look manufactures the exact false
assurance the ticket is about.

The convergence budget starts at the agent's acceptance (OMN-18573)
-------------------------------------------------------------------
``--wait-timeout`` is the budget the LANE gets, and it is measured from the
moment the deploy agent ACCEPTS the rebuild command -- read from the agent's own
``/job/<correlation_id>`` record -- not from the moment this guard starts. The
redeploy-start effect is serial, so the gap between a merge and the agent taking
its command is a queue the lane is not responsible for. Measured twice on
OMN-17214 (2026-09-16, and 2026-09-17 for merge ``50c57653d3cf``, receipt
artifact 10496150688): the lane converged three minutes after a step-anchored
window closed, the receipt read FAIL on ``deployed_revision`` alone with its
other seven checks green, and rule 24(b) then refused a good sha for staging.

``--wall-clock-seconds`` bounds how long THIS run may watch, because the job
still has to settle the lane and write the receipt. It is a bound, never the
budget, and it is never widened to absorb a queue.

That gives convergence mode a THIRD verdict, and the distinction is the point:

* ``ok`` -- the lane contains the merge sha;
* ``fail`` -- the lane was granted its whole budget from acceptance and does not
  contain it. A statement about the lane;
* ``indeterminate`` -- the acceptance could not be established (no correlation
  id, an unreachable agent, a command the effect never handed over), or this run
  ran out of its own clock first. A statement about the RUN. It is still not a
  pass: the exit status is non-zero and the receipt stays non-PASS, so rule
  24(b) still refuses the sha. What changes is what the refusal says.

Exit codes: ``0`` the lane tracks ``dev`` / converged, ``1`` stale or not
converged, ``3`` convergence indeterminate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess  # fixed argv, no shell, trusted docker/gh binaries
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci.lab_pass_receipt import read_lane_generation

DEFAULT_REPO = "OmniNode-ai/omnibase_infra"
DEFAULT_BRANCH = "dev"

# The dev lane on the .201 host. Both are asserted, not assumed: the name
# resolves the container and the compose project fences the lane, so this guard
# can never read a governed lane (prod / stability-test / judge / lakshman) even
# if a container were renamed onto the same daemon.
DEV_LANE_CONTAINER = "omninode-runtime"
DEV_LANE_COMPOSE_PROJECT = "omnibase-infra"

REVISION_LABEL = "org.opencontainers.image.revision"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

# AC4 sets no bound. These are a stated policy choice; see the module docstring.
DEFAULT_MAX_COMMITS_BEHIND = 3
DEFAULT_MAX_AGE = timedelta(hours=2)

# node_redeploy_orchestrator + node_redeploy_deploy_effect each declare
# timeout_ms: 660000 in their checked-in contracts (11 min each = 22 min), plus
# a 3-minute margin for bus hops and the compose recreate.
DEFAULT_CONVERGENCE_WAIT = timedelta(minutes=25)
DEFAULT_POLL_INTERVAL = timedelta(seconds=60)

# Values docker/Dockerfile.runtime or a non-workspace build can leave in the
# label. Treated as "unknown", never as "matches".
SENTINEL_REVISIONS = frozenset({"", "unknown", "none", "null", "dev", "HEAD"})

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

#: A FULL 40-character sha. Distinct from ``_SHA_RE`` above, which admits an
#: abbreviation because a docker label may carry one. The supersession field
#: comes from the agent's own job record and names a commit a receipt is keyed
#: by, and rule 24(b) gates on the exact sha -- so an abbreviation there is a
#: field this guard does not understand, not a shorter answer.
_EXACT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# OMN-18388. What the OBSERVED lane revision is, relative to the expected merge
# sha. ``compare/{base}...{head}`` describes the HEAD relative to the BASE, so
# with base=expected and head=observed, "ahead" means the lane is ahead of the
# merge commit — it contains it.
RELATION_IDENTICAL = "identical"
RELATION_DESCENDANT = "descendant"
RELATION_ANCESTOR = "ancestor"
RELATION_UNRELATED = "unrelated"

_RELATION_BY_COMPARE_STATUS = {
    "identical": RELATION_IDENTICAL,
    "ahead": RELATION_DESCENDANT,
    "behind": RELATION_ANCESTOR,
    "diverged": RELATION_UNRELATED,
}

# Relations that mean the lane has run the merged change.
_CONTAINING_RELATIONS = frozenset({RELATION_IDENTICAL, RELATION_DESCENDANT})

# ``compare/{observed}...{branch}`` statuses that mean the BRANCH contains the
# observed revision. "behind" means the branch is behind it (a build from an
# unmerged ref) and "diverged" means another line of history; neither is on the
# branch.
_CONTAINED_COMPARE_STATUSES = frozenset({"identical", "ahead"})

# GITHUB_OUTPUT is line-oriented and the evidence is re-read as one field of a
# receipt check, so it is rendered on a single line with nothing in it that a
# shell would re-interpret.
_EVIDENCE_UNSAFE = str.maketrans({"\n": " ", "\r": " ", '"': "'", "`": "'", "$": "S"})


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


@dataclass(frozen=True)
class LaneRevision:
    """What the running dev-lane container reports about itself."""

    revision: str
    compose_project: str
    build_source: str
    state: str


@dataclass(frozen=True)
class Ancestry:
    """How the lane's observed revision relates to one expected merge sha.

    Two independent facts, because either alone is insufficient: ``relation``
    says whether the observed revision contains the merge sha, and
    ``observed_on_branch`` says whether the tracked branch contains the observed
    revision. A branch build can be a descendant of the merge sha and still be
    code the branch has never carried.
    """

    relation: str
    #: commits the observed revision has that the expected merge sha does not
    commits_ahead: int
    observed_on_branch: bool
    branch: str


@dataclass(frozen=True)
class Divergence:
    """How the deployed revision relates to the tracked branch head."""

    status: str  # GitHub compare status: identical | ahead | behind | diverged
    commits_behind: int  # commits on the branch the lane does not have
    head_sha: str
    base_sha: str
    base_committed_at: datetime


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_age(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    hours, remainder = divmod(max(total, 0), 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes:02d}m"


def assert_lane_fence(lane: LaneRevision, container: str, project: str) -> None:
    """Refuse to evaluate anything but the declared dev lane.

    Reading a governed lane (prod, stability-test, judge, lakshman) by accident
    is a boundary violation even when the read is read-only, so a project-label
    mismatch aborts rather than degrading to a warning.
    """
    if lane.compose_project != project:
        raise ValueError(
            f"lane fence: container {container!r} reports "
            f"{COMPOSE_PROJECT_LABEL}={lane.compose_project!r}, expected {project!r}. "
            "This guard evaluates the dev lane only and refuses to read any other "
            "compose project."
        )


def normalize_revision(raw: str) -> str:
    """Reduce a label value to a comparable revision, or raise.

    A blank or sentinel label is the shape a non-workspace or mislabelled build
    leaves behind. It is NOT evidence that the lane is current, so it raises
    instead of comparing.
    """
    value = raw.strip()
    if value in SENTINEL_REVISIONS or value.lower() in SENTINEL_REVISIONS:
        raise ValueError(
            f"{REVISION_LABEL} on the running container is {raw!r} — a blank or "
            "sentinel identity. The deployed SHA is unknown, so staleness cannot "
            "be proven either way; failing closed. See "
            "scripts/check_runtime_image_identity.py for the labelling contract."
        )
    if not _SHA_RE.match(value):
        raise ValueError(
            f"{REVISION_LABEL}={raw!r} is not a git SHA; refusing to compare an "
            "unrecognised identity."
        )
    return value


def revisions_match(deployed: str, expected: str) -> bool:
    """Compare a possibly-abbreviated deployed SHA against a full one.

    The label is stamped from ``git rev-parse --short`` on some build paths, so a
    strict equality test would report a converged lane as never-converged. The
    comparison is a prefix test on the SHORTER of the two, which is what git
    itself does for abbreviated revisions.
    """
    left = deployed.strip().lower()
    right = expected.strip().lower()
    if not left or not right:
        return False
    shortest = min(len(left), len(right))
    return left[:shortest] == right[:shortest]


def evaluate(
    lane: LaneRevision,
    divergence: Divergence,
    now: datetime,
    max_commits_behind: int,
    max_age: timedelta,
) -> Verdict:
    """Decide whether the dev lane's deployed SHA tracks ``dev``.

    Pure over already-collected facts so the whole verdict is exercised by
    fixtures without a docker daemon or a network.
    """
    verdict = Verdict()
    age = now - divergence.base_committed_at

    if lane.state != "running":
        verdict.findings.append(
            Finding(
                "LANE_NOT_RUNNING",
                f"{DEV_LANE_CONTAINER} is in state {lane.state!r}, not 'running'. A "
                "stopped lane cannot be tracking dev, and the lane is KEEP-ALIVE by "
                "operator ruling (OMN-15190). Recovery: "
                "knowledge-base:runbooks/cold-lane-full-bringup.md.",
            )
        )

    if divergence.status == "identical":
        verdict.notes.append(
            f"dev lane runs {divergence.base_sha[:12]} == {DEFAULT_BRANCH} head; "
            f"image age {_format_age(age)}"
        )
        return verdict

    if divergence.status == "diverged":
        verdict.findings.append(
            Finding(
                "DIVERGED",
                f"deployed revision {divergence.base_sha[:12]} is NOT on "
                f"{DEFAULT_BRANCH}'s line of history (compare status 'diverged'). The "
                "lane is running code that dev does not contain — a hand-built or "
                "branch build, not a delivery. Failing closed rather than reporting "
                "a commit distance that would be meaningless.",
            )
        )
        return verdict

    if divergence.status == "behind":
        verdict.findings.append(
            Finding(
                "LANE_AHEAD_OF_BRANCH",
                f"deployed revision {divergence.base_sha[:12]} is AHEAD of "
                f"{DEFAULT_BRANCH} head {divergence.head_sha[:12]}. Either dev was "
                "rewound or the lane was deployed from an unmerged ref; both are "
                "conditions this guard refuses to call healthy.",
            )
        )
        return verdict

    if divergence.status != "ahead":
        verdict.findings.append(
            Finding(
                "UNKNOWN_COMPARE_STATUS",
                f"compare returned status {divergence.status!r}, which this guard does "
                "not interpret. Failing closed rather than guessing.",
            )
        )
        return verdict

    # status == "ahead": the branch has commits the lane does not.
    breaches: list[str] = []
    if divergence.commits_behind > max_commits_behind:
        breaches.append(
            f"{divergence.commits_behind} commits behind (bound {max_commits_behind})"
        )
    if age > max_age:
        breaches.append(
            f"deployed image is {_format_age(age)} old (bound {_format_age(max_age)})"
        )

    if not breaches:
        verdict.notes.append(
            f"dev lane runs {divergence.base_sha[:12]}, "
            f"{divergence.commits_behind} commit(s) behind {DEFAULT_BRANCH} head "
            f"{divergence.head_sha[:12]}, image age {_format_age(age)} — inside both "
            f"bounds ({max_commits_behind} commits / {_format_age(max_age)})"
        )
        return verdict

    verdict.findings.append(
        Finding(
            "LANE_STALE",
            f"the dev lane does NOT track {DEFAULT_BRANCH}: running "
            f"{divergence.base_sha[:12]} ({lane.build_source} build) while "
            f"{DEFAULT_BRANCH} head is {divergence.head_sha[:12]} — "
            + "; ".join(breaches)
            + ". A redeploy was either never published, never delivered, or never "
            "applied. Do NOT widen the bounds to clear this (OMN-17888 AC5); find "
            "which layer dropped it — the publisher's run, the redeploy-start topic "
            "watermark, the orchestrator consumer group, or the runtime-effects DLQ.",
        )
    )
    return verdict


def evaluate_convergence(
    lane: LaneRevision,
    expected_revision: str,
    waited: timedelta,
    wait_timeout: timedelta,
    ancestry: Ancestry | None,
) -> Verdict:
    """Decide whether the lane is running code that contains one merge SHA.

    ``ancestry`` is the resolved relation between the lane's revision and
    ``expected_revision``; ``None`` means it could not be resolved, which is a
    finding of its own and never a pass. It is not consulted at all when the two
    revisions are byte-equal, so the common case costs no API call.
    """
    verdict = Verdict()
    if revisions_match(lane.revision, expected_revision):
        verdict.notes.append(
            f"dev lane converged onto {expected_revision[:12]} after "
            f"{_format_age(waited)}"
        )
        return verdict

    if ancestry is None:
        verdict.findings.append(
            Finding(
                "ANCESTRY_UNPROVABLE",
                f"the dev lane runs {lane.revision[:12]} and the merge commit is "
                f"{expected_revision[:12]}, but the relation between them could not "
                "be resolved, so whether the lane contains this change is unknown. "
                "An unreadable compare is not evidence of convergence; failing "
                "closed. Check that the job's GH_TOKEN can read "
                "repos/<repo>/compare, and that both revisions still resolve — a "
                "force-push or a garbage-collected build ref makes them unresolvable.",
            )
        )
        return verdict

    if not ancestry.observed_on_branch:
        verdict.findings.append(
            Finding(
                "LANE_OFF_BRANCH",
                f"the dev lane runs {lane.revision[:12]}, which {ancestry.branch} "
                f"does NOT contain (relation to the merge commit "
                f"{expected_revision[:12]}: {ancestry.relation}). That is a "
                "hand-built or branch image, not a delivery of merged code, and it "
                "is not converged even when it happens to contain the merge commit "
                "— the lane is carrying commits the branch has never had. Do not "
                "clear this by rebuilding from a branch ref; rebuild the lane from "
                f"{ancestry.branch}.",
            )
        )
        return verdict

    if ancestry.relation in _CONTAINING_RELATIONS:
        verdict.notes.append(
            f"dev lane runs {lane.revision[:12]}, which CONTAINS the merge commit "
            f"{expected_revision[:12]} ({ancestry.commits_ahead} commit(s) ahead on "
            f"{ancestry.branch}) — the lane has exercised this change. Converged "
            f"after {_format_age(waited)}; the receipt sha stays "
            f"{expected_revision[:12]}."
        )
        return verdict

    verdict.findings.append(
        Finding(
            "NOT_CONVERGED",
            f"after {_format_age(waited)} (bound {_format_age(wait_timeout)}) the dev "
            f"lane runs {lane.revision[:12]}, which does NOT contain the merge commit "
            f"{expected_revision[:12]} this run published a redeploy-start for "
            f"(relation: {ancestry.relation} on {ancestry.branch}). "
            + (
                "The lane is running code OLDER than the merge — this is the "
                "delivered-but-not-applied shape. "
                if ancestry.relation == RELATION_ANCESTOR
                else "The lane relation is not a containing relation, so this is "
                "a divergent or unclassified non-convergence shape rather than "
                "the delivered-but-not-applied ancestor case. "
            )
            + "A descendant revision would have "
            "been accepted, so this is not the OMN-18388 window. WHAT THIS RUN "
            "ACTUALLY ATTESTS is that one redeploy-start command reached the broker "
            "— NOT that a rebuild-requested command ever reached the deploy agent. "
            "Those are different topics with an orchestrator and node_redeploy's "
            "deploy effect between them, and that effect is SERIAL: it publishes one "
            "rebuild-requested, then polls for completion until its own timeout, so "
            "a backlog of correlations ahead of this merge delays it by that timeout "
            "each. Check in this order: (1) whether a rebuild-requested exists for "
            "this sha at all, and what else is queued ahead of it on "
            "onex.cmd.deploy.rebuild-requested.v1; (2) the runtime-effects DLQ "
            "(onex.dlq.omnibase-infra.omnimarket.v1); (3) the orchestrator consumer "
            "group. Measured 2026-09-10: the command for one merge did not appear on "
            "the agent's topic until 3h31m after this guard's window closed, while "
            "publish-to-accept once it existed was 268ms — so a red verdict here is "
            "usually about the hop BEFORE the agent, not the agent or its clone.",
        )
    )
    return verdict


class EnumConvergenceOutcome(StrEnum):
    """What the convergence guard concluded. Three values, not two.

    ``FAIL`` is a statement ABOUT THE LANE: it was granted its declared budget,
    measured from the moment the deploy agent accepted the rebuild command, and
    it did not come to run code containing the merge sha.

    ``INDETERMINATE`` is the honest answer when that budget could not be
    established, or when this job ran out of its own clock before the budget
    expired. Nothing has been shown about the lane. It still exits non-zero and
    still leaves the receipt non-PASS, so rule 24(b) stays closed; it changes
    what the receipt says, not what it permits.

    ``QUEUED`` is a statement about the QUEUE, and it is the one outcome that
    emits NO RECEIPT (OMN-18976). The wait never started, because the deploy
    agent holds commands ahead of this one and the horizon is longer than this
    window can outlast -- the refusal :func:`queue_exceeds_bound` already
    returns, whose own text ends "this asserts nothing about the lane".

    Why it cannot be ``INDETERMINATE``. That value maps onto a lab-pass check
    with ``ok: false``, and the receipt verdict rule is "PASS iff every check
    passed", so a queued merge was emitted as a terminal ``FAIL`` and rule
    24(b) refused a sha whose only fault was being second in line. Measured on
    the .201 dev lane over two consecutive merges, each probed while the lane
    still carried the PREVIOUS sha: artifact 10622202672 (``22a0ca18``, lane at
    ``08db8946``) and artifact 10623436027 (``4277d6e9``, lane at
    ``22a0ca18``), both FAIL on this check alone. Under a steady merge rate the
    queue never drains inside one verify window, so that was the normal outcome
    for a busy period, not an edge case.

    Why NO receipt rather than a fourth check outcome. :class:`EnumLabPassResult`
    is deliberately two-valued and states the contract: an in-flight or
    indeterminate lab pass emits no receipt at all rather than a ``PENDING``
    one, so the gate's "absent" branch and its "not yet passing" branch are the
    same branch and both fail closed. A merge whose turn has not come is in
    flight. **Nothing opens** -- the sha is still refused -- but the refusal is
    now recoverable by a later convergence instead of being contradicted
    forever by a FAIL artifact that says the lane misbehaved.
    """

    OK = "ok"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"
    QUEUED = "queued"


#: The exit code the convergence mode returns for an unestablished budget.
#: Distinct from 1 so a caller reading only the exit status can still tell the
#: two apart, and non-zero so nothing treats it as a pass.
EXIT_INDETERMINATE: Final[int] = 3

#: The exit code the convergence mode returns when the wait never started,
#: because the deploy agent's queue ahead of this command is longer than this
#: window can outlast (OMN-18976). Distinct from ``EXIT_INDETERMINATE`` because
#: the two call for different handling: an indeterminate run still writes a
#: receipt saying it learnt nothing, and a queued one writes NO receipt at all.
#: Non-zero, so nothing reads it as a pass.
EXIT_QUEUED: Final[int] = 4

_CORRELATION_ID_RE: Final = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class AcceptanceUnresolvedError(RuntimeError):
    """The deploy agent's acceptance of this run's command could not be read.

    Its own class so a caller cannot catch it alongside a transport error and
    fall through to a default start time. There is no default: a budget with a
    guessed start is the defect OMN-18573 removes, one layer down.
    """


class EnumLaneBuildOutcome(StrEnum):
    """Why the agent's runtime image build ended, as this job can read it.

    OMN-18615 AC4. A MIRROR of ``deploy_agent.build_budget.EnumBuildOutcome``,
    which this job cannot import: the deploy agent is a separate package that
    runs on the lab host, and a CI script reaching into it would couple the
    receipt surface to the agent's install. The two are pinned against each
    other by ``tests/ci/test_build_outcome_token_mirror_omn18615.py``, which
    fails if either side adds, removes or renames a value -- the same
    mirror-and-pin shape the prod-promotion grant schema uses across its own
    repo boundary.

    WHY A LANE NEEDS THIS AT ALL. A lab-pass receipt that reports a non-
    converged lane says nothing about WHY, and the two reasons lead to
    opposite next actions: a build killed by its own ceiling is worth
    re-running when the host is quieter, while a build that errored is not
    worth re-running at all. On 2026-09-17 a second rebuild was issued in
    exactly the wrong belief and reproduced the kill two seconds from the
    first.
    """

    BUDGET_EXHAUSTED = "runtime_image_build_budget_exhausted"
    BUILD_ERRORED = "runtime_image_build_errored"

    @classmethod
    def classify(cls, errors: Sequence[str] | None) -> EnumLaneBuildOutcome | None:
        """Read the outcome token off the agent's own ``errors`` list.

        ``None`` means the job's errors name no build outcome -- a lane-lock
        contention, a gateway refusal, or no error at all. It is never a
        default standing in for an outcome that could not be read.
        """
        for error in errors or ():
            for outcome in cls:
                if outcome.value in error:
                    return outcome
        return None


@dataclass(frozen=True)
class ModelAgentAcceptance:
    """When the deploy agent took the rebuild command, and where that was read.

    ``accepted_at`` is the agent's OWN record (``JobState.accepted_at``, served
    by its ``/job/{correlation_id}`` endpoint), not a CI-side observation of
    one. The agent writes it when it accepts the command, which is the instant
    the lane's clock legitimately starts.

    ``build_outcome`` (OMN-18615) is read from the SAME response, off the
    agent's ``errors`` list. It is evidence, not a verdict input: nothing in
    this module branches on it, and a receipt reports it so a reader can tell
    a killed build from a broken one without shelling to the lab host.
    """

    correlation_id: str
    accepted_at: datetime
    source: str
    build_outcome: EnumLaneBuildOutcome | None = None

    def __post_init__(self) -> None:
        if self.accepted_at.tzinfo is None:
            msg = "accepted_at must be timezone-aware"
            raise ValueError(msg)


@dataclass(frozen=True)
class ModelAcceptanceProbe:
    """One look at the agent: the acceptance, or why there is not one yet."""

    acceptance: ModelAgentAcceptance | None
    reason: str

    def __post_init__(self) -> None:
        if (self.acceptance is None) == (not self.reason):
            msg = (
                "an acceptance probe carries EXACTLY one of an acceptance and a "
                "reason. Both, or neither, makes 'not accepted yet' and 'could "
                "not ask' the same value, which is the collapse this ticket "
                "exists to undo."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class ModelConvergenceBudget:
    """How long the LANE is granted to converge, from when, and what the job can afford.

    ``declared_seconds`` is the lane's budget. It is measured from
    ``acceptance.accepted_at``, never from the step's start: the redeploy-start
    effect is serial, so the wait between the merge and the agent taking the
    command is a queue the lane is not responsible for, and spending the lane's
    budget on it produced a FAIL receipt for a lane that converged three
    minutes later (OMN-17214, twice).

    ``wall_clock_seconds`` is what THIS job can afford before it must stop and
    write the receipt -- an unwritten receipt is the one outcome worse than a
    failing one. It bounds the wait; it never redefines the budget. When the
    job clock expires first the verdict is INDETERMINATE, naming both numbers,
    exactly as ``ModelSettleBudget`` does for the settle phase.
    """

    declared_seconds: int
    wall_clock_seconds: int
    acceptance: ModelAgentAcceptance | None
    unresolved_reason: str

    def __post_init__(self) -> None:
        if (self.acceptance is None) == (not self.unresolved_reason):
            msg = (
                "a convergence budget carries EXACTLY one of an acceptance and "
                "an unresolved reason. A budget with neither cannot say why it "
                "is unestablished, and one with both is two answers."
            )
            raise ValueError(msg)
        if self.declared_seconds <= 0:
            msg = f"declared_seconds must be positive, got {self.declared_seconds}"
            raise ValueError(msg)

    @property
    def established(self) -> bool:
        return self.acceptance is not None

    def elapsed_since_acceptance(self, now: datetime) -> timedelta | None:
        """Time the lane has had. Never negative.

        The runner and the lab host are different machines, so a small forward
        skew would otherwise read as a budget that has not started. Clamping at
        zero costs the lane nothing and refuses to produce a negative age.
        """
        if self.acceptance is None:
            return None
        return max(timedelta(0), now - self.acceptance.accepted_at)

    def deadline(self) -> datetime | None:
        if self.acceptance is None:
            return None
        return self.acceptance.accepted_at + timedelta(seconds=self.declared_seconds)

    def exhausted(self, now: datetime) -> bool:
        elapsed = self.elapsed_since_acceptance(now)
        return elapsed is not None and elapsed.total_seconds() >= self.declared_seconds

    def acceptance_phrase(self, now: datetime) -> str:
        """The clause every verdict's evidence carries when acceptance is known."""
        if self.acceptance is None:
            return ""
        elapsed = self.elapsed_since_acceptance(now)
        assert elapsed is not None
        outcome = self.acceptance.build_outcome
        # OMN-18615: additive evidence. An absent token adds no clause at all,
        # so every message this job has ever emitted is byte-unchanged unless
        # the agent actually reported a build outcome.
        outcome_clause = (
            f"; the deploy agent reported {outcome.value}" if outcome else ""
        )
        return (
            f"{_format_age(elapsed)} since the deploy agent accepted "
            f"{self.acceptance.correlation_id} at "
            f"{self.acceptance.accepted_at.isoformat()} (budget "
            f"{self.declared_seconds}s from that moment)"
            f"{outcome_clause}"
        )

    def shortfall_reason(self, now: datetime) -> str:
        """Why a job that ran out of ITS clock is not a statement about the lane."""
        elapsed = self.elapsed_since_acceptance(now)
        observed = _format_age(elapsed) if elapsed is not None else "0s"
        return (
            f"this job could NOT afford the lane's declared budget: it watched "
            f"for {observed} of the {self.declared_seconds}s granted from "
            f"acceptance, bounded by its own {self.wall_clock_seconds}s wall "
            "clock, so the lane still had budget left when the job had to stop "
            "and write the receipt"
        )


def convergence_check_outcome(outcome: EnumConvergenceOutcome) -> Any:
    """Map a convergence verdict onto the lab-pass check outcome it becomes.

    Lives here rather than in the workflow's shell, because a mapping written
    in a ``run:`` block is a mapping nothing tests.

    ``QUEUED`` maps to ``None``, which means NO CHECK AND THEREFORE NO RECEIPT
    (OMN-18976). It is the only outcome that does. Every other value maps onto
    a check exactly as it did before, and ``KeyError`` on an unknown outcome is
    deliberate: a fifth verdict must decide this question rather than inherit
    an answer.
    """
    from scripts.ci.lab_pass_receipt import EnumLabPassCheckOutcome

    return {
        EnumConvergenceOutcome.OK: EnumLabPassCheckOutcome.PASS,
        EnumConvergenceOutcome.FAIL: EnumLabPassCheckOutcome.FAIL,
        EnumConvergenceOutcome.INDETERMINATE: EnumLabPassCheckOutcome.INDETERMINATE,
        EnumConvergenceOutcome.QUEUED: None,
    }[outcome]


def read_agent_acceptance(
    agent_url: str,
    correlation_id: str,
    *,
    request_timeout_seconds: float = 10.0,
    opener: Callable[[str, float], tuple[int, str]] | None = None,
) -> ModelAgentAcceptance:
    """Read the deploy agent's own acceptance record for one correlation id.

    The agent serves ``GET /job/{correlation_id}`` with ``accepted_at`` from
    its durable ``JobStore`` (``deploy_agent/health.py``, ``job_state.py``).
    This job already reaches that surface for the lab-overlay record
    (``scripts/ci/fetch_lab_overlay_record.py``), so no new credential, host or
    network path is introduced.

    WHY THE AGENT AND NOT THE BROKER. The guard's own topic-level question --
    "did a command for this merge reach the agent" -- would need a broker
    credential this job does not hold. OMN-18144 measured publish-to-accept on
    the agent's control topic at 268ms, 962ms and 11.7s, so the agent's record
    and the topic answer the same question to within the poll interval, and the
    agent's record is the one reachable from here.

    Every failure RAISES :class:`AcceptanceUnresolvedError`. None returns a default.
    """
    if not _CORRELATION_ID_RE.match(correlation_id.strip()):
        msg = (
            f"correlation id {correlation_id!r} is not a uuid. Refusing to "
            "interpolate it into a request path."
        )
        raise AcceptanceUnresolvedError(msg)

    url = f"{agent_url.rstrip('/')}/job/{correlation_id.strip()}"
    fetch = opener or _http_get_json
    try:
        status, body = fetch(url, request_timeout_seconds)
    except Exception as exc:
        msg = f"{url} could not be read: {type(exc).__name__}: {exc}"
        raise AcceptanceUnresolvedError(msg) from exc

    if status == 404:
        msg = (
            f"the deploy agent reports no job for correlation {correlation_id} "
            f"({url} -> HTTP 404). The command has not been handed to the agent "
            "yet, so the lane's budget has not started."
        )
        raise AcceptanceUnresolvedError(msg)
    if status != 200:
        msg = f"{url} answered HTTP {status}: {body[:200]}"
        raise AcceptanceUnresolvedError(msg)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        msg = f"{url} returned an unreadable body ({exc}): {body[:200]}"
        raise AcceptanceUnresolvedError(msg) from exc
    if not isinstance(payload, dict) or "accepted_at" not in payload:
        msg = f"{url} returned no accepted_at field: {body[:200]}"
        raise AcceptanceUnresolvedError(msg)
    try:
        accepted_at = _parse_ts(str(payload["accepted_at"]))
    except ValueError as exc:
        msg = f"{url} returned an unparseable accepted_at {payload['accepted_at']!r}"
        raise AcceptanceUnresolvedError(msg) from exc
    raw_errors = payload.get("errors")
    errors = [str(item) for item in raw_errors] if isinstance(raw_errors, list) else []
    return ModelAgentAcceptance(
        correlation_id=correlation_id.strip(),
        accepted_at=accepted_at,
        source=url,
        build_outcome=EnumLaneBuildOutcome.classify(errors),
    )


@dataclass(frozen=True)
class ModelQueueFacts:
    """This command's place in the deploy agent's queue, or why that is unknown.

    OMN-18144. OMN-18573 anchored the LANE's budget to the agent's acceptance,
    which fixed everything after acceptance and left the wait BEFORE it bounded
    by nothing but this job's own ceiling. Measured 2026-09-18: four runtime
    merges inside 33 minutes against an agent servicing ~32 minutes per
    command put the fourth merge third in line; its ``/job`` answered 404 for
    the whole window and the receipt for ``11e8951f`` read FAIL against a lane
    that was healthy and strictly monotone throughout.

    Exactly one of a readable queue and an ``unread_reason``. An unread queue
    and an empty one must not be the same value -- that collapse is the defect,
    one layer up.

    These are EVIDENCE. Nothing here is a verdict input in the sense the
    receipt means: the lab-pass verdict stays derived from the checks, and the
    only thing this decides is whether the wait is worth starting.
    """

    commands_ahead: int | None
    mean_service_time_seconds: float | None
    service_sample_size: int
    in_flight_correlation_id: str | None
    unread_reason: str
    source: str

    def __post_init__(self) -> None:
        readable = self.commands_ahead is not None
        if readable == bool(self.unread_reason):
            msg = (
                "queue facts carry EXACTLY one of a commands-ahead count and "
                f"an unread reason; got commands_ahead={self.commands_ahead!r}, "
                f"unread_reason={self.unread_reason!r}"
            )
            raise ValueError(msg)

    @classmethod
    def unread(cls, reason: str, source: str = "") -> ModelQueueFacts:
        return cls(
            commands_ahead=None,
            mean_service_time_seconds=None,
            service_sample_size=0,
            in_flight_correlation_id=None,
            unread_reason=reason,
            source=source,
        )

    @property
    def queue_position_at_start(self) -> int | None:
        """This command's 1-based place in line. ``1`` means nothing is ahead of it."""
        if self.commands_ahead is None:
            return None
        return self.commands_ahead + 1

    def derived_wait_bound_seconds(
        self, *, lane_budget_seconds: int, margin_seconds: int
    ) -> int | None:
        """How long this run would have to watch for, given the queue.

        ``commands_ahead x mean service time`` is the wait for the agent to
        REACH this command; the lane's declared budget is what it gets after
        that; the margin is this guard's own poll interval, because it cannot
        notice anything sooner than it looks.

        Every term is declared or measured. ``None`` when the queue could not
        be read, or when this agent has completed nothing to take a mean over
        -- a bound derived from an absent service time would be a guess wearing
        a number's clothes.
        """
        if self.commands_ahead is None or self.mean_service_time_seconds is None:
            return None
        queue_wait = self.commands_ahead * self.mean_service_time_seconds
        return round(queue_wait + lane_budget_seconds + margin_seconds)

    def reach_bound_seconds(self, *, margin_seconds: int) -> int | None:
        """How long before the agent could REACH this command.

        Acceptance only. This is NOT the predicate -- see
        :meth:`convergence_horizon_seconds`, which is this plus the one more
        service time the lane needs before it carries the sha. Kept as its own
        term because the horizon is defined in relation to it and the two are
        pinned against each other in the tests.

        ``None`` on an unreadable queue: a bound with no measured service time
        behind it is a guess wearing a number's clothes.
        """
        if self.commands_ahead is None or self.mean_service_time_seconds is None:
            return None
        return round(
            self.commands_ahead * self.mean_service_time_seconds + margin_seconds
        )

    def convergence_horizon_seconds(self, *, margin_seconds: int) -> int | None:
        """How long before the LANE could be expected to carry this sha.

        This is the refusal predicate. OMN-18144, and the second revision of it
        in one day -- the reasoning of the first is recorded here because the
        correction is only legible against it.

        ``omnibase_infra#3823`` (12:33:02Z) moved the predicate off
        :meth:`derived_wait_bound_seconds` -- reach plus the lane's declared
        1500s post-acceptance grant -- and onto the reach alone, arguing that
        reaching acceptance is what makes a measurement possible. Half of that
        was right and half was wrong.

        RIGHT: the lane's ``--wait-timeout`` grant does not belong here. It is
        a declared ceiling, not an estimate of anything this command will
        spend, and the job ceiling funds it SEPARATELY for the probe step that
        runs after this one (``2700 - 1500 - 900 - 120``, see
        :data:`lane_settle_budget.STEP_OVERHEAD_SECONDS`). It stays out.

        WRONG: the reach is not the horizon. ``mean_service_time_seconds`` is
        the agent's ACCEPT-TO-COMPLETION time, so arriving at acceptance still
        leaves this command's own full service before the lane carries the
        sha. Lane ``post-merge-lab-verify-reds-diag-1230`` measured the cost
        within the hour: at ``commands_ahead = 1`` the reach predicate starts a
        watch that cannot converge, holds the single host-201 verify runner for
        the whole 1620s window, and arrives at the same INDETERMINATE the
        refusal writes in about a second -- with the next merge's guard queued
        behind it. Two of the four runs measured that morning sat at exactly
        that position, so it is the common case.

        So the horizon charges one measured service per place in line:
        ``queue_position_at_start x mean + margin``. This command is Nth, each
        takes a mean, convergence is at N means.

        AN EMPTY QUEUE IS NOT A HORIZON QUESTION and is handled by the caller,
        which never refuses on one. Charging a full mean to a command with
        nothing ahead of it would refuse the healthy path whenever the agent's
        mean drifted above the window -- and that path is the one that
        normally PASSES.

        ``None`` on an unreadable queue, for the same reason as above.
        """
        if self.commands_ahead is None or self.mean_service_time_seconds is None:
            return None
        position = self.commands_ahead + 1
        return round(position * self.mean_service_time_seconds + margin_seconds)

    def evidence_clause(self, *, lane_budget_seconds: int, margin_seconds: int) -> str:
        """The named fields AC5 requires the receipt's check to carry."""
        if self.commands_ahead is None:
            return (
                "queue_position_at_start=UNREAD commands_ahead=UNREAD "
                "derived_wait_bound_s=UNREAD mean_service_time_s=UNREAD "
                f"(the deploy agent's queue could not be read: {self.unread_reason}; "
                "the wait fell back to this job's wall-clock bound)"
            )
        bound = self.derived_wait_bound_seconds(
            lane_budget_seconds=lane_budget_seconds, margin_seconds=margin_seconds
        )
        mean = self.mean_service_time_seconds
        return (
            f"queue_position_at_start={self.queue_position_at_start} "
            f"commands_ahead={self.commands_ahead} "
            f"derived_wait_bound_s={bound if bound is not None else 'UNDERIVABLE'} "
            f"mean_service_time_s={f'{mean:.1f}' if mean is not None else 'UNREAD'} "
            f"(over {self.service_sample_size} completed job(s), read from "
            f"{self.source or 'the deploy agent'})"
        )


def read_agent_queue(
    agent_url: str,
    *,
    request_timeout_seconds: float = 10.0,
    opener: Callable[[str, float], tuple[int, str]] | None = None,
) -> ModelQueueFacts:
    """Read the deploy agent's ``/queue`` surface. Never raises.

    Every failure comes back as unread facts carrying the reason in words,
    because this read is additive: a guard that died because the queue surface
    was unreachable would be strictly worse than the clock-bounded guard it
    replaces. A 404 specifically means an agent too old to serve the route --
    it self-updates from ``dev``, so this is the normal state between a merge
    and the agent's next re-exec, and it is named as such rather than reported
    as an outage.
    """
    if not agent_url.strip():
        return ModelQueueFacts.unread("no deploy-agent URL was supplied to this guard")
    url = f"{agent_url.rstrip('/')}/queue"
    fetch = opener or _http_get_json
    try:
        status, body = fetch(url, request_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - transport failure IS the evidence
        return ModelQueueFacts.unread(
            f"{url} could not be read: {type(exc).__name__}: {exc}", source=url
        )
    if status == 404:
        return ModelQueueFacts.unread(
            f"{url} answered HTTP 404, so this deploy agent predates the queue "
            "endpoint and cannot report its depth",
            source=url,
        )
    if status != 200:
        return ModelQueueFacts.unread(
            f"{url} answered HTTP {status}: {body[:200]}", source=url
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return ModelQueueFacts.unread(
            f"{url} returned an unreadable body ({exc}): {body[:200]}", source=url
        )
    if not isinstance(payload, dict):
        return ModelQueueFacts.unread(
            f"{url} returned {type(payload).__name__}, not an object", source=url
        )
    ahead = payload.get("commands_ahead")
    if ahead is None:
        return ModelQueueFacts.unread(
            f"{url} reports its depth unknown: "
            f"{payload.get('control_topic_lag_reason') or 'no reason given'}",
            source=url,
        )
    if not isinstance(ahead, int) or isinstance(ahead, bool) or ahead < 0:
        return ModelQueueFacts.unread(
            f"{url} returned commands_ahead={ahead!r}, which is not a count",
            source=url,
        )
    raw_mean = payload.get("mean_service_time_seconds")
    mean = (
        float(raw_mean)
        if isinstance(raw_mean, (int, float)) and not isinstance(raw_mean, bool)
        else None
    )
    if mean is not None and mean <= 0:
        mean = None
    raw_sample = payload.get("service_sample_size")
    sample = raw_sample if isinstance(raw_sample, int) and raw_sample >= 0 else 0
    raw_in_flight = payload.get("in_flight_correlation_id")
    return ModelQueueFacts(
        commands_ahead=ahead,
        mean_service_time_seconds=mean if sample else None,
        service_sample_size=sample if mean else 0,
        in_flight_correlation_id=(
            str(raw_in_flight) if isinstance(raw_in_flight, str) else None
        ),
        unread_reason="",
        source=url,
    )


def queue_exceeds_bound(
    facts: ModelQueueFacts,
    *,
    lane_budget_seconds: int,
    wall_clock_seconds: int,
    margin_seconds: int,
) -> str:
    """Why this run cannot afford the queue, or ``""`` when it can.

    The point of answering this BEFORE the wait rather than after it. Today a
    queued-behind merge spends its whole affordable clock -- 1680s, the
    45-minute job ceiling less the declared settle budget and the reserved
    tail, on the one physical verify runner on the lab host -- to arrive at an
    INDETERMINATE that names no cause. Measured at the parent commit by
    replaying the 2026-09-18 timeline through this same loop: 0h28m watched,
    verdict INDETERMINATE, evidence naming no depth. Answering up front costs seconds, frees the runner for
    the next merge's guard (AC4), and the receipt says "third in line" instead
    of "ran out of clock".

    An unreadable queue is NOT a refusal. Falling back to the clock is exactly
    today's behaviour, which is the right fallback: the change may only ever
    make a verdict better informed, never harder to obtain.

    WHICH BOUND DECIDES, AND WHY IT IS NOT THE ONE IN THE EVIDENCE
    --------------------------------------------------------------
    OMN-18144, corrected twice on 2026-09-19. The question is "could the LANE
    be expected to carry this sha inside this window", so the predicate is
    :meth:`ModelQueueFacts.convergence_horizon_seconds` -- one measured
    service per place in line. That method carries the full reasoning,
    including why the reach alone (``omnibase_infra#3823``, 12:33:02Z) was too
    permissive and why the lane's declared grant stays out.

    TWO CASES NEVER REFUSE, and both are deliberate:

    * an EMPTY queue. A refusal is about a queue ahead of this run, and there
      is none; it is also the case that normally PASSES, so a horizon applied
      to it would refuse the healthy path whenever the agent's mean drifted
      above the window.
    * an UNREADABLE queue. Falling back to the clock is today's behaviour and
      the right fallback: this may only ever make a verdict better informed,
      never harder to obtain.

    The refusal that remains is the one AC4 asked for: a queue this window
    cannot outlast is a foregone conclusion, and the single verify runner is
    better given to the next merge's guard than spent arriving at it.
    """
    if not facts.commands_ahead:
        return ""
    horizon = facts.convergence_horizon_seconds(margin_seconds=margin_seconds)
    if horizon is None or horizon <= wall_clock_seconds:
        return ""
    bound = facts.derived_wait_bound_seconds(
        lane_budget_seconds=lane_budget_seconds, margin_seconds=margin_seconds
    )
    assert facts.mean_service_time_seconds is not None
    return (
        f"the deploy agent has {facts.commands_ahead} command(s) ahead of this "
        f"one, so this merge is number {facts.queue_position_at_start} in line. "
        f"At the agent's observed {facts.mean_service_time_seconds:.0f}s mean "
        f"service time that is {horizon}s before the lane could be expected to "
        f"carry this sha -- one service for each place in line, this command's "
        f"own included -- against the {wall_clock_seconds}s this job can watch "
        f"for (the receipt's worst case, which also charges the lane's declared "
        f"grant, is {bound}s). The wait is not started: the lane is not late, "
        "this run cannot outlast the queue ahead of it, and holding the verify "
        "runner open for the difference would delay the next merge's guard "
        "behind this one"
    )


def _http_get_json(url: str, timeout_seconds: float) -> tuple[int, str]:
    """GET one JSON surface. HTTP errors come back as a status, not an exception."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
            return int(response.status), response.read().decode(
                "utf-8", errors="replace"
            )
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ModelConvergenceResult:
    """Everything the emitting step needs, structured rather than re-derived."""

    outcome: EnumConvergenceOutcome
    reason: str
    lane: LaneRevision
    ancestry: Ancestry | None
    budget: ModelConvergenceBudget
    waited: timedelta
    finished_at: datetime
    #: OMN-18144. The queue this command was behind when the wait started, or
    #: the named reason it could not be read. Always present, on every verdict,
    #: because a PASS that does not say what the queue looked like cannot be
    #: used to check that a later non-PASS was really the queue's doing.
    queue: ModelQueueFacts | None = None
    #: OMN-18976. The revision the lane was ALREADY at when this run started.
    #: With the revision it converged at, it bounds the set of merges that were
    #: queued behind this one -- see :func:`reemission_window`, which is the
    #: only consumer and which refuses to produce a window without both ends.
    initial_revision: str = ""


def run_convergence_wait(
    *,
    expected_revision: str,
    read_lane: Callable[[], LaneRevision],
    resolve_ancestry: Callable[[str], Ancestry | None],
    resolve_acceptance: Callable[[], ModelAcceptanceProbe],
    declared_budget: timedelta,
    wall_clock: timedelta,
    poll_interval: timedelta,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    resolve_queue: Callable[[], ModelQueueFacts] | None = None,
) -> ModelConvergenceResult:
    """Wait for the lane to contain the merge sha, on the LANE's clock.

    The loop runs until one of three things is true, and the three are the
    three verdicts:

    * the lane contains the merge sha -- ``OK``;
    * acceptance is known and its budget is spent -- ``FAIL``, a statement
      about the lane;
    * this job's own wall clock expires first, or acceptance was never
      established -- ``INDETERMINATE``, a statement about the run.

    The wall clock is a BOUND, never the budget. It exists because the job must
    survive to write the receipt; it is not widened to absorb a queue, and when
    it is what stopped the wait, the verdict says so rather than blaming the
    lane.

    Every moving part is injected so the three shapes are testable without a
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

    lane = read_lane()
    ancestry = resolve_ancestry(lane.revision)
    # OMN-18976. The lane's revision BEFORE this run's rebuild landed. Read
    # once, here, because every later read may already have converged.
    initial_revision = lane.revision

    queue = (
        resolve_queue()
        if resolve_queue is not None
        else ModelQueueFacts.unread(
            "this guard was invoked with no queue reader, so the commands "
            "ahead of this one were never asked for"
        )
    )

    def _converged(observed: LaneRevision, relation: Ancestry | None) -> bool:
        if revisions_match(observed.revision, expected_revision):
            return True
        if relation is None or not relation.observed_on_branch:
            return False
        return relation.relation in _CONTAINING_RELATIONS

    # OMN-18144. Refuse the wait up front when the queue ahead of this command
    # is longer than this run can outlast.
    #
    # Scoped to an UNESTABLISHED acceptance on purpose, and the scope is what
    # makes this additive. Once the agent has accepted this command the queue
    # ahead of it is spent, the lane's own budget governs (OMN-18573), and this
    # branch is unreachable -- so every run whose command was already accepted
    # behaves byte-for-byte as it did before. What it changes is the one case
    # it was built for: a command the agent has not reached, behind a queue
    # whose length is now readable.
    if not budget.established and not _converged(lane, ancestry):
        refusal = queue_exceeds_bound(
            queue,
            lane_budget_seconds=int(declared_budget.total_seconds()),
            wall_clock_seconds=int(wall_clock.total_seconds()),
            margin_seconds=int(poll_interval.total_seconds()),
        )
        if refusal:
            now = clock()
            # OMN-18976. QUEUED, not INDETERMINATE. The refusal above ends
            # "this asserts nothing about the lane", and INDETERMINATE maps
            # onto a check with ``ok: false``, so this branch was emitting a
            # terminal FAIL receipt and rule 24(b) was refusing a sha whose
            # only fault was being behind another deploy. QUEUED emits no
            # receipt at all, which is what EnumLabPassResult's own contract
            # says an in-flight pass does, and the gate's absent branch keeps
            # the sha refused exactly as before.
            return ModelConvergenceResult(
                initial_revision=initial_revision,
                outcome=EnumConvergenceOutcome.QUEUED,
                reason=refusal,
                lane=lane,
                ancestry=ancestry,
                budget=budget,
                waited=now - started,
                finished_at=now,
                queue=queue,
            )

    while not _converged(lane, ancestry):
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
        lane = read_lane()
        ancestry = resolve_ancestry(lane.revision)

    now = clock()
    waited = now - started
    if _converged(lane, ancestry):
        return ModelConvergenceResult(
            initial_revision=initial_revision,
            outcome=EnumConvergenceOutcome.OK,
            reason="",
            lane=lane,
            ancestry=ancestry,
            budget=budget,
            waited=waited,
            finished_at=now,
            queue=queue,
        )
    if not budget.established:
        return ModelConvergenceResult(
            initial_revision=initial_revision,
            outcome=EnumConvergenceOutcome.INDETERMINATE,
            reason=budget.unresolved_reason,
            lane=lane,
            ancestry=ancestry,
            budget=budget,
            waited=waited,
            finished_at=now,
            queue=queue,
        )
    if budget.exhausted(now):
        return ModelConvergenceResult(
            initial_revision=initial_revision,
            outcome=EnumConvergenceOutcome.FAIL,
            reason="",
            lane=lane,
            ancestry=ancestry,
            budget=budget,
            waited=waited,
            finished_at=now,
            queue=queue,
        )
    return ModelConvergenceResult(
        initial_revision=initial_revision,
        outcome=EnumConvergenceOutcome.INDETERMINATE,
        reason=budget.shortfall_reason(now),
        lane=lane,
        ancestry=ancestry,
        budget=budget,
        waited=waited,
        finished_at=now,
        queue=queue,
    )


def convergence_evidence(
    lane: LaneRevision,
    expected_revision: str,
    ancestry: Ancestry | None,
    waited: timedelta,
    converged: bool,
    budget: ModelConvergenceBudget | None = None,
    now: datetime | None = None,
    indeterminate_reason: str = "",
    queue_clause: str = "",
) -> str:
    """Render the one-line evidence the lab-pass receipt's check carries.

    OMN-18388 AC2: the ``deployed_revision`` check must name BOTH shas and the
    ancestry relation, so a reader of the receipt can tell "the lane ran a newer
    dev head that contains this change" apart from "the lane never applied it".
    The static string it replaces named neither and was identical on every run.

    OMN-18573 adds the budget's own terms. When acceptance is known the line
    carries the agent's timestamp and the elapsed time since it, on EVERY
    verdict -- a passing receipt that does not say when the lane's clock
    started cannot be used to check that the clock started in the right place.
    When the verdict is INDETERMINATE the line says so in those words and
    carries the reason, because "the lane did not converge" and "we could not
    establish whether it had a chance to" are different findings.
    """
    observed = lane.revision[:12]
    expected = expected_revision[:12]
    waited_text = _format_age(waited)
    reference = now or datetime.now(UTC)
    acceptance_clause = ""
    if budget is not None:
        phrase = budget.acceptance_phrase(reference)
        acceptance_clause = f"; {phrase}" if phrase else ""

    # OMN-18144 AC5: the four queue fields ride on EVERY verdict, not only the
    # one they explain. A PASS that does not say what the queue looked like
    # cannot be used to check that a later non-PASS was really the queue's
    # doing, and that comparison is the whole reason the fields are recorded.
    queue_suffix = f" {queue_clause}" if queue_clause else ""

    if indeterminate_reason:
        line = (
            f"INDETERMINATE: lane at {observed}; whether it contains merge sha "
            f"{expected} was not established after {waited_text}{acceptance_clause}. "
            f"Reason: {indeterminate_reason}. This asserts nothing about the "
            "lane; the receipt is still non-PASS."
        )
        return " ".join((line + queue_suffix).translate(_EVIDENCE_UNSAFE).split())

    if converged and revisions_match(lane.revision, expected_revision):
        line = (
            f"lane at {observed} == merge sha {expected}, converged after "
            f"{waited_text} (check_dev_lane_staleness.py --expect-revision "
            f"{expected_revision} against the omnibase-infra compose project)"
        )
    elif ancestry is None:
        line = (
            f"lane at {observed}; its ancestry against merge sha {expected} could "
            f"not be resolved after {waited_text}, so convergence is unproven "
            "(failing closed)"
        )
    elif not ancestry.observed_on_branch:
        line = (
            f"lane at {observed}, which is not contained in {ancestry.branch} "
            f"(relation to merge sha {expected}: {ancestry.relation}); not "
            f"converged after {waited_text}"
        )
    elif ancestry.relation in _CONTAINING_RELATIONS:
        line = (
            f"lane at {observed}, which contains merge sha {expected} "
            f"({ancestry.commits_ahead} commit(s) ahead on {ancestry.branch}); "
            f"converged after {waited_text}"
        )
    else:
        line = (
            f"lane at {observed}, which does not contain merge sha {expected} "
            f"(relation: {ancestry.relation} on {ancestry.branch}); not converged "
            f"after {waited_text}"
        )
    return " ".join(
        (line + acceptance_clause + queue_suffix).translate(_EVIDENCE_UNSAFE).split()
    )


def _run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} {' '.join(argv[1:])} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def read_lane_revision(container: str) -> LaneRevision:
    """Read the running container's identity labels. Read-only; never mutates."""
    raw = _run(
        [
            "docker",
            "inspect",
            container,
            "--format",
            "{{json .}}",
        ]
    )
    return parse_docker_inspect(json.loads(raw))


def parse_docker_inspect(payload: Any) -> LaneRevision:
    """Project one ``docker inspect`` object onto the fields the verdict uses.

    Split out from :func:`read_lane_revision` so the incident replay drives this
    real projection over the captured bytes of the running container rather than
    a hand-built dict. A replay that reconstructs the shape by hand proves the
    verdict logic and nothing about whether the guard can read what docker
    actually returns.
    """
    if isinstance(payload, list):  # some docker versions wrap in a list
        payload = payload[0]
    labels = payload.get("Config", {}).get("Labels") or {}
    return LaneRevision(
        revision=str(labels.get(REVISION_LABEL, "")),
        compose_project=str(labels.get(COMPOSE_PROJECT_LABEL, "")),
        build_source=str(labels.get("com.omninode.build_source", "unknown")),
        state=str(payload.get("State", {}).get("Status", "unknown")),
    )


def _gh(args: list[str]) -> Any:
    return json.loads(_run(["gh", *args]))


def read_contained_commits(repo: str, previous: str, converged: str) -> tuple[str, ...]:
    """The commits ``converged`` has that ``previous`` does not, oldest first.

    OMN-18976. Feeds :func:`reemission_window`, which bounds them further. Uses
    ``compare/{previous}...{converged}`` for the same reason
    :func:`read_divergence` does: ``actions/checkout`` fetches depth 1, so a
    local ``git rev-list`` here would silently report an empty window and this
    guard would emit nothing while believing it had looked.

    UNREADABLE IS EMPTY, DELIBERATELY. Every failure returns ``()``, which
    re-emits nothing and leaves the delivery gate refusing those shas exactly
    as it does today. The alternative -- guessing a window -- writes receipts
    for merges nothing observed, and a wrong PASS is the one outcome this
    ticket must not introduce while removing a wrong FAIL.

    The comparison's ``commits`` array is capped at 250 entries by GitHub. That
    is not a limit worth working around: a window of 250 merges is not a deploy
    queue, it is a lane that stopped converging, and the receipts those shas
    are missing are not this function's problem to manufacture.
    """
    try:
        payload = _gh(["api", f"repos/{repo}/compare/{previous}...{converged}"])
    except Exception as exc:  # noqa: BLE001 - an unreadable compare IS the answer
        print(f"::warning::re-emission window unreadable: {exc}")
        return ()
    commits = payload.get("commits")
    if not isinstance(commits, list):
        return ()
    shas: list[str] = []
    for entry in commits:
        sha = entry.get("sha") if isinstance(entry, dict) else None
        if isinstance(sha, str) and _SHA_RE.match(sha):
            shas.append(sha)
    return tuple(shas)


def read_divergence(repo: str, branch: str, deployed_revision: str) -> Divergence:
    """Compare the deployed revision against the branch head via the GitHub API.

    Uses ``compare/{deployed}...{branch}`` rather than a local ``git rev-list``
    so the guard needs no deep clone: ``actions/checkout`` fetches depth 1 by
    default, and a guard that silently reported 0 commits behind because it had
    no history would be the same class of false green this ticket is about.

    ``ahead_by`` on that comparison is the number of commits the BRANCH has that
    the deployed revision does not — i.e. how far behind the lane is. The head
    SHA is resolved by its own call rather than taken from ``commits[-1]``, which
    is capped at 250 entries and is empty when the two are identical.
    """
    # --jq on a string field emits the bare value, not JSON, so this reads the
    # raw stdout rather than going through _gh()'s json.loads.
    head_sha = _run(
        ["gh", "api", f"repos/{repo}/commits/{branch}", "--jq", ".sha"]
    ).strip()
    if not _SHA_RE.match(head_sha):
        raise ValueError(f"could not resolve {branch} head: gh returned {head_sha!r}")
    payload = _gh(["api", f"repos/{repo}/compare/{deployed_revision}...{head_sha}"])
    return parse_compare(payload, head_sha)


def parse_compare(payload: Any, head_sha: str) -> Divergence:
    """Project one GitHub compare response onto the fields the verdict uses.

    Split out for the same reason as :func:`parse_docker_inspect`: the replay
    drives it over GitHub's own captured bytes, so the guard is proven able to
    read the real response and not only the shape a test author imagined.
    """
    return Divergence(
        status=str(payload["status"]),
        commits_behind=int(payload["ahead_by"]),
        head_sha=head_sha,
        base_sha=str(payload["base_commit"]["sha"]),
        base_committed_at=_parse_ts(
            payload["base_commit"]["commit"]["committer"]["date"]
        ),
    )


def read_ancestry(
    repo: str, branch: str, expected_revision: str, observed_revision: str
) -> Ancestry:
    """Resolve how the lane's revision relates to one merge SHA, and to ``branch``.

    Two compare calls rather than a local ``git merge-base``, for the same reason
    :func:`read_divergence` uses the API: the verify job checks out at depth 1, so
    a local ancestry test would be answering from history the runner does not
    have. Fetching enough depth to answer it would mean a full clone of a busy
    repository on every merge, on the single ``omnibase-deploy`` runner slot.

    No new credential: this is the ``GITHUB_TOKEN`` the job already holds for
    :func:`read_divergence`.
    """
    relation_payload = _gh(
        ["api", f"repos/{repo}/compare/{expected_revision}...{observed_revision}"]
    )
    containment_payload = _gh(
        ["api", f"repos/{repo}/compare/{observed_revision}...{branch}"]
    )
    return parse_ancestry(relation_payload, containment_payload, branch)


def parse_ancestry(
    relation_payload: Any, containment_payload: Any, branch: str
) -> Ancestry:
    """Project two GitHub compare responses onto the ancestry the verdict uses.

    Split out for the same reason as :func:`parse_compare`: the tests drive it
    over GitHub's own captured bytes, so the guard is proven able to read the
    real responses. An unrecognised compare status raises rather than defaulting
    to a relation — a guard that guessed here would guess in the direction of
    "converged", which is the failure this module exists to prevent.
    """
    status = str(relation_payload["status"])
    relation = _RELATION_BY_COMPARE_STATUS.get(status)
    if relation is None:
        raise ValueError(
            f"compare returned status {status!r}, which this guard does not "
            "interpret as an ancestry relation. Failing closed rather than guessing."
        )
    containment_status = str(containment_payload["status"])
    if containment_status not in _RELATION_BY_COMPARE_STATUS:
        raise ValueError(
            f"branch containment compare returned status {containment_status!r}, "
            "which this guard does not interpret. Failing closed rather than "
            "guessing."
        )
    return Ancestry(
        relation=relation,
        commits_ahead=int(relation_payload.get("ahead_by", 0)),
        observed_on_branch=containment_status in _CONTAINED_COMPARE_STATUSES,
        branch=branch,
    )


def _parse_duration(value: str) -> timedelta:
    """Parse ``90s`` / ``25m`` / ``2h`` into a timedelta."""
    match = re.fullmatch(r"(\d+)([smh])", value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a duration; use e.g. 90s, 25m, 2h"
        )
    amount, unit = int(match.group(1)), match.group(2)
    return {
        "s": timedelta(seconds=amount),
        "m": timedelta(minutes=amount),
        "h": timedelta(hours=amount),
    }[unit]


def _report(verdict: Verdict) -> int:
    for note in verdict.notes:
        print(f"ok: {note}")
    if verdict.ok:
        return 0
    for finding in verdict.findings:
        print(f"::error::{finding.render()}")
    return 1


def _summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OMN-17888 AC4 dev-lane staleness guard"
    )
    parser.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO)
    )
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--container", default=DEV_LANE_CONTAINER)
    parser.add_argument("--compose-project", default=DEV_LANE_COMPOSE_PROJECT)
    parser.add_argument(
        "--max-commits-behind",
        type=int,
        default=DEFAULT_MAX_COMMITS_BEHIND,
        help="AC4 states no bound; this default is a stated policy choice",
    )
    parser.add_argument(
        "--max-age",
        type=_parse_duration,
        default=DEFAULT_MAX_AGE,
        help="AC4 states no bound; this default is a stated policy choice",
    )
    parser.add_argument(
        "--expect-revision",
        default="",
        help="convergence mode: poll until the lane reports exactly this SHA",
    )
    parser.add_argument(
        "--wait-timeout", type=_parse_duration, default=DEFAULT_CONVERGENCE_WAIT
    )
    parser.add_argument(
        "--poll-interval", type=_parse_duration, default=DEFAULT_POLL_INTERVAL
    )
    # OMN-18573. The three inputs that move the convergence budget off this
    # step's own clock and onto the lane's. None of them widens the budget:
    # --wait-timeout is still the lane's grant, and --wall-clock-seconds only
    # says how long THIS job may watch before it must write the receipt.
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
        default=int(DEFAULT_CONVERGENCE_WAIT.total_seconds()),
        help=(
            "how long THIS job may watch before it must stop and let the "
            "receipt be written. A BOUND on the wait, never the lane's budget: "
            "when it is what stopped the wait, the verdict is INDETERMINATE "
            "naming both numbers, not a FAIL blaming the lane."
        ),
    )
    parser.add_argument(
        "--deployed-revision",
        default="",
        help="skip the docker read and evaluate this revision instead (positive control)",
    )
    parser.add_argument(
        "--positive-control",
        action="store_true",
        help=(
            "invert the exit contract: exit 0 only if the supplied "
            "--deployed-revision is reported STALE. Proves the guard fires."
        ),
    )
    args = parser.parse_args(argv)

    if args.positive_control and not args.deployed_revision:
        print("::error::--positive-control requires --deployed-revision")
        return 1

    try:
        if args.expect_revision:
            return _run_convergence_mode(args)
        return _run_staleness_mode(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        # Fail closed: an unreadable container or a failed API call is not
        # evidence that the lane tracks dev.
        print(f"::error::dev-lane staleness guard could not evaluate: {exc}")
        _summary(["## Dev-lane staleness (OMN-17888 AC4)", "", f"UNPROVABLE: {exc}"])
        return 1


def _read_lane(args: argparse.Namespace) -> LaneRevision:
    if args.deployed_revision:
        return LaneRevision(
            revision=normalize_revision(args.deployed_revision),
            compose_project=args.compose_project,
            build_source="supplied",
            state="running",
        )
    lane = read_lane_revision(args.container)
    assert_lane_fence(lane, args.container, args.compose_project)
    return LaneRevision(
        revision=normalize_revision(lane.revision),
        compose_project=lane.compose_project,
        build_source=lane.build_source,
        state=lane.state,
    )


def _run_staleness_mode(args: argparse.Namespace) -> int:
    lane = _read_lane(args)
    divergence = read_divergence(args.repo, args.branch, lane.revision)
    verdict = evaluate(
        lane=lane,
        divergence=divergence,
        now=datetime.now(UTC),
        max_commits_behind=args.max_commits_behind,
        max_age=args.max_age,
    )

    _summary(
        [
            "## Dev-lane staleness (OMN-17888 AC4)",
            "",
            f"- container: `{args.container}` (project `{lane.compose_project}`, "
            f"state `{lane.state}`, build_source `{lane.build_source}`)",
            f"- deployed revision: `{lane.revision}`",
            f"- `{args.branch}` head: `{divergence.head_sha[:12]}`",
            f"- compare status: `{divergence.status}`, "
            f"{divergence.commits_behind} commit(s) behind",
            f"- deployed image commit date: `{divergence.base_committed_at.isoformat()}`",
            f"- bounds: {args.max_commits_behind} commits / {_format_age(args.max_age)}",
            "",
            "RED here means a redeploy was published, delivered, or applied — and one "
            "of those did not happen. Widening the bounds is not a remedy "
            "(OMN-17888 AC5).",
        ]
    )

    if args.positive_control:
        stale = any(f.code == "LANE_STALE" for f in verdict.findings)
        if stale:
            print(
                f"positive control PASSED: revision {lane.revision[:12]} was reported "
                "stale, so the guard fires."
            )
            return 0
        print(
            "::error::positive control FAILED: the guard reported the supplied "
            f"revision {lane.revision[:12]} as fresh. A guard that cannot be shown "
            "to fire proves nothing when it reports green."
        )
        return 1

    return _report(verdict)


class _AncestryResolver:
    """Resolve ancestry at most once per distinct observed revision.

    The lane's label changes only when the lane is recreated, so a poll loop that
    asked GitHub every minute would spend fifty calls proving the same fact.
    Successful resolutions are cached. Transient failures are not: a rate limit,
    502, or token hiccup is recoverable on the next poll and must not poison the
    whole convergence window.
    """

    def __init__(self, repo: str, branch: str, expected: str) -> None:
        self._repo = repo
        self._branch = branch
        self._expected = expected
        self._cache: dict[str, Ancestry] = {}
        self.last_error: str = ""

    def resolve(self, observed: str) -> Ancestry | None:
        if revisions_match(observed, self._expected):
            return None  # never consulted; the exact match decides
        if observed in self._cache:
            return self._cache[observed]
        try:
            resolved = read_ancestry(self._repo, self._branch, self._expected, observed)
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            self.last_error = str(exc)
            print(f"::warning::ancestry of {observed[:12]} unresolved: {exc}")
            return None
        self._cache[observed] = resolved
        return resolved


@dataclass(frozen=True)
class ModelSupersessionProbe:
    """Whether the agent folded THIS command into a newer one (OMN-18143).

    WHY THIS IS READ AT ALL, AND WHY AFTER THE WAIT
    ------------------------------------------------
    The deploy agent now runs the NEWEST foldable rebuild command for a lane
    and records the ones it replaced. Convergence is containment (OMN-18388),
    so a superseded sha's guard sees the lane running a DESCENDANT and reports
    ``ok`` -- correctly, as a statement about the lane. What it cannot say on
    its own is that the sha this run is about never had its own tree built:
    the compose content the lane is running was pinned to the newer commit.

    Rule 24's receipt is a statement about ONE sha, so that distinction has to
    reach it, and this is the field that carries it. It is read AFTER the wait
    because the supersession happens when the agent dequeues, which is
    typically after this guard started watching -- reading it up front would
    ask the question before the answer existed.

    Never raises, and an unreadable answer is never "not superseded": the
    fields carry a reason instead, and the emitting step turns a reason into
    an indeterminate check rather than into silence.
    """

    superseded_by_sha: str | None = None
    superseded_by_correlation_id: str | None = None
    reason: str = ""

    @property
    def superseded(self) -> bool:
        return self.superseded_by_sha is not None

    @property
    def readable(self) -> bool:
        return not self.reason


def read_agent_supersession(
    agent_url: str,
    correlation_id: str,
    *,
    request_timeout_seconds: float = 10.0,
    opener: Callable[[str, float], tuple[int, str]] | None = None,
) -> ModelSupersessionProbe:
    """Ask the agent whether it ran this command or folded it into a newer one.

    A 404 is NOT a reason here, unlike in ``read_agent_acceptance``: a command
    with no job record was never dequeued, so it was certainly not superseded,
    and reporting that as unreadable would put an indeterminate check on every
    receipt whose verify job outran the agent.

    An agent too old to serve the fields answers 200 with them absent, which
    reads as "not superseded" for the same reason -- an agent that cannot
    coalesce has not coalesced.
    """
    if not correlation_id.strip():
        return ModelSupersessionProbe(
            reason=(
                "the publishing job recorded no correlation id, so the agent's "
                "job record for this merge cannot be located"
            )
        )
    if not _CORRELATION_ID_RE.match(correlation_id.strip()):
        return ModelSupersessionProbe(
            reason=f"correlation id {correlation_id!r} is not a uuid"
        )
    if not agent_url.strip():
        return ModelSupersessionProbe(
            reason="no deploy-agent URL was supplied to this guard"
        )

    url = f"{agent_url.rstrip('/')}/job/{correlation_id.strip()}"
    fetch = opener or _http_get_json
    try:
        status, body = fetch(url, request_timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - an unreachable agent is unreadable
        return ModelSupersessionProbe(
            reason=f"{url} could not be read: {type(exc).__name__}: {exc}"
        )
    if status == 404:
        return ModelSupersessionProbe()
    if status != 200:
        return ModelSupersessionProbe(reason=f"{url} answered HTTP {status}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return ModelSupersessionProbe(
            reason=f"{url} returned an unreadable body ({exc})"
        )
    if not isinstance(payload, dict):
        return ModelSupersessionProbe(reason=f"{url} returned a non-object body")
    sha = payload.get("superseded_by_sha")
    if sha is None:
        return ModelSupersessionProbe()
    if not isinstance(sha, str) or not _EXACT_SHA_RE.match(sha):
        return ModelSupersessionProbe(
            reason=(
                f"{url} reported superseded_by_sha={sha!r}, which is not a "
                "40-character lowercase commit sha"
            )
        )
    runner = payload.get("superseded_by_correlation_id")
    return ModelSupersessionProbe(
        superseded_by_sha=sha,
        superseded_by_correlation_id=str(runner) if runner is not None else None,
    )


def supersession_check_outcome(probe: ModelSupersessionProbe) -> str:
    """The receipt verdict for the ``superseded_by_newer_rebuild`` check.

    Three values, and which one a superseded sha gets is the decision this
    whole change turns on.

    ``fail`` when the command WAS folded. The fact is established and
    negative: the agent's own durable record names the commit that ran
    instead, so this is not a check that could not be resolved. A ``PASS``
    here would let rule 24(b) deliver a commit on the strength of a lab pass
    another commit earned -- ``deployed_revision`` is satisfied by CONTAINMENT
    (OMN-18388) and is therefore ``ok`` on exactly these runs, which is what
    makes a second check necessary rather than redundant.

    ``indeterminate`` only when the agent could not be ASKED. That is a
    question about the hop, not about the sha, and OMN-18573 put the
    distinction in the vocabulary precisely so a refusal says which it is.

    ``ok`` when the record exists and names no superseding command, or when
    there is no record at all -- a command the agent never dequeued cannot
    have been folded.
    """
    if probe.superseded:
        return "fail"
    if probe.readable:
        return "ok"
    return "indeterminate"


def supersession_evidence(probe: ModelSupersessionProbe, sha: str) -> str:
    """The receipt check's evidence line. Never empty -- the model refuses that.

    Rendered through the same one-line sanitiser every other evidence string
    here uses: ``GITHUB_OUTPUT`` is line-oriented, and the value is re-read as
    one field of a receipt check and interpolated into a shell variable.
    """
    return " ".join(
        _supersession_evidence(probe, sha).translate(_EVIDENCE_UNSAFE).split()
    )


def _supersession_evidence(probe: ModelSupersessionProbe, sha: str) -> str:
    if probe.superseded:
        runner = probe.superseded_by_correlation_id or "an unnamed correlation"
        return (
            f"the deploy agent did not build {sha}: it folded this rebuild "
            f"command into {probe.superseded_by_sha}, run under {runner}, which "
            "contains this sha. The lane is running that commit's tree, so no "
            "compose content was ever generated from this one"
        )
    if not probe.readable:
        return (
            f"whether {sha} was superseded by a newer rebuild could not be "
            f"established: {probe.reason}"
        )
    return (
        f"the deploy agent's own job record for {sha} names no superseding "
        "command, so this sha's rebuild ran on its own tree"
    )


def reemission_window(
    *,
    previous_revision: str,
    converged_revision: str,
    resolve_contained: Callable[[str, str], tuple[str, ...]],
) -> tuple[str, ...]:
    """The merge shas this convergence may answer for, and no others.

    OMN-18976 part two. A merge queued behind another deploy emits no receipt
    of its own, so only a LATER run can answer for it. This names which later
    run, and for which shas.

    A verify run reads the lane twice: at ``previous_revision`` when it starts
    and at ``converged_revision`` when it converges. Every merge that landed
    strictly between those two was queued behind this one -- its own verify run
    found the lane still at the earlier revision and exited without a receipt.
    That set is the window.

    IT IS NOT "EVERY ANCESTOR OF X", and the distinction is the whole bound.
    The lane has been running for weeks and contains thousands of shas it never
    individually converged on; emitting for those would manufacture receipts
    for merges this run observed nothing about. ``P..X`` is a queue, not a
    history.

    ``converged_revision`` is excluded because its own run is writing its own
    receipt in this same job, and two artifacts of one name from one job is a
    race rather than a re-emission.

    EVERY UNRESOLVED CASE YIELDS NOTHING, which is fail-closed here: emitting
    no receipt leaves the delivery gate refusing those shas, exactly as it does
    today. Emitting one on a guess would not be recoverable.
    """
    if not previous_revision or not converged_revision:
        return ()
    if previous_revision == converged_revision:
        return ()
    contained = resolve_contained(previous_revision, converged_revision)
    window: list[str] = []
    for sha in contained:
        if sha in (converged_revision, previous_revision):
            continue
        if sha in window:
            continue
        window.append(sha)
    return tuple(window)


def _write_output(name: str, value: str) -> None:
    """Publish one single-line value to the calling step's outputs."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _run_convergence_mode(args: argparse.Namespace) -> int:
    """Wait for the lane on the LANE's clock, and report one of three verdicts.

    OMN-18573. The wait used to be a fixed window opened when this step
    started, so the time the redeploy-start effect spent queueing was spent out
    of the budget the lane was meant to get. The budget now starts when the
    deploy agent ACCEPTS the command, read from the agent's own job record, and
    this step's remaining clock is only a BOUND on how long it can watch. When
    the bound is what stopped the wait, or when the acceptance could never be
    established, the verdict is INDETERMINATE and says which -- it is not a
    claim that the lane misbehaved.
    """
    expected = normalize_revision(args.expect_revision)
    resolver = _AncestryResolver(repo=args.repo, branch=args.branch, expected=expected)

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

    def _resolve_queue() -> ModelQueueFacts:
        return read_agent_queue(
            args.agent_url,
            request_timeout_seconds=args.agent_timeout_seconds,
        )

    result = run_convergence_wait(
        expected_revision=expected,
        read_lane=lambda: _read_lane(args),
        resolve_ancestry=resolver.resolve,
        resolve_acceptance=_resolve_acceptance,
        declared_budget=args.wait_timeout,
        wall_clock=timedelta(seconds=args.wall_clock_seconds),
        poll_interval=args.poll_interval,
        clock=lambda: datetime.now(UTC),
        sleep=time.sleep,
        resolve_queue=_resolve_queue,
    )
    lane = result.lane
    ancestry = result.ancestry
    queue = result.queue or ModelQueueFacts.unread(
        "the convergence wait returned no queue facts at all"
    )
    queue_clause = queue.evidence_clause(
        lane_budget_seconds=int(args.wait_timeout.total_seconds()),
        margin_seconds=int(args.poll_interval.total_seconds()),
    )

    evidence = convergence_evidence(
        lane=lane,
        expected_revision=expected,
        ancestry=ancestry,
        waited=result.waited,
        converged=result.outcome is EnumConvergenceOutcome.OK,
        budget=result.budget,
        now=result.finished_at,
        indeterminate_reason=(
            result.reason
            if result.outcome is EnumConvergenceOutcome.INDETERMINATE
            else ""
        ),
        queue_clause=queue_clause,
    )
    # OMN-18388 AC2: the receipt's deployed_revision check carries this, so the
    # artifact names the revision the lane was actually observed at and how it
    # relates to the sha the receipt is keyed by.
    _write_output("evidence", evidence)
    # OMN-18573: the emit step maps this onto the check's outcome. It is an
    # OUTPUT rather than the step's exit status because the status has two
    # values and this has three; the workflow still falls back to the status
    # when the step died before writing anything.
    #
    # OMN-18976: a QUEUED verdict maps to NO check, so the verdict output
    # carries the convergence outcome's own name and `emit_receipt` tells the
    # workflow not to write a receipt at all. Publishing both, rather than
    # making the emit step re-derive one from the other, keeps the decision in
    # the module that made it -- a mapping written in a `run:` block is a
    # mapping nothing tests.
    check_outcome = convergence_check_outcome(result.outcome)
    _write_output(
        "verdict",
        result.outcome.value if check_outcome is None else check_outcome.value,
    )
    _write_output("emit_receipt", "false" if check_outcome is None else "true")

    # OMN-18976 part two. On a convergence, name the merges that were queued
    # BEHIND this one and therefore emitted no receipt of their own, so the
    # emit step can answer for them with the probes it is about to take against
    # the very image that contains them.
    #
    # Only on OK. A run that did not converge has observed nothing it could
    # attest to on anybody else's behalf, and a run that was itself queued is
    # the thing being answered for, not the answerer.
    reemit: tuple[str, ...] = ()
    if result.outcome is EnumConvergenceOutcome.OK:
        reemit = reemission_window(
            previous_revision=result.initial_revision,
            converged_revision=lane.revision,
            resolve_contained=lambda previous, converged: read_contained_commits(
                args.repo, previous, converged
            ),
        )
        if reemit:
            print(
                f"::notice::re-emitting for {len(reemit)} sha(s) queued behind "
                f"this one between {result.initial_revision[:12]} and "
                f"{lane.revision[:12]}"
            )
    # JSON, not a space-separated list, because the consumer is a matrix job:
    # `strategy.matrix` takes `fromJSON(...)` and an EMPTY array skips the job
    # entirely, which is the behaviour wanted when nothing was queued. A
    # dynamic number of artifact uploads cannot be expressed any other way --
    # `uses:` steps cannot be looped.
    _write_output("reemit_shas", json.dumps(list(reemit)))

    # OMN-18143. Read AFTER the wait, because the agent folds a command when it
    # DEQUEUES it, which is normally after this guard started watching. The
    # emit step turns these two outputs into the receipt's
    # `superseded_by_newer_rebuild` check; see ModelSupersessionProbe for why a
    # lane that converged by containment is still not a statement that THIS
    # sha's tree was built. Never fails this step: the supersession is a fact
    # about the receipt, not about the lane.
    supersession = read_agent_supersession(
        agent_url=args.agent_url, correlation_id=args.correlation_id
    )
    _write_output("superseded_by", supersession.superseded_by_sha or "")
    _write_output("superseded_outcome", supersession_check_outcome(supersession))
    _write_output("superseded_evidence", supersession_evidence(supersession, expected))

    # OMN-18436: publish the identity of the container this guard actually read,
    # so the probe step that runs next can prove its HTTP reads came from the
    # SAME generation. The probe runs under `if: always()`, so when this guard
    # fails the lane still answers -- from the PREVIOUS generation -- and the
    # receipt recorded four green reads with nothing in it naming the container
    # that produced them (receipt 4853e0e1: deployed_revision FAIL with
    # ready_effects TRUE, two true statements about two different containers).
    #
    # Written on BOTH outcomes and on a converged-by-containment lane, because
    # the binding question is "which container answered", which has an answer
    # whether or not convergence succeeded. It is omitted only when the identity
    # itself could not be read, and the probe treats an absent record as a
    # failure rather than as permission to skip the check.
    if not args.deployed_revision:
        try:
            generation = read_lane_generation(args.container)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"::warning::lane generation unreadable: {exc}")
        else:
            _write_output("generation", generation.to_json())

    budget = result.budget
    _summary(
        [
            "## Dev-lane convergence (OMN-17888 AC4, OMN-18388, OMN-18573)",
            "",
            f"- verdict: `{result.outcome.value}`",
            f"- expected revision: `{expected}`",
            f"- lane revision: `{lane.revision}`",
            f"- relation: `{ancestry.relation if ancestry else 'unresolved'}`, "
            f"on `{args.branch}`: "
            f"`{ancestry.observed_on_branch if ancestry else 'unknown'}`",
            f"- agent acceptance: "
            f"`{budget.acceptance.accepted_at.isoformat() if budget.acceptance else 'UNESTABLISHED'}`",
            f"- lane budget: {budget.declared_seconds}s from acceptance; this "
            f"job could watch for {budget.wall_clock_seconds}s",
            f"- watched: {_format_age(result.waited)}",
            f"- queue at start: {queue_clause}",
            *([f"- reason: {result.reason}"] if result.reason else []),
            "",
            "Convergence is CONTAINMENT: a lane running a descendant of the merge "
            "sha on this branch has exercised the change. A lane running an "
            "ancestor has not, and is still a failure.",
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
    if result.outcome is EnumConvergenceOutcome.QUEUED:
        # OMN-18976. Not a finding about anything: this merge's turn has not
        # come. No receipt is written, so the gate's absent branch keeps the
        # sha refused and a later convergence can still answer for it, instead
        # of a FAIL artifact asserting forever that the lane misbehaved.
        print(f"::notice::{evidence}")
        return EXIT_QUEUED
    if result.outcome is EnumConvergenceOutcome.INDETERMINATE:
        print(f"::warning::{evidence}")
        return EXIT_INDETERMINATE
    verdict = evaluate_convergence(
        lane=lane,
        expected_revision=expected,
        waited=result.waited,
        wait_timeout=args.wait_timeout,
        ancestry=ancestry,
    )
    return _report(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
