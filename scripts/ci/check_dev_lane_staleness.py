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
``--expect-revision <sha>`` polls until the lane reports that exact revision, and
fails when it does not within ``--wait-timeout``. This is what the post-merge
job uses, so a delivered-but-not-applied redeploy surfaces on the SAME run that
published it rather than an hour later. The default 25-minute wait is derived
from checked-in contract values, not invented: ``node_redeploy_orchestrator``
and ``node_redeploy_deploy_effect`` each declare ``timeout_ms: 660000`` (11 min),
so the declared chain bound is 22 minutes, plus a 3-minute margin for the bus
hops and the container recreate.

Positive control
----------------
``--positive-control`` (with ``--deployed-revision``) inverts the exit contract:
it exits 0 only if the supplied revision is reported STALE. An empty or green
result from a verification surface is not evidence of health unless the surface
has been shown to fire, so the control is part of the guard rather than a thing
someone is supposed to remember to do by hand.

Fail-closed
-----------
Every indeterminate state is a FAIL: the container missing or not running, an
empty or sentinel revision label, a revision GitHub cannot resolve, a compare
whose status is ``diverged`` or ``behind``, a docker or API error. A staleness
guard that reports fresh when it could not look manufactures the exact false
assurance the ticket is about.

Exit codes: ``0`` the lane tracks ``dev``, ``1`` stale (or unprovable).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess  # fixed argv, no shell, trusted docker/gh binaries
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

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
) -> Verdict:
    """Decide whether the lane converged onto one specific merge SHA."""
    verdict = Verdict()
    if revisions_match(lane.revision, expected_revision):
        verdict.notes.append(
            f"dev lane converged onto {expected_revision[:12]} after "
            f"{_format_age(waited)}"
        )
        return verdict

    verdict.findings.append(
        Finding(
            "NOT_CONVERGED",
            f"after {_format_age(waited)} (bound {_format_age(wait_timeout)}) the dev "
            f"lane still runs {lane.revision[:12]}, not the merge commit "
            f"{expected_revision[:12]} this run published a redeploy-start for. The "
            "command was published; the lane did not apply it. This is the "
            "delivered-but-not-applied shape — check the runtime-effects DLQ "
            "(onex.dlq.omnibase-infra.omnimarket.v1) and the orchestrator consumer "
            "group before assuming the publisher failed.",
        )
    )
    return verdict


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


def _run_convergence_mode(args: argparse.Namespace) -> int:
    expected = normalize_revision(args.expect_revision)
    deadline = time.monotonic() + args.wait_timeout.total_seconds()
    started = time.monotonic()
    lane = _read_lane(args)

    while not revisions_match(lane.revision, expected):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.poll_interval.total_seconds(), remaining))
        lane = _read_lane(args)

    waited = timedelta(seconds=time.monotonic() - started)
    verdict = evaluate_convergence(lane, expected, waited, args.wait_timeout)

    _summary(
        [
            "## Dev-lane convergence (OMN-17888 AC4)",
            "",
            f"- expected revision: `{expected}`",
            f"- lane revision: `{lane.revision}`",
            f"- waited: {_format_age(waited)} (bound {_format_age(args.wait_timeout)})",
        ]
    )
    return _report(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
