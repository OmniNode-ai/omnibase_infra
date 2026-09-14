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

FAIL-CLOSED CASES
-----------------

An unreadable manifest, a sibling absent from it, a ``vcs_dirty`` tree (the SHA
does not describe what was built), a non-SHA value, a compose project that is not
the dev lane, a compare the API cannot resolve, and the timeout expiring are all
failures. None of them is evidence the lane is current.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # fixed argv, no shell, trusted docker/gh binaries
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

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
    parser.add_argument("--wait-timeout", type=_parse_duration, default=DEFAULT_WAIT)
    parser.add_argument(
        "--poll-interval", type=_parse_duration, default=DEFAULT_POLL_INTERVAL
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

    deadline = time.monotonic() + args.wait_timeout.total_seconds()
    started = time.monotonic()
    verdict = Verdict()
    lane_revision = ""

    while True:
        waited = timedelta(seconds=time.monotonic() - started)
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
            # A read failure early in the window is usually a lane mid-recreate;
            # it is only a verdict once the window is gone.
            if time.monotonic() >= deadline:
                print(f"::error::{exc}", file=sys.stderr)
                return 1
            print(f"lane read not yet available ({exc}); retrying")
            time.sleep(args.poll_interval.total_seconds())
            continue

        verdict = evaluate_sibling_convergence(
            repo=args.repo,
            lane_revision=lane_revision,
            expected_revision=expected,
            containment=containment,
            waited=waited,
            wait_timeout=args.wait_timeout,
        )
        if verdict.ok or time.monotonic() >= deadline:
            break
        print(
            f"lane vendors {args.repo} at {lane_revision[:12]} ({containment}); "
            f"waited {_format_age(waited)} of {_format_age(args.wait_timeout)}"
        )
        time.sleep(args.poll_interval.total_seconds())

    for note in verdict.notes:
        print(note)
    for finding in verdict.findings:
        print(f"::error::{finding.render()}", file=sys.stderr)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
