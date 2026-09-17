#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Assert the .201 compose dev lane's onex-api carries a specific revision (OMN-18572).

WHY A THIRD GUARD
-----------------

``check_dev_lane_staleness.py`` reads ``org.opencontainers.image.revision`` off
``omninode-runtime``. That label is stamped from the **omnibase_infra** commit and
answers the question omnibase_infra's own trigger asks.

``check_lane_sibling_revision.py`` reads ``/app/build-provenance.json`` out of the
runtime image, whose ``per_repo_vcs_provenance.siblings`` lists the repositories
``stage_workspace.sh`` VENDORS into that image -- ``omnibase_core``,
``omnibase_compat``, ``omnimarket``.

``omninode_infra`` is in neither, and not by oversight. ``onex-api`` is not
vendored into the runtime image at all: ``docker/onex-api`` is a self-contained
``python:3.12-slim`` app with no omnibase wheel, built as its OWN image by the
lab-overlay applier (``deploy_agent/lab_overlay.py``, ``API_IMAGE_NAME``) and run
as a separate compose service whose tag comes from ``ONEX_API_IMAGE`` in the
operator env file. Point either existing guard at an omninode_infra merge and it
reports converged having proven nothing about that merge -- the false-green shape
OMN-18200 removed from the compose-dev receipt and OMN-18268 refused to
reintroduce.

WHAT THIS READS
---------------

The ``onex-api`` container's own ``org.opencontainers.image.revision``, stamped by
``lab_overlay.build_and_import`` with the omninode_infra sha the image was built
from, alongside ``ai.omninode.image.source-repo=omninode_infra``. This is the same
label ``scripts/runtime_build/repoint_dev_lane_onex_api.py`` already cross-checks
against the tag's embedded sha8 before it will pin an image, so it is the
structured provenance fact for this image rather than a convention.

Read live 2026-09-17 from the ``onex-api`` container on the dev lane:
``99fdbd375f3b6c17d564b588f505754160b1d1f2``, tag
``onex-lab/omnicloud-core:99fdbd37-20260917T110254Z``.

CONTAINMENT, NOT EQUALITY
-------------------------

The applier builds from the omninode_infra clone's ``origin/dev`` at the moment
the deploy agent reaches it, so a rebuild that starts after a LATER merge
legitimately carries a DESCENDANT of the merge that fired this run. ``identical``
and ``ahead`` both mean the lane carries the change; ``behind`` and ``diverged``
mean it does not. Equality here would red on a lane more current than asked,
which is OMN-18388's defect in a new place.

AN ABSENT LABEL IS UNKNOWN, NEVER UNCHANGED
-------------------------------------------

Images built before OMN-18113 carry no OCI labels at all -- the lane ran one such
image, an eight-day-old hand build, until 2026-09-17T08:46Z. A guard that read a
missing label as "nothing to compare, carry on" would have reported that lane
converged. Every unreadable case here is a finding.

THE FENCE, THE FINDING SHAPE AND THE VERDICT ARE IMPORTED, NOT RETYPED
----------------------------------------------------------------------

``assert_lane_fence``, ``Finding`` and ``Verdict`` come from the sibling guard.
Two copies of a lane fence are two places to disagree about which compose
projects a CI job may read, and this one must never read a governed lane.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CI_DIR.parents[1]))

# A real package import, not an importlib-by-path load, so the imported dataclasses
# keep their types. A dynamic load returns `object` and every downstream access to
# `verdict.ok` becomes an untyped attribute lookup -- which is how a refactor of
# the shared shapes stops being a type error here and starts being a runtime one.
from scripts.ci.check_lane_sibling_revision import (
    CARRIES_STATUSES,
    MISSING_STATUSES,
    Finding,
    Verdict,
    _format_age,
    _parse_duration,
    assert_lane_fence,
)
from scripts.ci.lab_pass_receipt import read_lane_generation

#: The dev lane's onex-api container, and the compose project the fence pins it
#: to. Named rather than derived so this guard can never read a governed lane.
DEFAULT_CONTAINER = "onex-api"
DEFAULT_COMPOSE_PROJECT = "omnibase-infra"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

#: Stamped by ``lab_overlay.build_and_import`` on the api build alone.
REVISION_LABEL = "org.opencontainers.image.revision"
SOURCE_REPO_LABEL = "ai.omninode.image.source-repo"
SOURCE_REPO = "omninode_infra"
SOURCE_REPO_SLUG = "OmniNode-ai/omninode_infra"

#: Matches the bound the two sibling guards use on this lane, for the same
#: reason: node_redeploy_orchestrator and node_redeploy_deploy_effect each
#: declare ``timeout_ms: 660000``, plus margin for the compose recreate.
DEFAULT_WAIT = timedelta(minutes=25)
DEFAULT_POLL_INTERVAL = timedelta(seconds=60)

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def parse_onex_api_revision(labels: object) -> str:
    """Project a container's label map onto the omninode_infra commit it was built from.

    Split out from the docker read so the tests drive it over a label set read
    from the RUNNING container rather than a dict rebuilt by hand.
    """
    if not isinstance(labels, dict):
        raise ValueError(
            f"the onex-api container reports labels as "
            f"{type(labels).__name__}, not an object. `docker inspect` renders "
            "an image built with no labels this way, and an image with no "
            "provenance is UNKNOWN, never unchanged."
        )
    source = str(labels.get(SOURCE_REPO_LABEL, "")).strip()
    if source and source != SOURCE_REPO:
        raise ValueError(
            f"the onex-api container carries {SOURCE_REPO_LABEL}={source!r}, not "
            f"{SOURCE_REPO!r}. Its revision label describes a commit in some "
            "other repository and cannot be compared against this merge."
        )
    revision = str(labels.get(REVISION_LABEL, "")).strip().lower()
    if not revision:
        raise ValueError(
            f"the onex-api container carries no {REVISION_LABEL}. Images built "
            "before OMN-18113 carry no OCI labels at all -- the lane ran one "
            "such hand build for eight days -- so this is UNKNOWN, never "
            "unchanged."
        )
    if not _SHA_RE.match(revision):
        raise ValueError(
            f"onex-api {REVISION_LABEL}={revision!r} is not a git SHA; refusing "
            "to compare an unrecognised identity."
        )
    return revision


def evaluate_onex_api_convergence(
    *,
    lane_revision: str,
    expected_revision: str,
    containment: str,
    waited: timedelta,
    wait_timeout: timedelta,
) -> Verdict:
    """Decide whether the lane's onex-api carries ``expected_revision``."""
    verdict = Verdict()
    status = containment.strip().lower()
    if status in CARRIES_STATUSES:
        verdict.notes.append(
            f"lane runs onex-api at {lane_revision[:12]}, which carries "
            f"{expected_revision[:12]} ({status}), after {_format_age(waited)}"
        )
        return verdict

    if status in MISSING_STATUSES:
        verdict.findings.append(
            Finding(
                "ONEX_API_NOT_CONVERGED",
                f"after {_format_age(waited)} (bound {_format_age(wait_timeout)}) "
                f"the dev lane still runs onex-api at {lane_revision[:12]}, which "
                f"is {status} relative to {expected_revision[:12]} -- the merge "
                "this run asked for. The lane does NOT carry that change. Check, "
                "in order: (1) whether the deploy agent took a job for this "
                "correlation at all, in its state directory; (2) whether the "
                "lab-overlay applier built an image for this sha, in the host's "
                "own tag list; (3) what ONEX_API_IMAGE names in the operator env "
                "file, which is the pin the lane actually resolves and the one "
                "thing an applier build does not move on its own. Do NOT widen "
                "the bound to clear this.",
            )
        )
        return verdict

    verdict.findings.append(
        Finding(
            "CONTAINMENT_UNKNOWN",
            f"the GitHub compare of {expected_revision[:12]}...{lane_revision[:12]} "
            f"in {SOURCE_REPO} returned status {containment!r}, which this guard "
            "does not recognise. An unrecognised status is not a pass; failing "
            "closed.",
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


def read_container_labels(container: str) -> object:
    """Read the whole label map, rather than one templated label at a time.

    ``{{json .Config.Labels}}`` on an image with NO labels renders ``null``
    rather than ``{}``, and ``{{ index .Config.Labels "x" }}`` on that same image
    is a template error with a non-zero exit -- the exact trap that made the
    OMN-18113 repoint refuse a lane it should have advanced. Parsing the whole
    document keeps "no labels" a value this guard can reason about.
    """
    raw = _run(["docker", "inspect", container, "--format", "{{json .Config.Labels}}"])
    return json.loads(raw)


def read_compose_project(container: str) -> str:
    labels = read_container_labels(container)
    if not isinstance(labels, dict):
        raise ValueError(
            f"container {container!r} carries no labels at all, so the compose "
            "project it belongs to cannot be established. The lane fence "
            "refuses rather than assuming."
        )
    return str(labels.get(COMPOSE_PROJECT_LABEL, "")).strip()


def read_containment(expected: str, actual: str) -> str:
    """Ask GitHub whether ``actual`` has ``expected`` in its history.

    The compare API rather than a local ``git merge-base``, for the same reason
    the two sibling guards use it: this job has a depth-1 checkout of ONE
    repository, and omninode_infra is not that repository. A guard answering
    from history it does not have is the false green being removed.
    """
    payload = json.loads(
        _run(["gh", "api", f"repos/{SOURCE_REPO_SLUG}/compare/{expected}...{actual}"])
    )
    return str(payload["status"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-revision",
        required=True,
        help=(
            "The omninode_infra commit the lane's onex-api must carry (the push "
            "that fired this run). No default: a guard that defaults its own "
            "expectation asserts nothing."
        ),
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--compose-project", default=DEFAULT_COMPOSE_PROJECT)
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
            lane_revision = parse_onex_api_revision(
                read_container_labels(args.container)
            )
            containment = read_containment(expected, lane_revision)
        except (RuntimeError, ValueError) as exc:
            # A read failure early in the window is usually a container
            # mid-recreate; it is only a verdict once the window is gone.
            if time.monotonic() >= deadline:
                print(f"::error::{exc}", file=sys.stderr)
                _write_output_evidence(str(exc))
                return 1
            print(f"onex-api read not yet available ({exc}); retrying")
            time.sleep(args.poll_interval.total_seconds())
            continue

        verdict = evaluate_onex_api_convergence(
            lane_revision=lane_revision,
            expected_revision=expected,
            containment=containment,
            waited=waited,
            wait_timeout=args.wait_timeout,
        )
        if verdict.ok or time.monotonic() >= deadline:
            break
        print(
            f"lane runs onex-api at {lane_revision[:12]} ({containment}); "
            f"waited {_format_age(waited)} of {_format_age(args.wait_timeout)}"
        )
        time.sleep(args.poll_interval.total_seconds())

    # The identity of the container this guard read, so the lab-pass probe that
    # runs next can prove its HTTP reads came from the SAME generation. Written
    # on both outcomes, because "which container answered" has an answer either
    # way (OMN-18436).
    _write_output_generation(args.container)
    _write_output_evidence(
        "; ".join(verdict.notes)
        or "; ".join(finding.render() for finding in verdict.findings)
    )

    for note in verdict.notes:
        print(note)
    for finding in verdict.findings:
        print(f"::error::{finding.render()}", file=sys.stderr)
    return 0 if verdict.ok else 1


def _write_output_evidence(evidence: str) -> None:
    """Publish one line naming what was observed, for the receipt's own check.

    ``ModelLabPassCheck`` refuses an empty evidence string, so an emit step with
    nothing to say would write no receipt at all -- the one outcome worse than a
    failing one (OMN-18388).
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path or not evidence:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"evidence={evidence.replace(chr(10), ' ')}\n")


def _write_output_generation(container: str) -> None:
    """Publish one single-line generation record to the calling step's outputs."""
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
