# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The single lane-presence guard for container-inspecting integration tests (OMN-18345).

A test that inspects live compose containers answers a question about a lane.
Run where that lane does not exist, it does not answer the question wrongly —
it cannot answer it at all, and a failure there is noise that hides the real
verdict. `Nightly Tests` failed 8/8 nights on exactly this: two `@pytest.mark.slow`
tests asserting against `.201` dev-lane containers inside a job that provisions no
containers at all (``Available: []``).

The guard is deliberately narrow. It answers ONE question — *is this lane here?* —
and never *is this lane well?*. Presence is the precondition; health is the finding.
So:

- docker absent, or NONE of the named containers present  -> skip (the lane is absent)
- at least one named container present                    -> return, and the caller's
                                                             own assertions run and
                                                             FAIL on an unhealthy or
                                                             partially-missing lane

That split is what keeps this a guard rather than a mute button: a lane that is up
and broken is still red. Adding a health condition here would convert every real
finding into a skip, which is the failure mode this module exists to avoid.

Dev-lane runtime liveness itself is observed by `dev-lane-liveness.yml`, so skipping
here removes no observation — it leaves the verdict with the workflow that actually
provisions the lane.

Related:
    - OMN-18322: the 100%-failure scheduled-workflow finding this came from
    - OMN-18345: this guard
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

import pytest

__all__ = ["container_statuses", "docker_path", "require_lane"]

#: `docker ps -a` is used rather than `docker ps` so a stopped or exited
#: container still counts as PRESENT — an exited runtime is a finding, not an
#: absent lane, and must reach the caller's assertions.
_DOCKER_PS: tuple[str, ...] = (
    "docker",
    "ps",
    "-a",
    "--format",
    "{{.Names}}\t{{.Status}}",
)


def docker_path() -> str | None:
    """Absolute path to the `docker` executable, or None when it is not installed.

    A named seam rather than an inline `shutil.which` call so the
    docker-absent branch is testable without monkeypatching the stdlib.
    """
    return shutil.which("docker")


def container_statuses() -> dict[str, str]:
    """Return every local container name mapped to its docker status string.

    Returns an empty mapping when docker is present but the command fails or
    lists nothing. Callers distinguish "no lane" from "lane unwell" through
    :func:`require_lane`, never by reading an empty mapping directly.
    """
    result = subprocess.run(
        list(_DOCKER_PS),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    statuses: dict[str, str] = {}
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            name, status = line.split("\t", 1)
            statuses[name] = status
    return statuses


def require_lane(containers: Sequence[str]) -> dict[str, str]:
    """Skip the calling test when the lane those containers belong to is absent.

    Args:
        containers: the container names the calling test inspects. Presence of
            ANY ONE of them is taken as "the lane is here", so a lane that is up
            with one container missing reaches the caller and fails there.

    Returns:
        The full name -> status mapping, so a caller that already needed it does
        not pay for a second `docker ps`.

    Raises:
        pytest.skip.Exception: via :func:`pytest.skip`, when docker is not
            installed or none of ``containers`` is present.
    """
    wanted = [c for c in containers if c]
    if not wanted:
        raise ValueError(
            "require_lane() needs at least one container name; an empty list "
            "would skip unconditionally, which is the mute button this guard "
            "exists to avoid."
        )

    if docker_path() is None:
        pytest.skip(
            "docker is not installed on this host, so no lane container can be "
            f"inspected (wanted any of: {', '.join(wanted)})"
        )

    statuses = container_statuses()
    present = [c for c in wanted if c in statuses]
    if not present:
        pytest.skip(
            "lane absent: none of the required containers "
            f"({', '.join(wanted)}) exist on this host. "
            f"Present containers: {sorted(statuses) or 'none'}. "
            "Start the lane (infra-up-runtime) to run this test; dev-lane "
            "liveness itself is observed by dev-lane-liveness.yml."
        )
    return statuses
