# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pypi-cache service must run an init process as PID 1 (OMN-18846).

`devpi-server` is a plain Python application with no SIGCHLD reaper, and the
service's HEALTHCHECK forks `sh` + `curl` every 30s. Those exec'd processes
re-parent to PID 1, so without an init they are never collected and accumulate
against `pids_limit` until the container cannot fork at all — at which point the
healthcheck is the first thing that stops working.

Measured on 192.168.86.201 2026-09-19T16:23Z: 936 defunct curls under the
container's PID 1, `pids.current` 1023 against a `pids_limit` of 1024, 7687
cgroup fork denials, and a healthcheck failing streak of 3384 consecutive checks
reaching back to 2026-09-18T12:14Z. A restart clears the count and not the cause,
so these two keys are pinned together: `init: true` is only load-bearing while a
finite `pids_limit` makes the leak terminal, and the limit is only survivable
while an init reaps.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.pypi-cache.yml"
)

SERVICE_NAME = "omninode-pypi-cache"


@pytest.mark.unit
def test_pypi_cache_declares_an_init_process() -> None:
    """The service must declare `init: true` so docker-init (tini) reaps orphans."""
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)

    service = compose["services"][SERVICE_NAME]

    assert service.get("init") is True, (
        f"{SERVICE_NAME} must declare `init: true` (OMN-18846). Without it, "
        "devpi-server is PID 1 with no SIGCHLD reaper and the forked healthcheck "
        "children accumulate as zombies until the container hits pids_limit and "
        "can no longer fork the healthcheck itself."
    )


@pytest.mark.unit
def test_pypi_cache_still_bounds_its_pid_count() -> None:
    """The finite `pids_limit` stays: it is the blast-radius bound, not the bug."""
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)

    service = compose["services"][SERVICE_NAME]

    assert isinstance(service.get("pids_limit"), int), (
        f"{SERVICE_NAME} must keep a finite integer `pids_limit` so a future "
        "fork leak is capped at this container instead of the host (OMN-18846). "
        "The 2026-09-18 incident was contained precisely because this limit held."
    )


@pytest.mark.unit
def test_pypi_cache_healthcheck_still_forks_a_shell() -> None:
    """Pin the premise: remove the forking healthcheck and this test says so.

    If the healthcheck ever stops shelling out, the reasoning behind `init: true`
    changes and the comment above it goes stale. Fail here rather than let the two
    drift apart silently.
    """
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)

    test_spec = compose["services"][SERVICE_NAME]["healthcheck"]["test"]

    assert test_spec[0] == "CMD-SHELL", (
        f"{SERVICE_NAME}'s healthcheck no longer shells out, so the OMN-18846 "
        "rationale recorded beside `init: true` needs re-reading before this "
        "assertion is updated."
    )
