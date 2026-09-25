# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19070: every runner that runs the disk-admission hook can pause itself.

Why this exists. ``docker/runners/runner-job-started.sh`` refuses a job when
the runner's disk is under the 5 GB admission floor, counts consecutive
refusals, and at the backoff threshold stops its own container so the listener
stops taking jobs. That last step needs the pause-marker directory bind-mounted
at ``/home/runner/.onex-disk-admission-pause``. Without it the hook logs
``self-pause skipped ... not mounted`` and returns 0, so a starved runner keeps
accepting and failing jobs.

On 2026-09-21 that is what happened to ``omninode-air-runner-1`` on .105: it
refused every ``arm64-verify-proof (host-105)`` job, its counter passed the
threshold, and it never stood down. Measured live on 2026-09-23, the gap was
wider than that one container. The per-host compose files for .105 and .101
never declared the mount, and neither did four service blocks in the primary
host's file (the deploy runner and the three host-201 verify runners), whose
live containers read the same.

The invariant is pinned on the thing that needs the mount, not on a list of
names: any service that mounts the job-started hook must also mount the pause
directory. A new runner service added without it fails here, before a host
ever runs it.

The second half pins why the mount is enough for the required context. A
paused runner's container is stopped, so a job routed to its label waits in the
queue. CI Summary's external sweep reports a queued row as in flight and does
not fail on it. A runner that refuses jobs instead of pausing produces a
completed ``failure`` row, which the sweep fails closed on. That was the
2026-09-21 outage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import evaluate_external_sweep

_DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
_RUNNER_COMPOSE_FILES = sorted(_DOCKER_DIR.glob("docker-compose.runners*.yml"))
_HOOK_TARGET = "/usr/local/bin/runner-job-started.sh"
_PAUSE_TARGET = "/home/runner/.onex-disk-admission-pause"
_HOST_105_LEG = "arm64-verify-proof (host-105)"


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Passthrough constructor for Docker Compose `!override` / `!reset` tags."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that unwraps compose override tags."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)


def _load(path: Path) -> dict[str, Any]:
    # _ComposeLoader extends SafeLoader; the extra constructors only unwrap
    # compose override tags.
    data: dict[str, Any] = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_ComposeLoader,  # noqa: S506
    )
    return data


def _mount_targets(service: dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for entry in service.get("volumes") or []:
        if isinstance(entry, dict):
            targets.add(str(entry.get("target", "")))
            continue
        # short syntax: SOURCE:TARGET[:MODE]. The source may itself contain a
        # `${VAR:-default}` or `${VAR:?message}` expansion, whose colons make a
        # split ambiguous, so match the two known targets instead.
        text = str(entry)
        for target in (_HOOK_TARGET, _PAUSE_TARGET):
            if f":{target}" in text:
                targets.add(target)
    return targets


def _hook_services() -> list[tuple[str, str, set[str]]]:
    rows: list[tuple[str, str, set[str]]] = []
    for path in _RUNNER_COMPOSE_FILES:
        services = _load(path).get("services") or {}
        for name, service in services.items():
            targets = _mount_targets(service or {})
            if _HOOK_TARGET in targets:
                rows.append((path.name, name, targets))
    return rows


@pytest.mark.unit
def test_the_population_is_not_empty() -> None:
    """Rule 16: a zero-missing result over zero services would prove nothing.

    The three runner compose files and the per-host runners are named so a
    rename that drops a file out of the glob fails here instead of passing.
    """
    names = {p.name for p in _RUNNER_COMPOSE_FILES}
    assert {
        "docker-compose.runners.yml",
        "docker-compose.runners-omninode-air-runner.yml",
        "docker-compose.runners-omninode-mini-runner.yml",
        "docker-compose.runners-omnipc2-verify-runner.yml",
    } <= names
    services = {(f, s) for f, s, _ in _hook_services()}
    assert (
        "docker-compose.runners-omninode-air-runner.yml",
        "omninode-air-runner-1",
    ) in services
    assert (
        "docker-compose.runners-omninode-mini-runner.yml",
        "omninode-mini-runner-1",
    ) in services
    assert (
        "docker-compose.runners-omnipc2-verify-runner.yml",
        "omnipc2-verify-runner-1",
    ) in services
    assert ("docker-compose.runners.yml", "omninode-verify-runner-1") in services
    assert len(services) >= 60


@pytest.mark.unit
def test_every_service_running_the_hook_mounts_the_pause_directory() -> None:
    missing = [
        f"{compose}:{service}"
        for compose, service, targets in _hook_services()
        if _PAUSE_TARGET not in targets
    ]
    assert missing == [], (
        "these runner services run the disk-admission hook without the pause "
        f"directory, so their self-pause is inert: {missing}"
    )


def _row(conclusion: str | None, status: str) -> dict[str, Any]:
    return {
        "id": 1905,
        "name": _HOST_105_LEG,
        "status": status,
        "conclusion": conclusion,
        "started_at": "2026-09-21T18:24:00Z",
        "completed_at": "2026-09-21T18:24:02Z" if status == "completed" else None,
        "head_sha": "a" * 40,
    }


def _sweep(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures, in_flight, _swept, _excluded, _provisional = evaluate_external_sweep(
        [row],
        expected=(),
        in_run_names=frozenset(),
        self_name="CI Summary",
        exclusions={},
        events={},
        now=None,
    )
    return failures, in_flight


@pytest.mark.unit
def test_a_paused_runners_leg_is_queued_and_does_not_fail_the_required_context() -> (
    None
):
    failures, in_flight = _sweep(_row(None, "queued"))

    assert failures == []
    assert in_flight == [_HOST_105_LEG]


@pytest.mark.unit
def test_a_refusing_runners_leg_fails_the_required_context_closed() -> None:
    """Positive control: the 2026-09-21 shape, a runner that refuses rather than pauses."""
    failures, in_flight = _sweep(_row("failure", "completed"))

    assert in_flight == []
    assert len(failures) == 1
    assert _HOST_105_LEG in failures[0]
