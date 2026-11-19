# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev runtime that runs but stays unhealthy reads as down, so it is restored (OMN-25106).

Measured on the lab dev lane, 2026-11-16 to 2026-11-17: ``omninode-runtime``
started at 23:03:13Z, its event bus consumers exited at 23:34:34Z, and from then
on ``/health`` and ``/ready`` did not answer. Docker marked the container
``unhealthy`` (FailingStreak 72 at 00:13Z, a probe every 30 s), but its state
stayed ``running``. The deploy agent's idle restore read the lane through
``read_lane_presence``, which looked at ``State.Status`` alone, so the lane read
``running`` and was never restored. Every bus path that could have redeployed it
(redeploy-start, the orchestrator, the deploy effect) runs inside that runtime.

The restore's rebuild rolls one compose service at a time, and compose recreates
only a service whose configuration or image moved, so a wedged container is also
restarted by the roll step before compose waits on it (``restart_if_wedged``).
Only a container whose state is ``running`` can be wedged.

Control: a running container that is healthy, still starting, or unhealthy for
fewer probes than the wedge bound reads ``running``, so a slow boot or one failed
probe does not restore a lane.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod

pytestmark = pytest.mark.unit

MAIN = "omninode-runtime"
EFFECTS = "omninode-runtime-effects"
REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG = REPO_ROOT / "docker" / "catalog" / "services"
TARGETS = ((MAIN, 8085), (EFFECTS, 8086))
MAIN_ONLY = ((MAIN, 8085), (MAIN, 8086))


def _docker(answers: dict[str, str]) -> Callable[..., Any]:
    """A ``docker inspect`` fake answering with the template's fields for a name."""

    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        name = cmd[-1]
        if name not in answers:
            return subprocess.CompletedProcess(
                cmd, 1, "", f"Error: No such object: {name}"
            )
        return subprocess.CompletedProcess(cmd, 0, f"{answers[name]}\n", "")

    return run


def _presence(
    answers: dict[str, str], targets: tuple[tuple[str, int], ...] = MAIN_ONLY
) -> tuple[str, str]:
    return guard_mod.read_lane_presence(
        targets,
        frozenset(),
        run=_docker(answers),
        unreadable=(OSError, subprocess.TimeoutExpired),
    )


def test_the_recorded_wedge_reads_down() -> None:
    state, detail = _presence({MAIN: "running unhealthy 72"})

    assert state == "down"
    assert MAIN in detail
    assert "unhealthy" in detail
    assert "72" in detail


def test_the_wedge_bound_is_twenty_failed_probes() -> None:
    assert guard_mod.WEDGED_FAILING_STREAK == 20
    bound = guard_mod.WEDGED_FAILING_STREAK

    assert _presence({MAIN: f"running unhealthy {bound}"})[0] == "down"
    assert _presence({MAIN: f"running unhealthy {bound - 1}"})[0] == "running"


@pytest.mark.parametrize(
    ("service", "wedge_window_minutes"),
    [("omninode-runtime", 10), ("runtime-effects", 15)],
)
def test_the_wedge_bound_is_the_window_each_containers_own_probe_interval_gives(
    service: str, wedge_window_minutes: int
) -> None:
    """20 probes is 10 minutes only at a 30 s interval; read each interval."""
    manifest = yaml.safe_load((CATALOG / f"{service}.yaml").read_text())
    interval = manifest["healthcheck"]["interval_s"]

    window = guard_mod.WEDGED_FAILING_STREAK * interval

    assert window == wedge_window_minutes * 60


@pytest.mark.parametrize(
    "answer",
    [
        "running healthy 0",
        "running starting 3",
        "running unhealthy 5",
        "running",
        "running ",
        # A failing streak on a container that is not running is not a wedge.
        "paused unhealthy 25",
        "restarting unhealthy 25",
    ],
)
def test_a_running_container_that_is_not_wedged_reads_running(answer: str) -> None:
    assert _presence({MAIN: answer})[0] == "running"


def test_a_wedged_companion_reads_companion_down() -> None:
    state, detail = _presence(
        {MAIN: "running healthy 0", EFFECTS: "running unhealthy 40"}, TARGETS
    )

    assert state == "companion_down"
    assert EFFECTS in detail


def test_a_down_companion_outranks_a_wedged_one() -> None:
    answers = {MAIN: "running healthy 0", EFFECTS: "exited"}

    assert _presence(answers, TARGETS)[0] == "companion_down"


def test_the_read_asks_docker_for_the_whole_health_template() -> None:
    seen: list[list[str]] = []

    def run(cmd: list[str], timeout: int, **_kw: object) -> Any:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "running healthy 0\n", "")

    guard_mod.read_lane_presence(
        MAIN_ONLY,
        frozenset(),
        run=run,
        unreadable=(OSError, subprocess.TimeoutExpired),
    )

    # The ``if .State.Health`` guard is what lets a container with no
    # healthcheck inspect cleanly; without it docker fails and the lane reads
    # absent.
    assert seen[0] == [
        "docker",
        "inspect",
        "--format",
        "{{.State.Status}}"
        " {{if .State.Health}}{{.State.Health.Status}} "
        "{{.State.Health.FailingStreak}}{{end}}",
        MAIN,
    ]


@pytest.mark.parametrize("stopped", ["created", "exited", "dead"])
def test_a_stopped_container_still_reads_down(stopped: str) -> None:
    assert _presence({MAIN: stopped})[0] == "down"


def _module(execution: object) -> ModuleType:
    """The fake as the executor module the handler's roll functions are handed."""
    return cast("ModuleType", execution)


class _Execution:
    """The executor module's boundary: ``_run`` and what the roll step reads."""

    subprocess = subprocess
    SERVICE_HEALTH_TARGET_INDEX = {"omninode-runtime": 0, "runtime-effects": 1}
    RUNTIME_HEALTH_PROBE_TIMEOUT_SECONDS = 10
    CONTAINER_VERIFY_TIMEOUT_SECONDS = 30

    class EnumRuntimeLane:
        DEV = "dev"

    def __init__(self, answers: dict[str, str], restart_rc: int = 0) -> None:
        self.answers = answers
        self.restart_rc = restart_rc
        self.calls: list[list[str]] = []

    def _run(self, cmd: list[str], timeout: int, **_kw: object) -> Any:
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "restart"]:
            return subprocess.CompletedProcess(cmd, self.restart_rc, cmd[-1], "")
        if cmd[:2] == ["docker", "inspect"]:
            name = cmd[-1]
            if name not in self.answers:
                return subprocess.CompletedProcess(cmd, 1, "", "No such object")
            return subprocess.CompletedProcess(cmd, 0, f"{self.answers[name]}\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _health_target_container(self, lane: object, service: str) -> str:
        return {"omninode-runtime": MAIN, "runtime-effects": EFFECTS}[service]

    def runtime_health_probe_budget(self, lane: object, service: str) -> int:
        return 60

    def _compose_env(self, extra: object) -> dict[str, str]:
        return {}

    def verify_containers_up(self, *_a: object, **_k: object) -> tuple[bool, list[str]]:
        return True, []

    @property
    def restarts(self) -> list[str]:
        return [c[-1] for c in self.calls if c[:2] == ["docker", "restart"]]


def test_a_wedged_container_is_restarted() -> None:
    execution = _Execution({MAIN: "running unhealthy 72"})

    assert guard_mod.restart_if_wedged(MAIN, 60, execution=_module(execution)) is True
    assert execution.restarts == [MAIN]


@pytest.mark.parametrize(
    "answer",
    [
        "running healthy 0",
        "running unhealthy 3",
        "running",
        "paused unhealthy 25",
        "exited",
    ],
)
def test_a_container_that_is_not_wedged_is_not_restarted(answer: str) -> None:
    execution = _Execution({MAIN: answer})

    assert guard_mod.restart_if_wedged(MAIN, 60, execution=_module(execution)) is False
    assert execution.restarts == []


def test_a_missing_container_or_a_refused_restart_is_not_an_error() -> None:
    assert (
        guard_mod.restart_if_wedged(MAIN, 60, execution=_module(_Execution({})))
        is False
    )
    refused = _Execution({MAIN: "running unhealthy 72"}, restart_rc=1)

    assert guard_mod.restart_if_wedged(MAIN, 60, execution=_module(refused)) is False
    assert refused.restarts == [MAIN]


def _roll(execution: _Execution, service: str) -> list[str]:
    argv = ["docker", "compose", "up", "-d", "--no-deps", service]
    guard_mod.run_runtime_roll_step(argv, 60, execution=_module(execution))
    return [" ".join(c[:2]) + " " + c[-1] for c in execution.calls]


def test_the_roll_restarts_a_wedged_service_before_compose_waits_on_it() -> None:
    execution = _Execution({MAIN: "running unhealthy 72"})

    calls = _roll(execution, "omninode-runtime")

    restart = calls.index(f"docker restart {MAIN}")
    compose = next(i for i, c in enumerate(calls) if c.startswith("docker compose"))
    assert restart < compose


def test_the_roll_restarts_the_effects_container_by_its_own_name() -> None:
    execution = _Execution({MAIN: "running healthy 0", EFFECTS: "running unhealthy 40"})

    _roll(execution, "runtime-effects")

    assert execution.restarts == [EFFECTS]


def test_the_roll_leaves_a_healthy_service_alone() -> None:
    execution = _Execution({MAIN: "running healthy 0"})

    calls = _roll(execution, "omninode-runtime")

    assert execution.restarts == []
    assert any(c.startswith("docker compose") for c in calls)


def test_the_roll_does_not_inspect_a_service_with_no_health_target() -> None:
    execution = _Execution({})

    calls = _roll(execution, "onex-api")

    assert not [c for c in calls if c.startswith("docker inspect")]
