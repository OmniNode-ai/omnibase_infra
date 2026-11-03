# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21866: C16's wait-while-busy does not count a k3s lab_overlay settle.

C16 runs 41370897700 and 41383900853 ended RED at the 2400 s deploy wait with
the dev-201 agent reading busy throughout. The fixture is the agent read live on
2026-11-02: between compose jobs it reads ``settling`` / ``lab_overlay`` with no
active job and commands queued behind the settle, and that was the only window
a compose job was not running.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.ci.test_resolve_lab_lane_deploy_wait_omn19811 import (
    IDLE_BUT_COMMAND_QUEUED,
    LANE_A,
    POLL_SECONDS,
    World,
    resolver,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (REPO_ROOT / "tests/fixtures/omn21866/dev201_lab_overlay_settle.json").read_text(
        encoding="utf-8"
    )
)
SETTLING: dict[str, Any] = {
    "health": FIXTURE["settling"]["health"],
    "queue": FIXTURE["settling"]["queue"],
}
NEXT_JOB: dict[str, Any] = {
    "health": FIXTURE["next_job"]["health"],
    "queue": FIXTURE["next_job"]["queue"],
}


def _with_stage(stage: str) -> dict[str, Any]:
    health = dict(SETTLING["health"])
    health["last_result"] = {**health["last_result"], "settling_stage": stage}
    return {"health": health, "queue": SETTLING["queue"]}


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: World,
    *,
    budget: float = 2400,
    flag: str = "true",
) -> tuple[int, dict[str, str]]:
    monkeypatch.setattr(resolver, "http_health", world.health)
    monkeypatch.setattr(resolver, "read_deploy_agent", world.read_agent)
    monkeypatch.setattr(time, "sleep", world.sleep)
    monkeypatch.setattr(time, "monotonic", world.monotonic)
    env_file = tmp_path / "env"
    env_file.write_text("")
    code = resolver.main(
        {
            "LAB_LANES_JSON": json.dumps([LANE_A]),
            "LANE_REQUIRE": "gateway_url",
            "LANE_DEPLOY_WAIT_SECONDS": str(budget),
            "LANE_DEPLOY_POLL_SECONDS": str(POLL_SECONDS),
            "LANE_DEPLOY_WAIT_WHILE_BUSY": flag,
            "GITHUB_ENV": str(env_file),
        }
    )
    written = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line
    )
    return code, written


def test_recorded_lab_overlay_settle_is_not_a_compose_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: the recorded lab_overlay settle proceeds instead of exhausting the wait.

    Falsifier, run at this PR's head with only the resolver taken from the base
    (the test imports nothing this PR adds beyond the fixture, so it collects):
    ``git checkout origin/dev -- .github/actions/resolve-lab-lane/resolve_lab_lane.py``
    then this test alone fails ``assert code == 0`` with ``1 == 0``, the same
    exit 1 as run 41383900853; ``git checkout HEAD -- <that path>`` restores it.
    The other tests in this file name ``compose_busy_reason`` and fail at the
    base by AttributeError, which is not the falsifier.
    """
    world = World(ready_at=0, busy_until=10**6, busy_payload=SETTLING)
    code, written = _run(tmp_path, monkeypatch, world)
    assert code == 0
    assert written["LANE_NAME"] == LANE_A["name"]
    assert "LANE_WAITED_SECONDS" not in written
    assert world.sleeps == []


def test_recorded_settle_still_reads_busy_to_the_shared_test() -> None:
    assert resolver.deploy_agent_busy_reason(SETTLING) == "state=settling"
    assert resolver.compose_busy_reason(SETTLING) is None


def test_recorded_next_job_is_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(ready_at=0, busy_until=10**6, busy_payload=NEXT_JOB)
    code, written = _run(tmp_path, monkeypatch, world, budget=120)
    assert code == 1
    assert "LANE_NAME" not in written
    out = capsys.readouterr().out
    assert "state=deploying" in out
    assert "RED, not a pass" in out


@pytest.mark.parametrize("stage", ["onex_api_pin", "repair_build", "publish"])
def test_other_settle_stages_are_still_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    world = World(ready_at=0, busy_until=10**6, busy_payload=_with_stage(stage))
    code, _ = _run(tmp_path, monkeypatch, world, budget=120)
    assert code == 1
    assert world.sleeps


def test_lab_overlay_settle_beside_an_active_job_is_still_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _with_stage("lab_overlay")
    payload["health"] = {
        **payload["health"],
        "active_job": NEXT_JOB["health"]["active_job"],
    }
    world = World(ready_at=0, busy_until=10**6, busy_payload=payload)
    code, _ = _run(tmp_path, monkeypatch, world, budget=120)
    assert code == 1


def test_idle_agent_with_queued_command_is_still_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(ready_at=0, busy_until=10**6, busy_payload=IDLE_BUT_COMMAND_QUEUED)
    code, _ = _run(tmp_path, monkeypatch, world, budget=120)
    assert code == 1


def test_silent_lane_still_waits_out_a_lab_overlay_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """OMN-20509 holds: a lane that does not answer during a settle is waited for."""
    world = World(ready_at=30, busy_until=10**6, busy_payload=SETTLING)
    code, written = _run(tmp_path, monkeypatch, world, budget=120)
    assert code == 0
    assert 30 <= int(written["LANE_WAITED_SECONDS"]) <= 30 + POLL_SECONDS
    assert "state=settling" in capsys.readouterr().out


NODE_DIR = REPO_ROOT / "src/omnibase_infra/nodes/node_lab_lane_compose_busy_compute"
ACTION_DIR = REPO_ROOT / ".github/actions/resolve-lab-lane"


def test_the_settle_rule_is_node_contract_data_not_a_resolver_constant() -> None:
    from omnibase_infra.models.health.model_lab_lane_compose_busy_policy import (
        ModelLabLaneComposeBusyPolicy,
    )

    contract = yaml.safe_load((NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    declared = contract["config"]["compose_busy_policy"]
    assert declared["non_deploy_settling_stages"] == ["lab_overlay"]
    policy = ModelLabLaneComposeBusyPolicy.from_node_contract()
    assert policy.non_deploy_settling_stages == ("lab_overlay",)


def test_the_resolver_exemption_follows_the_contract_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolver's answer moves with the policy it is handed, not with its source."""
    from omnibase_infra.models.health.model_lab_lane_compose_busy_policy import (
        ModelLabLaneComposeBusyPolicy,
    )

    assert resolver.compose_busy_reason(SETTLING) is None
    monkeypatch.setattr(
        ModelLabLaneComposeBusyPolicy,
        "from_node_contract",
        classmethod(lambda cls: cls(non_deploy_settling_stages=())),
    )
    assert resolver.compose_busy_reason(SETTLING) == "state=settling"
    monkeypatch.setattr(
        ModelLabLaneComposeBusyPolicy,
        "from_node_contract",
        classmethod(lambda cls: cls(non_deploy_settling_stages=("onex_api_pin",))),
    )
    assert resolver.compose_busy_reason(SETTLING) == "state=settling"
    assert resolver.compose_busy_reason(_with_stage("onex_api_pin")) is None


def test_the_handler_is_bound_from_the_node_contract() -> None:
    from omnibase_infra.handlers import handler_lab_lane_compose_busy as module

    contract = yaml.safe_load((NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    (route,) = contract["handler_routing"]["handlers"]
    assert route["handler"]["module"] == module.__name__
    assert route["handler"]["name"] == module.HandlerLabLaneComposeBusy.__name__
    assert not (NODE_DIR / "handlers").exists()


def test_handler_reads_the_declared_stages_from_its_request() -> None:
    from omnibase_infra.handlers.handler_lab_lane_compose_busy import (
        HandlerLabLaneComposeBusy,
    )
    from omnibase_infra.models.health.model_lab_lane_compose_busy_policy import (
        ModelLabLaneComposeBusyPolicy,
    )
    from omnibase_infra.models.health.model_lab_lane_compose_busy_request import (
        ModelLabLaneComposeBusyRequest,
    )

    handler = HandlerLabLaneComposeBusy()
    settle = {
        "state": "settling",
        "settling_stage": "lab_overlay",
        "shared_busy_reason": "state=settling",
    }
    declared = ModelLabLaneComposeBusyRequest(
        **settle,
        policy=ModelLabLaneComposeBusyPolicy(
            non_deploy_settling_stages=("lab_overlay",)
        ),
    )
    undeclared = ModelLabLaneComposeBusyRequest(
        **settle, policy=ModelLabLaneComposeBusyPolicy(non_deploy_settling_stages=())
    )
    assert handler.handle(declared).busy_reason == ""
    assert handler.handle(undeclared).busy_reason == "state=settling"


FLAGS: list[tuple[str, bool]] = [
    ("true", True),
    ("TRUE", True),
    ("TrUe", True),
    ("1", True),
    ("yes", True),
    ("YeS", True),
    ("false", False),
    ("0", False),
    ("no", False),
    ("", False),
    ("anything", False),
    (" true ", False),
    ("TRU\u0130", False),
    ("\u0442rue", False),
]


def _stub(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize("shell", ["/bin/sh", "bash"])
@pytest.mark.parametrize(("flag", "_waits"), FLAGS)
def test_action_step_is_posix_and_always_starts_python3(
    tmp_path: Path, shell: str, flag: str, _waits: bool
) -> None:
    """The shipped step, run by a POSIX sh too: no ``${var,,}``, no flag parse."""
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text(encoding="utf-8"))
    step = action["runs"]["steps"][0]
    assert step["shell"] == "bash"
    for name in ("uv", "python3"):
        _stub(tmp_path / name, f"printf '%s\\n' '{name}' \"$@\"\nexit 37\n")
    action_path = tmp_path / "action directory with spaces"
    result = subprocess.run(
        [shell, "-eu", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_ACTION_PATH": str(action_path),
            "LANE_DEPLOY_WAIT_WHILE_BUSY": flag,
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 37, result.stderr
    assert result.stdout.splitlines() == [
        "python3",
        str(action_path / "resolve_lab_lane.py"),
    ]


@pytest.mark.parametrize(("flag", "waits"), FLAGS)
def test_launcher_and_wait_mode_share_one_flag_parse(flag: str, waits: bool) -> None:
    env = {"LANE_DEPLOY_WAIT_WHILE_BUSY": flag}
    assert resolver.wait_while_busy_requested(env) is waits
    argv = resolver.launcher_argv(env, "/a b/resolve_lab_lane.py")
    expected = ["uv", "run", "--frozen", "python", "/a b/resolve_lab_lane.py"]
    assert argv == (expected if waits else None)
    assert resolver.launcher_argv({**env, resolver.UNDER_UV_MARKER: "1"}, "x") is None


@pytest.mark.parametrize(("flag", "waits"), [("TrUe", True), ("no", False)])
def test_shipped_script_reexecs_under_uv_only_for_wait_mode(
    tmp_path: Path, flag: str, waits: bool
) -> None:
    """End to end: the real file under an isolated python3, then the real package.

    The stub ``uv`` records its argv and the loop marker, then runs the script
    under this test's interpreter, which has the repository environment. The
    run ends on the resolver's own RED for an empty overlay.
    """
    import sys

    record = tmp_path / "uv-argv"
    _stub(
        tmp_path / "uv",
        f'printf \'%s\\n\' "$@" "marker=$LANE_RESOLVER_UNDER_UV" > "{record}"\n'
        f'shift 3\nexec "{sys.executable}" -c '
        '"import omnibase_infra, runpy, sys; '
        "sys.argv = sys.argv[1:]; "
        'runpy.run_path(sys.argv[0], run_name=\'__main__\')" "$@"\n',
    )
    script = ACTION_DIR / "resolve_lab_lane.py"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {resolver.UNDER_UV_MARKER, "LAB_LANES_JSON"}
    }
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        env={
            **env,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "LANE_DEPLOY_WAIT_WHILE_BUSY": flag,
            "LANE_REQUIRE": "gateway_url",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 1, result.stderr
    assert "no lab lane is declared" in result.stdout
    if waits:
        assert record.read_text().splitlines() == [
            "run",
            "--frozen",
            "python",
            str(script),
            "marker=1",
        ]
    else:
        assert not record.exists()


@pytest.mark.parametrize(
    ("flag", "expected_code"),
    [
        ("true", 1),
        ("TRUE", 1),
        ("TrUe", 1),
        ("1", 1),
        ("yes", 1),
        ("YeS", 1),
        ("false", 0),
        ("0", 0),
        ("no", 0),
        ("", 0),
        ("anything", 0),
        (" true ", 0),
    ],
)
def test_python_wait_mode_follows_the_one_flag_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, expected_code: int
) -> None:
    world = World(ready_at=0, busy_until=10**6, busy_payload=NEXT_JOB)
    code, written = _run(tmp_path, monkeypatch, world, budget=1, flag=flag)
    assert code == expected_code
    assert ("LANE_NAME" in written) == (expected_code == 0)
    assert bool(world.sleeps) == (expected_code == 1)


@pytest.mark.parametrize(
    ("health", "reason"),
    [
        (None, None),
        ([], None),
        ({}, None),
        ({"state": "settling"}, "state=settling"),
        ({"state": "settling", "last_result": None}, "state=settling"),
        ({"state": "settling", "last_result": []}, "state=settling"),
        (
            {"state": "settling", "last_result": {"settling_stage": ["lab_overlay"]}},
            "state=settling",
        ),
        (
            {"state": "settling", "last_result": {"settling_stage": None}},
            "state=settling",
        ),
    ],
)
def test_malformed_snapshots_do_not_exempt_an_unknown_settle(
    health: object, reason: str | None
) -> None:
    assert resolver.compose_busy_reason({"health": health, "queue": None}) == reason
