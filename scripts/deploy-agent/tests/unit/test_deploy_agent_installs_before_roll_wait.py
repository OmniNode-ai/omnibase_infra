# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registered dev-lane installs load before the roll's compose wait judges health (OMN-26314).

OMN-26300 moved the in-job apply ahead of the boot guard wait, but the dev roll's
compose ``up --wait`` (``run_runtime_roll_step``) runs inside ``rebuild_scope``, so a
replacement that is unhealthy WITHOUT its registered install raised there and the apply
never ran. Recorded on 2026-11-21 (jobs 7c4427d8 and 1a6f753e): HTTP 503 at the compose
wait with 7 installs declared and no apply line in the journal.

The roll is run here against a fake ``docker`` executable on ``PATH`` that keeps one
container's writable layer on disk, so the test reads what was in the container when
the compose wait was issued and does not depend on how the apply is started.

Each test name carries the criterion it proves: ``apply_before_roll_wait`` (AC1),
``wait_not_weakened`` (AC2), ``dev_lane_only`` (AC3), ``recorded_roll_1a6f753e`` (AC4).
Run each from the repository root::

    uv run --project scripts/deploy-agent --extra dev pytest \\
        scripts/deploy-agent/tests/unit/test_deploy_agent_installs_before_roll_wait.py -k <name> -q
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane

from .test_branch_installs_agent_omn17427 import REAPPLY_ENTRY, REPO_SRC, SERVICE

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "recorded_roll_1a6f753e_omn26314.json"
REPO_ROOT = REPO_SRC.parent
INSTALL_DIR = (
    "/app/.venv/lib/python3.12/site-packages/omnimarket/nodes/node_recorded/handlers"
)
BUDGET = 1800
STATE_ENV = "DEPLOY_AGENT_STATE_DIR"

# The fake docker. One container per compose service, its writable layer a directory.
# A compose up creates the container when absent (a recreate leaves the layer empty).
# A compose up with --wait is healthy only when every required path is in the layer.
FAKE_DOCKER = r"""#!{python}
import json, os, shutil, sys, time
root = os.environ["FAKE_DOCKER_ROOT"]
argv = sys.argv[1:]
def log(kind, **kw):
    with open(os.path.join(root, "log.jsonl"), "a") as out:
        out.write(json.dumps({{"kind": kind, "argv": argv, "t": time.monotonic_ns(), **kw}}) + "\n")
def layer(container):
    return os.path.join(root, "containers", container.removeprefix("cid-"))
def ensure(service):
    os.makedirs(layer(service), exist_ok=True)
def healthy(service):
    if os.environ.get("FAKE_DOCKER_ALWAYS_UNHEALTHY") == "1":
        return False
    required = json.loads(os.environ.get("FAKE_DOCKER_REQUIRED", "[]"))
    return all(os.path.isfile(os.path.join(layer(service), p.lstrip("/"))) for p in required)
cmd = argv[0]
if cmd == "compose":
    service = argv[-1]
    ensure(service)
    if "--wait" in argv:
        budget = argv[argv.index("--wait-timeout") + 1]
        ok = healthy(service)
        log("wait", service=service, budget=budget, healthy=ok)
        if not ok:
            sys.stderr.write(os.environ["FAKE_DOCKER_WAIT_STDERR"] + "\n")
            sys.exit(1)
    else:
        log("up", service=service)
    sys.exit(0)
if cmd == "inspect":
    log("inspect")
    if argv[1] == "-f":
        print("running|healthy")
    else:
        print("running starting 0")
    sys.exit(0)
if cmd == "ps":
    label = [a for a in argv if a.startswith("label=com.docker.compose.service=")][0]
    service = label.split("=")[-1]
    if os.path.isdir(layer(service)):
        print("cid-" + service)
    sys.exit(0)
if cmd == "exec":
    rest = argv[1:]
    if rest[0] == "-u":
        rest = rest[2:]
    container, tool, args = rest[0], rest[1], rest[2:]
    if tool == "sha256sum":
        target = os.path.join(layer(container), args[-1].lstrip("/"))
        if not os.path.isfile(target):
            sys.stderr.write("sha256sum: " + args[-1] + ": No such file or directory\n")
            sys.exit(1)
        import hashlib
        print(hashlib.sha256(open(target, "rb").read()).hexdigest() + "  " + args[-1])
    elif tool == "stat":
        print("0:0 644")
    sys.exit(0)
if cmd == "cp":
    if os.environ.get("FAKE_DOCKER_CP_FAILS") == "1":
        log("cp-refused")
        sys.stderr.write("Error response from daemon: cp refused\n")
        sys.exit(1)
    blob, dest = argv[1], argv[2]
    container, _, path = dest.partition(":")
    target = os.path.join(layer(container), path.lstrip("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(blob, target)
    log("cp", path=path)
    sys.exit(0)
if cmd == "restart":
    log("restart", container=argv[1])
    sys.exit(0)
log("other")
sys.exit(0)
"""


class Roll:
    """One dev roll of the runtime service against the fake docker."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = tmp_path / "docker"
        self.state = tmp_path / "state"
        self.state.mkdir()
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        self.root.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(FAKE_DOCKER.format(python=sys.executable))
        docker.chmod(0o755)
        self.sources = tmp_path / "branch"
        self.sources.mkdir()
        self.paths: list[str] = []
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("FAKE_DOCKER_ROOT", str(self.root))
        monkeypatch.setenv(
            "FAKE_DOCKER_WAIT_STDERR",
            f"Runtime roll stopped at {SERVICE}: Waiting container {SERVICE} is unhealthy",
        )
        monkeypatch.setenv(STATE_ENV, str(self.state))
        monkeypatch.setattr(executor_mod, "deploy_repo_dir", lambda: str(REPO_ROOT))
        monkeypatch.setattr(executor_mod, "_compose_env", lambda extra: {**os.environ})
        monkeypatch.setattr(
            executor_mod, "runtime_health_probe_budget", lambda lane, service: BUDGET
        )
        monkeypatch.setattr(
            executor_mod, "verify_containers_up", lambda *a, **k: (True, [])
        )
        self.monkeypatch = monkeypatch

    def register(
        self, count: int, service: str = SERVICE, restart: bool = True
    ) -> None:
        for n in range(1, count + 1):
            install_id = f"recorded-install-{n}"
            path = f"{INSTALL_DIR}/handler_recorded_{n}.py"
            source = self.sources / f"{install_id}.py"
            source.write_bytes(f"# {install_id}\n".encode())
            argv = [
                sys.executable,
                str(REPO_SRC / REAPPLY_ENTRY),
                "register",
                "--state-dir",
                str(self.state),
                "--lane",
                "dev",
                "--id",
                install_id,
                "--repo",
                "omnimarket",
                "--pr",
                str(6500 + n),
                "--branch",
                f"omn-26152-{install_id}",
                "--head",
                f"{n:040x}",
                "--ticket",
                "OMN-26152",
                "--registered-by",
                "devwindow-lane",
                "--service",
                service,
                "--expires-in-hours",
                "48",
                "--file",
                f"{path}={source}",
            ]
            if not restart:
                argv.append("--no-restart")
            done = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
            )
            assert done.returncode == 0, done.stdout + done.stderr
            self.paths.append(path)
        self.monkeypatch.setenv("FAKE_DOCKER_REQUIRED", json.dumps(self.paths))

    def run(self, lane: EnumRuntimeLane = EnumRuntimeLane.DEV) -> BaseException | None:
        """Run the roll step the way ``_compose_up`` does; the refusal, if any."""
        guard = executor_mod.load_dev_boot_guard()
        argv = ["docker", "compose", "up", "-d", "--no-deps", SERVICE]
        try:
            if lane is EnumRuntimeLane.DEV:
                guard.run_runtime_roll_step(argv, 60, execution=executor_mod)
            else:
                guard.run_compose_up_steps(
                    [argv],
                    60,
                    rolling=True,
                    lane=lane,
                    execution=executor_mod,
                )
        except RuntimeError as exc:
            return exc
        return None

    def log(self) -> list[dict[str, object]]:
        path = self.root / "log.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def kinds(self) -> list[str]:
        return [str(row["kind"]) for row in self.log()]

    def copies_before_first_wait(self) -> int:
        count = 0
        for row in self.log():
            if row["kind"] == "wait":
                return count
            if row["kind"] == "cp":
                count += 1
        return count


@pytest.fixture
def roll(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Roll:
    return Roll(tmp_path, monkeypatch)


def test_apply_before_roll_wait_the_wait_reads_a_runtime_that_carries_the_install(
    roll: Roll,
) -> None:
    """A replacement healthy only with its install passes the wait, because the install is in first."""
    roll.register(1)

    refusal = roll.run()

    assert roll.copies_before_first_wait() == 1, (
        "the registered install was not in the container when the compose wait ran: "
        f"{roll.kinds()}; roll refused with: {refusal}"
    )
    assert refusal is None
    waits = [r for r in roll.log() if r["kind"] == "wait"]
    assert [w["healthy"] for w in waits] == [True]


def test_apply_before_roll_wait_restarts_the_changed_container_once_and_the_reapply_is_idempotent(
    roll: Roll,
) -> None:
    roll.register(3)

    refusal = roll.run()
    after_roll = roll.kinds().count("restart")
    again = roll.run()

    assert roll.copies_before_first_wait() == 3, f"{roll.kinds()}; refused: {refusal}"
    assert after_roll == 1, "three installs in one container restart it once"
    assert again is None
    assert roll.kinds().count("cp") == 3, (
        "a file that already matches is not copied again"
    )
    assert roll.kinds().count("restart") == 1, (
        "and its container is not restarted again"
    )


def test_apply_before_roll_wait_only_the_rolled_services_installs_are_applied(
    roll: Roll,
) -> None:
    roll.register(1, service="omninode-runtime-effects")
    roll.paths.clear()
    roll.monkeypatch.setenv("FAKE_DOCKER_REQUIRED", "[]")

    refusal = roll.run()

    assert refusal is None
    assert "cp" not in roll.kinds(), "an install for another service is not copied in"
    assert not (roll.root / "containers" / "omninode-runtime-effects").exists()


def test_wait_not_weakened_an_unhealthy_runtime_is_refused_with_the_same_wait_and_budget(
    roll: Roll, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install is in, the runtime is still unhealthy: the roll is refused as before."""
    roll.register(2)
    monkeypatch.setenv("FAKE_DOCKER_ALWAYS_UNHEALTHY", "1")

    refusal = roll.run()

    assert roll.copies_before_first_wait() == 2, (
        f"installs were not applied: {roll.kinds()}"
    )
    assert refusal is not None
    assert type(refusal).__name__ == "DevBootFailedError"
    assert f"Runtime roll stopped at {SERVICE}" in str(refusal)
    waits = [r for r in roll.log() if r["kind"] == "wait"]
    assert len(waits) == 1, "the wait is neither skipped nor repeated"
    assert waits[0]["budget"] == str(BUDGET), "the wait keeps the probe budget"
    argv = waits[0]["argv"]
    assert isinstance(argv, list)
    assert argv.index("--wait") < argv.index("--wait-timeout")
    assert roll.kinds()[-1] == "wait", "nothing runs after the refused wait"


def test_wait_not_weakened_no_roll_step_or_install_path_carries_a_skip_flag(
    roll: Roll,
) -> None:
    source = (
        REPO_ROOT / "src/omnibase_infra/handlers/handler_dev_boot_guard.py"
    ).read_text()
    installs = (
        REPO_ROOT
        / "src/omnibase_infra/nodes/node_dev_roll_events_effect/runtime_dev_service_installs.py"
    )

    assert source.count('"--wait"') == 1, "one place issues the compose wait"
    for text in (source, installs.read_text() if installs.exists() else ""):
        assert "SKIP_WAIT" not in text.upper().replace("-", "_")
        assert "NO_WAIT" not in text.upper().replace("-", "_")


@pytest.mark.parametrize("lane", [EnumRuntimeLane.STABILITY_TEST, EnumRuntimeLane.PROD])
def test_dev_lane_only_a_roll_of_another_lane_applies_no_install(
    roll: Roll, lane: EnumRuntimeLane
) -> None:
    roll.register(1)

    roll.run(lane)

    assert "cp" not in roll.kinds()
    assert "restart" not in roll.kinds()


def test_dev_lane_only_the_apply_refuses_a_stability_or_prod_compose_project(
    roll: Roll,
) -> None:
    roll.register(1)
    entry = (
        REPO_SRC
        / "omnibase_infra/nodes/node_dev_roll_events_effect/runtime_dev_service_installs.py"
    )
    for project in ("omnibase-infra-stability", "omnibase-infra-prod"):
        done = subprocess.run(
            [
                sys.executable,
                str(entry),
                "--state-dir",
                str(roll.state),
                "--compose-project",
                project,
                "--service",
                SERVICE,
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
        )
        assert done.returncode == 3, done.stdout + done.stderr
        assert "is not the dev lane" in done.stdout
    assert "cp" not in roll.kinds()


def test_dev_lane_only_a_failed_apply_is_logged_at_error_and_never_a_pass(
    roll: Roll, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    roll.register(1)
    monkeypatch.setenv("FAKE_DOCKER_CP_FAILS", "1")

    with caplog.at_level(logging.ERROR):
        refusal = roll.run()

    assert "cp-refused" in roll.kinds(), "the apply did not run before the wait"
    assert any(
        record.levelno >= logging.ERROR and "branch install" in record.getMessage()
        for record in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert refusal is not None, "an install that could not read back is no pass verdict"
    waits = [r for r in roll.log() if r["kind"] == "wait"]
    assert [w["healthy"] for w in waits] == [False], (
        "the wait judged the runtime as it is"
    )


def test_recorded_roll_1a6f753e_seven_declared_installs_reach_the_wait_in_the_container(
    roll: Roll, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded 01:00 EST roll: 7 installs declared, HTTP 503 at the compose wait."""
    recorded = json.loads(FIXTURE.read_text())
    count = int(recorded["declared_install_count"])
    assert count == 7
    assert recorded["service"] == SERVICE
    assert recorded["roll_failure"].startswith(f"Runtime roll stopped at {SERVICE}")
    assert "HTTP 503" in recorded["roll_failure"]
    monkeypatch.setenv("FAKE_DOCKER_WAIT_STDERR", recorded["compose_wait_stderr"])
    roll.register(count)

    refusal = roll.run()

    assert roll.copies_before_first_wait() == count, (
        f"the recorded roll refused at the wait with {count} installs unapplied "
        f"(apply lines seen: {recorded['branch_install_apply_lines_seen']}): "
        f"{roll.kinds()}; {refusal}"
    )
    assert refusal is None
    assert roll.kinds().count("restart") == 1
    waits = [r for r in roll.log() if r["kind"] == "wait"]
    assert [w["healthy"] for w in waits] == [True]
