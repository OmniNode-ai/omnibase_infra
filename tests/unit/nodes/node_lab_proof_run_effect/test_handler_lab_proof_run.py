# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The run effect's execution rules, and the host programs the plan hands it.

The first half drives HandlerLabProofRun with a fake process runner: abort on a
must-succeed failure, teardown and residue always run, polling, expectations,
pattern counts, output kept out of the report. The second half runs the plan's
own ``python3 -c`` programs for real, because a wrong program would make every
proof lie in the same direction.

Ticket: OMN-19572
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from omnibase_infra.lab_proof.enum_lab_proof_attribution import (
    EnumLabProofAttribution,
)
from omnibase_infra.lab_proof.enum_lab_proof_kind import (
    EnumLabProofKind,
)
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase
from omnibase_infra.lab_proof.model_lab_proof_plan import ModelLabProofPlan
from omnibase_infra.lab_proof.model_lab_proof_retry import ModelLabProofRetry
from omnibase_infra.lab_proof.model_lab_proof_step import ModelLabProofStep
from omnibase_infra.lab_proof.model_lab_proof_subject import ModelLabProofSubject
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_plan import (
    HOST_LOAD_PY,
    PATH_ABSENT_PY,
    PORTS_OPEN_PY,
    REMOVE_TREE_PY,
    SABOTAGE_LINE,
    SABOTAGE_PY,
    TREE_HASH_PY,
)
from omnibase_infra.nodes.node_lab_proof_run_effect.handlers.handler_lab_proof_run import (
    HandlerLabProofRun,
)

pytestmark = pytest.mark.unit

_ID = EnumLabProofStepId
_P = EnumLabProofStepPhase
SHA = "a" * 40


class FakeRunner:
    """Scripted subprocess.run: step argv[0] -> list of (exit, stdout) per attempt."""

    def __init__(self, script: dict[str, list[tuple[int, str]]]) -> None:
        self.script = script
        self.calls: list[str] = []

    def __call__(
        self, argv: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        key = argv[0]
        self.calls.append(key)
        answers = self.script.get(key, [(0, "")])
        code, out = answers.pop(0) if len(answers) > 1 else answers[0]
        return subprocess.CompletedProcess(argv, code, out, "")


def _step(
    key: str,
    step_id: EnumLabProofStepId,
    phase: EnumLabProofStepPhase,
    cwd: Path,
    *,
    must_succeed: bool = True,
    **extra: object,
) -> ModelLabProofStep:
    return ModelLabProofStep.model_validate(
        {
            "step_id": step_id,
            "phase": phase,
            "attribution": EnumLabProofAttribution.HARNESS,
            "purpose": key,
            "argv": (key,),
            "cwd": str(cwd),
            "timeout_seconds": 5,
            "must_succeed": must_succeed,
            **extra,
        }
    )


def _plan(tmp_path: Path, steps: list[ModelLabProofStep]) -> ModelLabProofPlan:
    lane = tmp_path / "prove-test"
    return ModelLabProofPlan(
        run_key="run-test-1",
        host="localhost",
        profile_key="omnibase_core.foundation_override",
        profile_version=1,
        variant_key="foundation",
        proof_kind=EnumLabProofKind.FOUNDATION_OVERRIDE,
        subject=ModelLabProofSubject(
            repo="OmniNode-ai/omnibase_core",
            pr_number=1,
            head_sha=SHA,
            base_sha="b" * 40,
            proved_sha=SHA,
            fetch_ref="refs/pull/1/head",
        ),
        infra_sha="c" * 40,
        workdir=str(lane / "run-test-1"),
        log_dir=str(lane / "logs" / "run-test-1"),
        negative_control=False,
        steps=tuple(steps),
    )


def _handler(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> HandlerLabProofRun:
    ticks = iter(range(0, 10_000, 10))
    return HandlerLabProofRun(
        runner=runner, sleep=lambda _s: None, clock=lambda: float(next(ticks))
    )


def test_a_must_succeed_failure_stops_setup_and_prove_but_never_teardown(
    tmp_path: Path,
) -> None:
    runner = FakeRunner({"build": [(1, "boom")]})
    steps = [
        _step("load", _ID.HOST_LOAD, _P.SETUP, tmp_path),
        _step("build", _ID.BUILD_BASE, _P.SETUP, tmp_path),
        _step("up", _ID.UP, _P.SETUP, tmp_path),
        _step(
            "health", _ID.HEALTH_RUNTIME_MAIN, _P.PROVE, tmp_path, must_succeed=False
        ),
        _step("down", _ID.DOWN, _P.TEARDOWN, tmp_path, must_succeed=False),
        _step(
            "residue", _ID.RESIDUE_CONTAINERS, _P.RESIDUE, tmp_path, must_succeed=False
        ),
    ]
    report = _handler(runner).handle(_plan(tmp_path, steps))
    assert report.aborted_at is _ID.BUILD_BASE
    assert runner.calls == ["load", "build", "down", "residue"]
    up = report.get(_ID.UP)
    assert up is not None and not up.ran and "build_base failed" in up.skip_reason
    down = report.get(_ID.DOWN)
    assert down is not None and down.ran and down.ok


def test_a_failing_step_that_need_not_succeed_does_not_abort(tmp_path: Path) -> None:
    runner = FakeRunner({"tests": [(1, "1 failed")]})
    steps = [
        _step("tests", _ID.FOCUSED_TESTS, _P.SETUP, tmp_path, must_succeed=False),
        _step("up", _ID.UP, _P.SETUP, tmp_path),
    ]
    report = _handler(runner).handle(_plan(tmp_path, steps))
    assert report.aborted_at is None
    assert runner.calls == ["tests", "up"]


def test_polling_retries_until_the_expected_stdout(tmp_path: Path) -> None:
    runner = FakeRunner(
        {"inspect": [(0, "starting\n"), (0, "unhealthy\n"), (0, "healthy\n")]}
    )
    step = _step(
        "inspect",
        _ID.HEALTH_RUNTIME_MAIN,
        _P.PROVE,
        tmp_path,
        must_succeed=False,
        expect_stdout_equals="healthy",
        retry=ModelLabProofRetry(interval_seconds=1, deadline_seconds=900),
    )
    observation = (
        _handler(runner).handle(_plan(tmp_path, [step])).get(_ID.HEALTH_RUNTIME_MAIN)
    )
    assert observation is not None
    assert observation.ok and observation.attempts == 3


def test_polling_gives_up_at_the_deadline(tmp_path: Path) -> None:
    runner = FakeRunner({"inspect": [(0, "unhealthy\n")]})
    step = _step(
        "inspect",
        _ID.HEALTH_RUNTIME_MAIN,
        _P.PROVE,
        tmp_path,
        must_succeed=False,
        expect_stdout_equals="healthy",
        retry=ModelLabProofRetry(interval_seconds=1, deadline_seconds=30),
    )
    observation = (
        _handler(runner).handle(_plan(tmp_path, [step])).get(_ID.HEALTH_RUNTIME_MAIN)
    )
    assert observation is not None
    assert not observation.ok and observation.exit_code == 0
    assert 2 <= observation.attempts <= 4


def test_empty_and_nonempty_expectations(tmp_path: Path) -> None:
    runner = FakeRunner({"ps": [(0, "abc123\n")], "pc": [(0, "")]})
    steps = [
        _step(
            "ps",
            _ID.RESIDUE_CONTAINERS,
            _P.RESIDUE,
            tmp_path,
            must_succeed=False,
            expect_stdout_empty=True,
        ),
        _step(
            "pc",
            _ID.RESIDUE_POSITIVE_CONTROL,
            _P.RESIDUE,
            tmp_path,
            must_succeed=False,
            expect_stdout_nonempty=True,
        ),
    ]
    report = _handler(runner).handle(_plan(tmp_path, steps))
    residue = report.get(_ID.RESIDUE_CONTAINERS)
    control = report.get(_ID.RESIDUE_POSITIVE_CONTROL)
    assert residue is not None and control is not None
    assert not residue.ok and residue.exit_code == 0
    assert not control.ok


def test_patterns_are_counted_and_unrecorded_output_stays_in_the_host_log(
    tmp_path: Path,
) -> None:
    log = "ok\nAuto-wiring failed for HandlerX\npassword=hunter2\nAuto-wiring failed for Y\n"
    runner = FakeRunner({"logs": [(0, log)]})
    step = _step(
        "logs",
        _ID.WIRING_LOGS_RUNTIME_MAIN,
        _P.PROVE,
        tmp_path,
        must_succeed=False,
        grep_patterns=(
            "Auto-wiring failed for",
            "Cannot register duplicate dispatcher ID",
        ),
        record_output=False,
    )
    observation = (
        _handler(runner)
        .handle(_plan(tmp_path, [step]))
        .get(_ID.WIRING_LOGS_RUNTIME_MAIN)
    )
    assert observation is not None
    assert observation.pattern_counts == {
        "Auto-wiring failed for": 2,
        "Cannot register duplicate dispatcher ID": 0,
    }
    assert observation.stdout_tail == ""
    assert "hunter2" in Path(observation.log_path).read_text(encoding="utf-8")


def test_a_timeout_is_recorded_not_raised(tmp_path: Path) -> None:
    def runner(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 5, output="partial")

    step = _step("slow", _ID.BUILD_BASE, _P.SETUP, tmp_path)
    report = _handler(runner).handle(_plan(tmp_path, [step]))
    observation = report.get(_ID.BUILD_BASE)
    assert observation is not None
    assert (
        observation.timed_out and observation.exit_code is None and not observation.ok
    )
    assert report.aborted_at is _ID.BUILD_BASE


def test_a_missing_working_directory_is_a_failed_step(tmp_path: Path) -> None:
    runner = FakeRunner({})
    step = _step("make", _ID.LOCAL_ENV, _P.SETUP, tmp_path / "nope")
    report = _handler(runner).handle(_plan(tmp_path, [step]))
    assert report.aborted_at is _ID.LOCAL_ENV
    assert runner.calls == []


# --- the plan's host programs, run for real ------------------------------------


def _py(program: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", program, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_tree_hash_of_an_installed_package_equals_the_hash_of_its_directory() -> None:
    import pydantic

    directory = str(Path(pydantic.__file__).parent)
    by_path = json.loads(_py(TREE_HASH_PY, "path", directory).stdout)
    by_module = json.loads(_py(TREE_HASH_PY, "module", "pydantic", "pydantic").stdout)
    assert by_path["files"] > 0
    assert by_module["sha256"] == by_path["sha256"]
    assert by_module["version"] == pydantic.VERSION


def test_tree_hash_changes_with_one_byte(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    before = json.loads(_py(TREE_HASH_PY, "path", str(package)).stdout)["sha256"]
    (package / "__init__.py").write_text("x = 2\n", encoding="utf-8")
    after = json.loads(_py(TREE_HASH_PY, "path", str(package)).stdout)["sha256"]
    assert before != after


def test_tree_hash_finds_a_sabotaged_package_without_importing_it(
    tmp_path: Path,
) -> None:
    package = tmp_path / "sabotaged_pkg"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("x = 1\n", encoding="utf-8")
    assert _py(SABOTAGE_PY, str(init), SABOTAGE_LINE).returncode == 0
    assert "raise ImportError" in init.read_text(encoding="utf-8")
    imported = subprocess.run(
        [sys.executable, "-c", "import sabotaged_pkg"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode != 0
    found = subprocess.run(
        [sys.executable, "-c", TREE_HASH_PY, "path", str(package)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode == 0 and json.loads(found.stdout)["files"] == 1


def test_sabotage_of_a_missing_file_fails(tmp_path: Path) -> None:
    assert _py(SABOTAGE_PY, str(tmp_path / "missing.py"), SABOTAGE_LINE).returncode != 0


def test_host_load_ceiling() -> None:
    assert _py(HOST_LOAD_PY, "100000").returncode == 0
    assert _py(HOST_LOAD_PY, "0.0000001").returncode in (0, 3)


def test_ports_program_reports_a_listening_port() -> None:
    import socket

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = str(server.getsockname()[1])
    try:
        assert _py(PORTS_OPEN_PY, port).stdout.strip() == port
    finally:
        server.close()
    assert _py(PORTS_OPEN_PY, port).stdout.strip() == ""


def test_remove_tree_refuses_anything_but_lane_root_slash_run_id(
    tmp_path: Path,
) -> None:
    lane = tmp_path / "prove-x"
    run = lane / "run-1"
    run.mkdir(parents=True)
    outside = tmp_path / "keep"
    outside.mkdir()
    assert _py(REMOVE_TREE_PY, str(outside), str(lane), "keep").returncode != 0
    assert _py(REMOVE_TREE_PY, str(lane), str(lane), "run-1").returncode != 0
    assert outside.exists() and lane.exists()
    assert _py(REMOVE_TREE_PY, str(run), str(lane), "run-1").returncode == 0
    assert not run.exists()
    assert _py(PATH_ABSENT_PY, str(run)).stdout.strip() == ""
    assert _py(PATH_ABSENT_PY, str(lane)).stdout.strip() == "present"
