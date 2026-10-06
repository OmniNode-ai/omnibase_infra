# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19445 AC2 -- the R1 front-door probe recorder, offline.

The producer (`scripts/ci/r1_front_door_probe.py`) runs
`onex delegate "Reply with exactly the word: ok" --bus kafka --lane dev` on a
lab-host runner, outside the runtime container, and its EXIT CODE is the
verdict (the C11/C12/C15/provider-rung-canary shape already used in this
repo). This module tests the recorder's grading and record shape against a
stubbed subprocess call -- no live lane, no live CLI -- and is reached by the
required pytest job on every PR touching the script.

The positive controls below are the reason a green here means anything: a
grader that cannot be made to fail has not passed, it has not run.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.r1_front_door_probe import (
    PROBE_LANE,
    PROBE_PROMPT,
    ModelR1ProbeResult,
    build_argv,
    classify_failure,
    main,
    record_result,
    run_probe,
)

STDERR_A = (
    "drift_guard: mode=off-registry omnimarket=ABSENT anchor=omnibase-infra@0.38.63 "
    "pin=omnibase-compat expected===0.5.7 installed=0.5.7 pins=3 unsatisfied=0 "
    "verdict=IN_SYNC reason=packaged_pins_satisfied\n"
    "Error: no installed distribution advertises 'task_class_authority' in the "
    "'onex.contracts' entry-point group, so the contract that owns it cannot be read. "
    "Advertised: (none)\n"
)
STDERR_B = (
    "drift_guard: mode=off-registry omnimarket=ABSENT anchor=omnibase-infra@0.38.60 "
    "pin=omnibase-compat expected===0.5.7 installed=0.5.7 pins=3 unsatisfied=0 "
    "verdict=IN_SYNC reason=packaged_pins_satisfied\n"
    "Error: omnimarket is not installed in this environment, so the task-class "
    "contract that declares delegate task classes cannot be resolved. "
    "Pass --task-type explicitly, or repair the co-install.\n"
)
FRONT_DOOR_STDERR = "Error: delegation terminal refused: quality_gate_failed"


class _StubCompleted:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_build_argv_uses_the_corrected_lane_dev_form() -> None:
    """AC1's premise made mechanical: no --kafka-bootstrap literal, ever."""
    argv = build_argv(onex_bin="onex")
    assert argv == [
        "onex",
        "delegate",
        PROBE_PROMPT,
        "--bus",
        "kafka",
        "--lane",
        PROBE_LANE,
    ]
    joined = " ".join(argv)
    assert "--kafka-bootstrap" not in joined
    assert "192.168.86.201" not in joined


def test_run_probe_success_is_graded_ok() -> None:
    def _fake_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        assert argv[0] == "onex"
        return _StubCompleted(0, "status: completed\nresponse: ok\n", "")

    result = run_probe(onex_bin="onex", runner=_fake_run)
    assert isinstance(result, ModelR1ProbeResult)
    assert result.exit_code == 0
    assert result.ok is True
    assert result.lane == PROBE_LANE
    assert "--kafka-bootstrap" not in " ".join(result.argv)


def test_run_probe_nonzero_exit_is_graded_not_ok() -> None:
    def _fake_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        return _StubCompleted(1, "", "KafkaConnectionError: Connection closed")

    result = run_probe(onex_bin="onex", runner=_fake_run)
    assert result.exit_code == 1
    assert result.ok is False
    assert "KafkaConnectionError" in result.stderr_tail


def test_run_probe_timeout_is_graded_not_ok_and_named() -> None:
    def _raising_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        raise TimeoutError(f"probe exceeded {timeout}s")

    result = run_probe(onex_bin="onex", runner=_raising_run)
    assert result.ok is False
    assert result.exit_code != 0
    assert "exceeded" in (result.timeout_reason or "")
    assert result.exit_code == 2
    assert result.leg == "probe-timeout"
    assert result.named_cause == "probe exceeded 90.0s"


def test_record_result_writes_json_and_returns_the_probes_own_exit_code(
    tmp_path: Path,
) -> None:
    result = ModelR1ProbeResult(
        argv=["onex", "delegate", PROBE_PROMPT, "--bus", "kafka", "--lane", PROBE_LANE],
        lane=PROBE_LANE,
        exit_code=0,
        ok=True,
        duration_s=1.23,
        stdout_tail="status: completed",
        stderr_tail="",
        timeout_reason=None,
    )
    out = tmp_path / "r1-front-door-probe.json"
    exit_code = record_result(result, out)
    assert exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["ok"] is True
    assert payload["lane"] == PROBE_LANE
    assert payload["exit_code"] == 0


def test_record_result_propagates_a_failing_exit_code(tmp_path: Path) -> None:
    result = ModelR1ProbeResult(
        argv=["onex", "delegate", PROBE_PROMPT, "--bus", "kafka", "--lane", PROBE_LANE],
        lane=PROBE_LANE,
        exit_code=1,
        ok=False,
        duration_s=0.5,
        stdout_tail="",
        stderr_tail="refused",
        timeout_reason=None,
    )
    out = tmp_path / "r1-front-door-probe.json"
    exit_code = record_result(result, out)
    assert exit_code == 1
    assert json.loads(out.read_text())["ok"] is False


# --- Positive control: prove the grader can actually distinguish pass/fail. ---


def test_positive_control_ok_and_not_ok_records_differ(tmp_path: Path) -> None:
    ok_result = ModelR1ProbeResult(
        argv=["onex"],
        lane=PROBE_LANE,
        exit_code=0,
        ok=True,
        duration_s=0.1,
        stdout_tail="",
        stderr_tail="",
        timeout_reason=None,
    )
    bad_result = ModelR1ProbeResult(
        argv=["onex"],
        lane=PROBE_LANE,
        exit_code=1,
        ok=False,
        duration_s=0.1,
        stdout_tail="",
        stderr_tail="",
        timeout_reason=None,
    )
    assert record_result(ok_result, tmp_path / "a.json") == 0
    assert record_result(bad_result, tmp_path / "b.json") == 1


@pytest.mark.parametrize("stderr", [STDERR_A, STDERR_B])
def test_classify_runner_environment_uses_the_actual_error(stderr: str) -> None:
    """An in-sync drift guard does not prove the task-class owner is installed."""
    leg, cause = classify_failure(stderr=stderr, timeout_reason=None, spawn_error=False)
    assert leg == "runner-env"
    assert cause == stderr.splitlines()[-1].removeprefix("Error: ")
    assert "drift_guard" not in cause


@pytest.mark.parametrize(
    ("stderr", "leg"),
    [
        (
            "the lane declaration /w/omnimarket/config/ci_bus_lanes.yaml does not exist",
            "lane-declaration",
        ),
        (
            "cannot locate the lane declaration: no workspace root is set",
            "lane-declaration",
        ),
        ("--lane 'dev' is not usable: no such lane", "lane-declaration"),
        ("requires non-empty credential fields: sasl_plain_username", "lane-login"),
        ("holds no identity", "lane-login"),
        (
            "Error: canonical $OMNIBASE_PATH/omnimarket clone is on a DETACHED HEAD at 750808959945 -- to dispatch anyway set ONEX_ALLOW_OMNIMARKET_DRIFT=1",
            "runner-env",
        ),
        ("set ONEX_ALLOW_OMNIMARKET_DRIFT=1", "runner-env"),
        ("DelegateLaneCredentialError", "lane-login"),
        ("Half a credential", "lane-login"),
        ("SaslAuthenticationFailed", "lane-login"),
        ("Authentication failed", "lane-login"),
        ("KafkaConnectionError", "broker-route"),
        ("NoBrokersAvailable", "broker-route"),
        ("Unable to bootstrap", "broker-route"),
        ("Connection closed", "broker-route"),
        ("Connection refused", "broker-route"),
        (FRONT_DOOR_STDERR, "front-door"),
        ("task-class contract", "runner-env"),
    ],
)
def test_classify_cli_failure_markers(stderr: str, leg: str) -> None:
    actual_leg, cause = classify_failure(
        stderr=stderr, timeout_reason=None, spawn_error=False
    )
    assert actual_leg == leg
    assert cause == stderr.removeprefix("Error: ")


@pytest.mark.parametrize(
    ("stderr", "timeout_reason", "spawn_error", "leg", "cause"),
    [
        (STDERR_A, "probe exceeded 90s", False, "probe-timeout", "probe exceeded 90s"),
        (
            "onex missing",
            "probe exceeded 90s",
            True,
            "probe-cannot-run",
            "onex missing",
        ),
        (
            "task_class_authority\nAuthentication failed\nConnection refused",
            None,
            False,
            "runner-env",
            "Connection refused",
        ),
        (
            "Authentication failed\nConnection refused",
            None,
            False,
            "lane-login",
            "Connection refused",
        ),
    ],
)
def test_classify_failure_precedence(
    stderr: str,
    timeout_reason: str | None,
    spawn_error: bool,
    leg: str,
    cause: str,
) -> None:
    """Name the first broken dependency even when downstream noise also matches."""
    assert classify_failure(
        stderr=stderr, timeout_reason=timeout_reason, spawn_error=spawn_error
    ) == (leg, cause)


@pytest.mark.parametrize(
    ("stderr", "cause"),
    [
        (
            "Error: first error\nError: final\t error\ncontroller: noise\ntrailing noise\n",
            "final error",
        ),
        (
            "first line\n final\t detail \n\ndrift_guard: noise\nidentity: noise\ncontroller: noise\n",
            "final detail",
        ),
        (
            "\ndrift_guard: noise\nidentity: noise\ncontroller: noise\n",
            "no stderr output",
        ),
        ("", "no stderr output"),
        ("Error: " + "x" * 300, "x" * 240),
    ],
)
def test_classify_cause_selects_one_bounded_line(stderr: str, cause: str) -> None:
    _, actual_cause = classify_failure(
        stderr=stderr, timeout_reason=None, spawn_error=False
    )
    assert actual_cause == cause
    assert "\n" not in actual_cause
    assert len(actual_cause) <= 240


@pytest.mark.parametrize("returncode", [0, 1, 7])
def test_run_probe_names_environment_failure_only_for_nonzero_exit(
    returncode: int,
) -> None:
    def _fake_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        return _StubCompleted(returncode, "", STDERR_A)

    result = run_probe(runner=_fake_run)
    assert result.exit_code == returncode
    if returncode == 0:
        assert result.leg == ""
        assert result.named_cause == ""
    else:
        assert result.leg == "runner-env"
        assert result.named_cause
        assert "\n" not in result.named_cause


@pytest.mark.parametrize(
    ("exception", "leg"),
    [
        (FileNotFoundError("onex binary missing"), "probe-cannot-run"),
        (TimeoutError("probe exceeded 90s"), "probe-timeout"),
        (subprocess.TimeoutExpired("onex", 90), "probe-timeout"),
    ],
)
def test_run_probe_exception_has_a_named_cause(exception: Exception, leg: str) -> None:
    def _raising_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        raise exception

    result = run_probe(runner=_raising_run)
    assert result.exit_code == 2
    assert result.ok is False
    assert result.leg == leg
    assert result.named_cause == str(exception)


def test_positive_control_environment_and_front_door_failures_have_different_legs() -> (
    None
):
    """Two exit-1 outcomes must distinguish a missing co-install from refusal."""

    def _environment_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        return _StubCompleted(1, "", STDERR_A)

    def _refused_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        return _StubCompleted(1, "", FRONT_DOOR_STDERR)

    environment = run_probe(runner=_environment_run)
    refused = run_probe(runner=_refused_run)
    assert environment.leg == "runner-env"
    assert refused.leg == "front-door"
    assert environment.leg != refused.leg


def test_run_probe_classifies_before_truncating_terminal_evidence() -> None:
    """Long diagnostics must not erase the dependency that actually failed."""

    def _fake_run(argv: list[str], *, timeout: float) -> _StubCompleted:
        return _StubCompleted(1, "", STDERR_A + "controller: noise\n" * 200)

    result = run_probe(runner=_fake_run)
    assert len(result.stderr_tail) == 2000
    assert "task_class_authority" not in result.stderr_tail
    assert result.leg == "runner-env"
    assert "task_class_authority" in result.named_cause


@pytest.mark.parametrize("github_actions", [None, "false", "true"])
def test_main_records_names_and_appends_both_verdicts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    github_actions: str | None,
) -> None:
    """Exercise the real subprocess boundary and retain both runs in the summary."""
    if github_actions is None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    else:
        monkeypatch.setenv("GITHUB_ACTIONS", github_actions)
    script = tmp_path / "onex-stub"
    script.write_text(f"#!/bin/sh\nprintf '%s' {shlex.quote(STDERR_A)} >&2\nexit 1\n")
    script.chmod(0o755)
    record = tmp_path / "record.json"
    summary = tmp_path / "summary.md"
    summary.write_text("Existing step evidence\n")
    argv = [
        "--onex-bin",
        str(script),
        "--record",
        str(record),
        "--summary-file",
        str(summary),
    ]

    assert main(argv) == 1
    payload = json.loads(record.read_text())
    assert payload["leg"] == "runner-env"
    assert payload["named_cause"] == STDERR_A.splitlines()[-1].removeprefix("Error: ")
    captured = capsys.readouterr()
    assert "FAIL leg=runner-env exit=1" in captured.err
    assert len(captured.err.splitlines()) == 1
    if github_actions == "true":
        assert captured.out.startswith(
            "::error title=R1 front-door probe (runner-env)::"
        )
    else:
        assert captured.out == ""
    failed_summary = summary.read_text()
    assert failed_summary.startswith("Existing step evidence\n")
    assert "## R1 front-door probe: FAIL" in failed_summary
    assert "runner-env" in failed_summary
    assert "exit code: 1" in failed_summary
    assert STDERR_A in failed_summary
    assert "### stdout_tail\n\n```" in failed_summary
    assert "### stderr_tail\n\n```" in failed_summary
    assert "duration:" in failed_summary
    assert " ".join(build_argv(str(script))) in failed_summary

    script.write_text("#!/bin/sh\nprintf 'ok\\n'\nexit 0\n")
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert "[r1-front-door-probe] OK: exit=0" in captured.out
    assert "::error" not in captured.out
    assert captured.err == ""
    payload = json.loads(record.read_text())
    assert payload["leg"] == ""
    assert payload["named_cause"] == ""
    successful_summary = summary.read_text()
    assert successful_summary.startswith(failed_summary)
    assert "## R1 front-door probe: OK" in successful_summary
    assert "exit code: 0" in successful_summary
    assert "```\nok\n" in successful_summary


# --- The workflow composes the environment the CLI needs (AC2). ---------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "r1-front-door-probe.yml"
_PROBE_JOB = "r1-front-door-probe"


def _steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    steps: list[dict[str, Any]] = workflow["jobs"][_PROBE_JOB]["steps"]
    return steps


def _step(prefix: str) -> dict[str, Any]:
    return next(step for step in _steps() if step["name"].startswith(prefix))


def _index(prefix: str) -> int:
    return next(i for i, step in enumerate(_steps()) if step["name"].startswith(prefix))


def test_workflow_runner_selection_is_unchanged() -> None:
    """The fix never touches the runner: the pool variable stays the only selector."""
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    job = workflow["jobs"][_PROBE_JOB]
    assert job["runs-on"] == "${{ fromJSON(vars.LAB_PROBE_RUNS_ON_JSON) }}"


def test_workflow_checks_out_omnimarket_at_the_sibling_pin_as_a_workspace_root() -> (
    None
):
    pins = yaml.safe_load((_REPO_ROOT / ".github" / "sibling-pins.yaml").read_text())
    step = _step("Checkout omnimarket")
    assert step["with"]["repository"] == "OmniNode-ai/omnimarket"
    assert step["with"]["ref"] == pins["pins"]["omnimarket"]
    assert step["with"]["path"] == ".probe-workspace/omnimarket"
    # The CLI reads <OMNIBASE_PATH>/omnimarket/config/ci_bus_lanes.yaml.
    assert _step("Run the R1 front-door probe")["env"]["OMNIBASE_PATH"] == (
        "${{ github.workspace }}/.probe-workspace"
    )


def test_workflow_co_installs_omnimarket_before_the_probe_runs() -> None:
    assert _index("Checkout omnimarket") < _index("Co-install omnimarket")
    assert _index("Co-install omnimarket") < _index("Run the R1 front-door probe")
    script = _step("Co-install omnimarket")["run"]
    assert '"omnimarket @ git+https://github.com/OmniNode-ai/omnimarket@' in script
    assert "--no-deps" in script
    # a sha checkout is detached, which the drift guard refuses: attach it
    assert "checkout -q -B dev" in script


def test_workflow_probe_step_uses_the_composed_cli_and_publishes_the_terminal() -> None:
    step = _step("Run the R1 front-door probe")
    script = step["run"]
    assert "--onex-bin .venv/bin/onex" in script
    assert '--summary-file "${GITHUB_STEP_SUMMARY}"' in script
    assert 'exit "${status}"' in script
    assert "--kafka-bootstrap" not in script
    # The dev-lane SCRAM identity arrives as environment, never on argv.
    assert step["env"]["KAFKA_SASL_USERNAME"] == "${{ secrets.KAFKA_SASL_USERNAME }}"
    assert step["env"]["KAFKA_SASL_PASSWORD"] == "${{ secrets.KAFKA_SASL_PASSWORD }}"


def test_workflow_record_and_upload_still_run_after_a_failing_probe() -> None:
    assert _step("Assert the probe record")["if"] == "always()"
    assert _step("Upload the probe record")["if"] == "always()"


@pytest.mark.parametrize(
    ("failed_step", "leg"),
    [
        ("checkout_infra", "runner-env"),
        ("install_uv", "runner-env"),
        ("setup_python", "runner-env"),
        ("install_dependencies", "runner-env"),
        ("checkout_provider", "runner-env"),
        ("coinstall_provider", "runner-env"),
        ("lane_transport", "lane-declaration"),
    ],
)
def test_workflow_setup_failure_publishes_a_named_terminal_without_a_venv(
    tmp_path: Path, failed_step: str, leg: str
) -> None:
    steps = _steps()
    setup_ids = [step["id"] for step in steps[: _index("Run the R1 front-door probe")]]
    outcomes = dict.fromkeys(setup_ids, "success")
    failed_index = setup_ids.index(failed_step)
    outcomes[failed_step] = "failure"
    for step_id in setup_ids[failed_index + 1 :]:
        outcomes[step_id] = "skipped"
    probe = _step("Run the R1 front-door probe")
    assert "always()" in probe["if"]
    assert "!cancelled()" in probe["if"]
    for step_id in setup_ids:
        assert f"steps.{step_id}.outcome" in probe["env"]["SETUP_OUTCOMES"]
    summary = tmp_path / "summary.md"
    summary.write_text("Existing setup evidence\n")

    completed = subprocess.run(
        ["bash", "-c", probe["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "SETUP_OUTCOMES": json.dumps(outcomes),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 2
    result = ModelR1ProbeResult.model_validate_json(
        (tmp_path / "r1-front-door-probe.json").read_text()
    )
    assert result.ok is False
    assert result.exit_code == 2
    assert result.leg == leg
    assert result.argv == []  # the CLI was never invoked
    assert (
        result.named_cause
        == f"setup step {failed_step} ended with failure; probe not run"
    )
    assert result.named_cause in completed.stdout
    text = summary.read_text()
    assert text.startswith("Existing setup evidence\n")
    assert "## R1 front-door probe: FAIL" in text
    assert "exit code: 2" in text
    assert result.named_cause in text


@pytest.mark.parametrize("exit_code", [0, 7])
def test_workflow_successful_setup_runs_cli_and_preserves_its_verdict(
    tmp_path: Path, exit_code: int
) -> None:
    setup_ids = [
        step["id"] for step in _steps()[: _index("Run the R1 front-door probe")]
    ]
    python_bin = tmp_path / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > cli-argv.txt\n"
        "printf 'cli record' > r1-front-door-probe.json\n"
        f"exit {exit_code}\n"
    )
    python_bin.chmod(0o755)
    summary = tmp_path / "summary.md"
    summary.write_text("Existing setup evidence\n")
    completed = subprocess.run(
        ["bash", "-c", _step("Run the R1 front-door probe")["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "SETUP_OUTCOMES": json.dumps(dict.fromkeys(setup_ids, "success")),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == exit_code
    assert (tmp_path / "cli-argv.txt").read_text().splitlines() == [
        "scripts/ci/r1_front_door_probe.py",
        "--onex-bin",
        ".venv/bin/onex",
        "--record",
        "r1-front-door-probe.json",
        "--summary-file",
        str(summary),
    ]
    assert (tmp_path / "r1-front-door-probe.json").read_text() == "cli record"
    assert summary.read_text() == "Existing setup evidence\n"
