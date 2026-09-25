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
from pathlib import Path

from scripts.ci.r1_front_door_probe import (
    PROBE_LANE,
    PROBE_PROMPT,
    ModelR1ProbeResult,
    build_argv,
    record_result,
    run_probe,
)


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
