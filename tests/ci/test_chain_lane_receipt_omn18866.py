# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18866 -- the chain canary emits its own sha-keyed verdict receipt.

The ticket's NEXT STEP says the delegation check comes back as a receipt the
canary writes about itself, which a delivery gate can later READ, rather than a
canary dispatch fired inside the job that gates delivery (the shape reverted in
omnibase_infra#3875). This module pins the emitting half and, just as
deliberately, pins that nothing reads it yet: a receipt that nothing gates on
cannot refuse a delivery, and that is what lets the emitter be proven live
before anything depends on it.

Every claim has both controls. The probe's verdict for a good and a bad canary
receipt is asserted through the CLI that the workflow runs, so a pairing rule
that only held in the helper would not pass here.

The lane's own name is reached through ``EnumLabLane`` and never spelled: the
public-repo hygiene gate refuses new lines that carry lab lane ids, and a
workflow that has to spell one has the CLI hand it the value instead.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from scripts.ci import lab_pass_receipt as lpr
from scripts.ci.lab_pass_receipt import (
    DELEGATION_GOLDEN_CHAIN_CHECK,
    EnumLabLane,
    EnumLabPassResult,
    artifact_name,
    chain_lane_checks,
    main,
    parse_receipt,
    resolve_lane_revision,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANARY = _REPO_ROOT / ".github" / "workflows" / "chain-canary.yml"
_DELIVERY = (
    _REPO_ROOT / ".github" / "workflows" / "deliver-dev-candidate-to-staging.yml"
)
_CHAIN = EnumLabLane.COMPOSE_DEV_CHAIN
_SHA = "0123456789abcdef0123456789abcdef01234567"
_OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"
_STARTED = "2026-10-10T11:00:00Z"
_FINISHED = "2026-10-10T11:02:00Z"
_EMIT_STEP = "Emit the chain canary's own sha-keyed receipt"
_UPLOAD_STEP = "Upload the chain canary's receipt"


def _canary_receipt(
    tmp_path: Path, *, success: bool, verdict: str = "chain_ok"
) -> Path:
    path = tmp_path / "chain-canary-receipt.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "success": success,
                    "verdict": verdict,
                    "detail": "one delegation, one correlation",
                    "links_proven": 4,
                    "links_total": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _emit(
    out: Path,
    canary: Path | None,
    lane: EnumLabLane | None = None,
    github_output: Path | None = None,
) -> list[str]:
    argv = [
        "emit",
        "--sha",
        _SHA,
        "--started-at",
        _STARTED,
        "--finished-at",
        _FINISHED,
        "--out",
        str(out),
    ]
    if lane is not None:
        argv += ["--lane", lane.value]
    if canary is not None:
        argv += ["--chain-canary-receipt", str(canary)]
    if github_output is not None:
        argv += ["--github-output", str(github_output)]
    return argv


# ---------------------------------------------------------------------------
# the pairing of lane and canary receipt, both directions
# ---------------------------------------------------------------------------


def test_a_good_canary_receipt_yields_a_passing_delegation_check(
    tmp_path: Path,
) -> None:
    """POSITIVE CONTROL."""
    checks = chain_lane_checks(_CHAIN, _canary_receipt(tmp_path, success=True))
    assert [c.name for c in checks] == [DELEGATION_GOLDEN_CHAIN_CHECK]
    assert checks[0].ok is True
    assert checks[0].evidence


def test_a_red_canary_receipt_yields_a_failing_check_naming_the_cause(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL -- a delegation with no terminal."""
    checks = chain_lane_checks(
        _CHAIN,
        _canary_receipt(tmp_path, success=False, verdict="terminal_missing"),
    )
    assert checks[0].ok is False
    assert checks[0].indeterminate is False
    assert "terminal_missing" in checks[0].evidence


def test_the_chain_lane_refuses_to_exist_without_a_canary_receipt() -> None:
    """A verdict about a delegation nobody fired."""
    with pytest.raises(ValueError, match="requires --chain-canary-receipt"):
        chain_lane_checks(_CHAIN, None)


@pytest.mark.parametrize(
    "lane",
    [
        EnumLabLane.COMPOSE_DEV,
        EnumLabLane.ONEX_LAB_K3S,
        EnumLabLane.COMPOSE_DEV_CORPUS,
    ],
)
def test_no_other_lane_may_carry_the_delegation_check(
    lane: EnumLabLane, tmp_path: Path
) -> None:
    """The OMN-18872 revert shape: the check on a receipt the release train reads."""
    with pytest.raises(ValueError, match=f"belongs to --lane {_CHAIN.value}"):
        chain_lane_checks(lane, _canary_receipt(tmp_path, success=True))


def test_a_lane_without_a_canary_receipt_gains_no_check() -> None:
    """The flag is inert for every receipt that does not use it."""
    assert chain_lane_checks(EnumLabLane.COMPOSE_DEV, None) == []


# ---------------------------------------------------------------------------
# the emit CLI the workflow runs
# ---------------------------------------------------------------------------


def test_emit_writes_a_passing_chain_receipt_with_no_lane_flag(
    tmp_path: Path,
) -> None:
    """The canary receipt selects the lane, so the workflow never spells it."""
    out = tmp_path / "receipt.json"
    assert main(_emit(out, _canary_receipt(tmp_path, success=True))) == 0
    receipt = parse_receipt(out.read_text(encoding="utf-8"))
    assert receipt.lane is _CHAIN
    assert receipt.sha == _SHA
    assert receipt.result is EnumLabPassResult.PASS
    assert [c.name for c in receipt.checks] == [DELEGATION_GOLDEN_CHAIN_CHECK]


def test_emit_hands_the_upload_step_the_name_the_gate_will_query(
    tmp_path: Path,
) -> None:
    out = tmp_path / "receipt.json"
    github_output = tmp_path / "github_output"
    canary = _canary_receipt(tmp_path, success=True)
    assert main(_emit(out, canary, github_output=github_output)) == 0
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        f"sha={_SHA}",
        f"artifact={artifact_name(_CHAIN, _SHA)}",
    ]


def test_emit_records_a_red_canary_as_a_failed_receipt_not_as_no_receipt(
    tmp_path: Path,
) -> None:
    """The step's exit code is 1, and the file and the outputs are still there."""
    out = tmp_path / "receipt.json"
    github_output = tmp_path / "github_output"
    canary = _canary_receipt(tmp_path, success=False, verdict="terminal_missing")
    assert main(_emit(out, canary, github_output=github_output)) == 1
    receipt = parse_receipt(out.read_text(encoding="utf-8"))
    assert receipt.result is EnumLabPassResult.FAIL
    assert "terminal_missing" in receipt.checks[0].evidence
    assert f"artifact={artifact_name(_CHAIN, _SHA)}" in github_output.read_text(
        encoding="utf-8"
    )


def test_emit_records_a_canary_that_never_reported_as_not_a_pass(
    tmp_path: Path,
) -> None:
    """AC5: an absent terminal receipt leaves the receipt non-PASS."""
    out = tmp_path / "receipt.json"
    assert main(_emit(out, tmp_path / "never-written.json")) == 1
    receipt = parse_receipt(out.read_text(encoding="utf-8"))
    assert receipt.result is not EnumLabPassResult.PASS
    assert receipt.checks[0].indeterminate is True


def test_emit_names_no_outputs_when_nothing_was_written(tmp_path: Path) -> None:
    """The upload step keys on the `artifact` output, so it must stay absent."""
    out = tmp_path / "receipt.json"
    github_output = tmp_path / "github_output"
    assert main(_emit(out, None, lane=_CHAIN, github_output=github_output)) == 1
    assert not out.exists()
    assert not github_output.exists()


def test_emit_without_a_lane_or_a_canary_receipt_writes_nothing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "receipt.json"
    argv = _emit(out, None)
    argv += ["--check", "ready_main:ok:the lane answered"]
    assert main(argv) == 1
    assert not out.exists()


def test_emit_refuses_the_flag_on_the_primary_lab_pass_lane(tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    canary = _canary_receipt(tmp_path, success=True)
    assert main(_emit(out, canary, lane=EnumLabLane.COMPOSE_DEV)) == 1
    assert not out.exists()


def test_an_existing_emit_with_an_explicit_lane_is_unchanged(tmp_path: Path) -> None:
    """POSITIVE CONTROL for every other caller of emit: --lane still works."""
    out = tmp_path / "receipt.json"
    argv = _emit(out, None, lane=EnumLabLane.COMPOSE_DEV)
    argv += ["--check", "ready_main:ok:the lane answered"]
    assert main(argv) == 0
    assert parse_receipt(out.read_text(encoding="utf-8")).lane is (
        EnumLabLane.COMPOSE_DEV
    )


# ---------------------------------------------------------------------------
# keying the receipt on what the lane says it runs
# ---------------------------------------------------------------------------


def test_the_agent_record_is_read_before_the_ready_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lpr, "read_agent_loaded_code_sha", lambda url: _SHA)
    monkeypatch.setattr(lpr, "read_ready_revision", lambda url: _OTHER_SHA)
    assert resolve_lane_revision("http://agent", "http://ready") == _SHA


def test_the_ready_revision_answers_when_the_agent_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lpr, "read_agent_loaded_code_sha", lambda url: "")
    monkeypatch.setattr(lpr, "read_ready_revision", lambda url: _OTHER_SHA)
    assert resolve_lane_revision("http://agent", "http://ready") == _OTHER_SHA


def test_a_lane_that_reports_nothing_cannot_key_a_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(lpr, "read_agent_loaded_code_sha", lambda url: "")
    monkeypatch.setattr(lpr, "read_ready_revision", lambda url: "")
    assert resolve_lane_revision("http://agent", "") == ""
    assert main(["lane-revision", "--agent-url", "http://agent"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "", "stdout is the sha the workflow captures"
    assert "no code sha" in captured.err


def test_the_cli_prints_only_the_sha(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(lpr, "read_agent_loaded_code_sha", lambda url: _SHA)
    assert main(["lane-revision", "--agent-url", "http://agent"]) == 0
    assert capsys.readouterr().out == f"{_SHA}\n"


@contextlib.contextmanager
def _serve(body: bytes) -> Iterator[str]:
    """Answer every GET with ``body`` on a loopback port: the recorded bytes
    reach the real reader over a real socket, with no seam replaced."""

    class Answer(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Answer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.live_contact("tests/ci/fixtures/omn18866_dev_lane_agent_health.json")
def test_the_recorded_dev_lane_agent_answer_keys_the_receipt(
    recorded_response: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The bytes the dev lane's deploy agent really returned, read the way the
    workflow reads them.

    The agent's /health body is the only input `lane-revision` has in
    production. Its live shape (a `loaded_code_sha` among unrelated fields) is
    what the reader must pick out, and the receipt keyed on it must carry the
    same sha the lane reported.
    """
    body = recorded_response["body"]
    assert isinstance(body, dict)
    recorded_sha = str(body["loaded_code_sha"])
    with _serve(json.dumps(body).encode("utf-8")) as agent_url:
        assert main(["lane-revision", "--agent-url", agent_url]) == 0
    assert capsys.readouterr().out == f"{recorded_sha}\n"

    out = tmp_path / "receipt.json"
    github_output = tmp_path / "github_output"
    argv = _emit(
        out, _canary_receipt(tmp_path, success=True), github_output=github_output
    )
    argv[argv.index(_SHA)] = recorded_sha
    assert main(argv) == 0
    receipt = parse_receipt(out.read_text(encoding="utf-8"))
    assert receipt.sha == recorded_sha
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        f"sha={recorded_sha}",
        f"artifact={artifact_name(_CHAIN, recorded_sha)}",
    ]


# ---------------------------------------------------------------------------
# the workflow wiring
# ---------------------------------------------------------------------------


def _steps() -> list[dict[str, object]]:
    workflow = yaml.safe_load(_CANARY.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["chain-canary"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step(prefix: str) -> dict[str, object]:
    matches = [s for s in _steps() if str(s.get("name", "")).startswith(prefix)]
    assert len(matches) == 1, (
        f"expected one step starting {prefix!r}, got {len(matches)}"
    )
    return matches[0]


def test_the_canary_emits_and_uploads_its_receipt_after_the_verdict() -> None:
    names = [str(s.get("name", "")) for s in _steps()]
    emit = names.index(str(_step(_EMIT_STEP)["name"]))
    verdict = names.index("Publish receipt and set the verdict")
    upload = names.index(str(_step(_UPLOAD_STEP)["name"]))
    assert verdict < emit < upload


def test_the_emit_step_grades_the_dispatch_receipt_and_hands_over_the_name() -> None:
    run = str(_step(_EMIT_STEP)["run"])
    assert "lab_pass_receipt.py lane-revision" in run
    assert "--chain-canary-receipt chain-canary-receipt.json" in run
    assert '--github-output "${GITHUB_OUTPUT}"' in run
    assert "--lane" not in run, "the canary receipt selects the lane"
    assert "onex skill chain_canary" not in run, "the dispatch has one owner"


def test_recording_a_receipt_can_never_change_the_runs_conclusion() -> None:
    """The C15 delivery step reads the run's conclusion.

    A step that failed to RECORD would then read as a failed DELEGATION, which
    is the coupling this design exists to avoid. So the step neither exits
    non-zero nor carries `continue-on-error`.
    """
    for prefix in (_EMIT_STEP, _UPLOAD_STEP):
        step = _step(prefix)
        assert "continue-on-error" not in step
        assert "exit 1" not in str(step.get("run", ""))
    emit_run = str(_step(_EMIT_STEP)["run"])
    assert "set -e" not in emit_run
    assert emit_run.rstrip().endswith("exit 0")


def test_only_scheduled_runs_emit_a_receipt() -> None:
    """A manual dispatch can aim the probe at another URL."""
    condition = str(_step(_EMIT_STEP)["if"])
    assert "github.event_name == 'schedule'" in condition
    assert "workflow_dispatch" not in condition


def test_the_upload_uses_the_name_emit_computed_and_only_when_it_exists() -> None:
    upload = _step(_UPLOAD_STEP)
    inputs = upload["with"]
    assert isinstance(inputs, dict)
    assert inputs["name"] == "${{ steps.chain-receipt.outputs.artifact }}"
    assert "steps.chain-receipt.outputs.artifact != ''" in str(upload["if"])
    assert inputs["if-no-files-found"] == "warn", (
        "absence is not fatal: `error` would enrol this upload in the evidence "
        "policy, whose assertion step would fail the run C15 reads"
    )


def test_nothing_gates_on_the_chain_receipt_yet() -> None:
    """EMIT-ONLY. The day a gate reads it, this test is replaced, not skipped.

    Falsifier: the delivery workflow requires the chain lane before the
    emitter has produced a single live receipt, which would stop delivery on a
    surface nobody has observed working.
    """
    delivery = _DELIVERY.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in delivery.splitlines() if not line.lstrip().startswith("#")
    )
    assert _CHAIN.value not in executable
