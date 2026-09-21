# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Controls for the durable lab alarm (OMN-18867).

Each test is a FALSIFIER -- a condition the alarm must fire on -- or a
POSITIVE CONTROL -- one it must stay quiet through. The pairing is the point:
an alarm proven only to fire is one that fires on everything and gets muted,
and a muted alarm is worse than none because it reads as coverage.

The single most important test here is
``test_a_high_but_flat_lag_raises_nothing``. Flat-lag alarming is the noise
that would have got this muted, and muting is how the nine-day freeze
(OMN-18851) would have been missed a second time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    EnumLabPassResult,
    ModelBrokerAccess,
    ModelLabPassCheck,
    ModelLabPassReceipt,
    ReceiptLookupError,
)
from scripts.lab_alarm import (
    EnumAlarmCondition,
    EnumConditionOutcome,
    ModelAlarmConfig,
    ModelAlarmRun,
    ModelAlarmState,
    ModelConditionReport,
    evaluate_consumer_group_lag,
    evaluate_container_restarts,
    evaluate_lab_pass_receipt,
    expand_env,
    make_runner,
    resolve_posting_consent,
    run_once,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHD_DIR = REPO_ROOT / "scripts" / "launchd"
PLIST_TEMPLATE = LAUNCHD_DIR / "com.omninode.lab-alarm.plist.template"
INSTALLER = LAUNCHD_DIR / "install-lab-alarm.sh"
CONFIG = REPO_ROOT / "config" / "lab_alarm.json"

SHA = "a" * 40
CHANNEL = "#onex-lab-alarms"

ACCESS = ModelBrokerAccess(container="broker", brokers="localhost:19092")


def _receipt(result: EnumLabPassResult, *, ok: bool) -> ModelLabPassReceipt:
    return ModelLabPassReceipt(
        sha=SHA,
        lane=EnumLabLane.COMPOSE_DEV,
        started_at="2026-09-21T00:00:00+00:00",
        finished_at="2026-09-21T00:01:00+00:00",
        result=result,
        checks=(
            ModelLabPassCheck(
                name="ready_main",
                ok=ok,
                evidence="HTTP 200" if ok else "HTTP 503",
            ),
        ),
        agent_command_id=None,
    )


def _completed(
    stdout: str, code: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["x"], returncode=code, stdout=stdout, stderr=stderr
    )


def _docker_runner(table: dict[str, subprocess.CompletedProcess[str]]):
    """A runner answering ``docker inspect`` per container from *table*."""

    def run(argv: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        container = argv[-1]
        return table.get(container, _completed("", 1, f"No such object: {container}"))

    return run


def _lag_runner(lags: dict[str, int | str]):
    """A runner answering ``rpk group describe`` per group from *lags*."""

    def run(argv: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        group = argv[argv.index("describe") + 1]
        value = lags.get(group)
        if value is None:
            return _completed("", 1, f"unknown group {group}")
        if isinstance(value, str):
            return _completed(value)
        return _completed(f"GROUP {group}\nTOTAL-LAG {value}\n")

    return run


# ---------------------------------------------------------------------------
# Condition 1 — the lab-pass receipt
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_fail_receipt_raises_one_alarm_naming_the_sha() -> None:
    """AC2 falsifier: the alarm stays silent on a FAIL receipt."""
    report = evaluate_lab_pass_receipt(
        repo="o/r",
        lane=EnumLabLane.COMPOSE_DEV,
        sha=SHA,
        reader=lambda repo, lane, sha: _receipt(EnumLabPassResult.FAIL, ok=False),
    )
    assert report.outcome is EnumConditionOutcome.ALARM
    assert len(report.alarms) == 1
    assert report.alarms[0].subject == SHA
    assert "ready_main" in report.alarms[0].detail


@pytest.mark.unit
def test_a_pass_receipt_raises_nothing() -> None:
    """Positive control: a healthy receipt is not an alarm."""
    report = evaluate_lab_pass_receipt(
        repo="o/r",
        lane=EnumLabLane.COMPOSE_DEV,
        sha=SHA,
        reader=lambda repo, lane, sha: _receipt(EnumLabPassResult.PASS, ok=True),
    )
    assert report.outcome is EnumConditionOutcome.OK
    assert report.alarms == ()


@pytest.mark.unit
def test_an_unreadable_receipt_surface_is_indeterminate_not_ok() -> None:
    """Fail-closed: unread is not passed (Operating Rule 16)."""

    def boom(repo: str, lane: EnumLabLane, sha: str) -> ModelLabPassReceipt:
        raise ReceiptLookupError("the artifact surface returned 502")

    report = evaluate_lab_pass_receipt(
        repo="o/r", lane=EnumLabLane.COMPOSE_DEV, sha=SHA, reader=boom
    )
    assert report.outcome is EnumConditionOutcome.INDETERMINATE
    assert "502" in report.evidence


@pytest.mark.unit
def test_an_unresolved_delivered_sha_is_indeterminate_not_ok() -> None:
    """A lane whose revision could not be read is not a lane that passed."""
    report = evaluate_lab_pass_receipt(
        repo="o/r",
        lane=EnumLabLane.COMPOSE_DEV,
        sha="",
        reader=lambda repo, lane, sha: pytest.fail("must not look anything up"),
    )
    assert report.outcome is EnumConditionOutcome.INDETERMINATE


# ---------------------------------------------------------------------------
# Condition 2 — container restart bounds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_container_past_its_bound_raises_an_alarm_naming_it() -> None:
    """AC3 falsifier: no alarm on a container driven past its declared bound."""
    report = evaluate_container_restarts(
        {"savings-writer": 2, "redpanda": 5},
        runner=_docker_runner(
            {
                "savings-writer": _completed("7 restarting"),
                "redpanda": _completed("0 running"),
            }
        ),
    )
    assert report.outcome is EnumConditionOutcome.ALARM
    assert [alarm.subject for alarm in report.alarms] == ["savings-writer"]
    assert "7" in report.alarms[0].detail


@pytest.mark.unit
def test_a_container_inside_its_bound_raises_nothing() -> None:
    """AC3's other half: an alarm on a container inside its bound is the noise."""
    report = evaluate_container_restarts(
        {"savings-writer": 2},
        runner=_docker_runner({"savings-writer": _completed("2 running")}),
    )
    assert report.outcome is EnumConditionOutcome.OK
    assert "2/2" in report.evidence


@pytest.mark.unit
def test_an_unreadable_container_is_indeterminate_not_ok() -> None:
    """A container docker cannot answer for is not a container inside its bound.

    This is the arm that matters most in practice: the lane runs on a
    different host, so a broken transport makes EVERY container unreadable at
    once. Grading that OK would be a permanently green monitor.
    """
    report = evaluate_container_restarts({"gone": 5}, runner=_docker_runner({}))
    assert report.outcome is EnumConditionOutcome.INDETERMINATE
    assert "UNREADABLE" in report.evidence


@pytest.mark.unit
def test_no_declared_container_is_indeterminate_not_ok() -> None:
    """An empty subject list is not a healthy lane."""
    report = evaluate_container_restarts({}, runner=_docker_runner({}))
    assert report.outcome is EnumConditionOutcome.INDETERMINATE


# ---------------------------------------------------------------------------
# Condition 3 — consumer group lag GROWTH
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_growing_lag_raises_an_alarm_naming_the_group() -> None:
    """AC4 falsifier: growth does not alarm, which is the nine-day freeze."""
    report, sample = evaluate_consumer_group_lag(
        ACCESS,
        ["savings"],
        previous={"savings": 498},
        runner=_lag_runner({"savings": 631}),
    )
    assert report.outcome is EnumConditionOutcome.ALARM
    assert report.alarms[0].subject == "savings"
    assert "498" in report.alarms[0].detail
    assert "631" in report.alarms[0].detail
    assert sample == {"savings": 631}


@pytest.mark.unit
def test_a_high_but_flat_lag_raises_nothing() -> None:
    """AC4's decisive half, and the reason the condition is growth not bound.

    498 is the exact reading the savings writer held for nine days. A bound
    tight enough to flag it flags every healthy busy group as well, and that
    noise is what gets an alarm muted -- which is how the same freeze would
    have been ignored a second time. Falsifier: flat lag alarms.
    """
    report, _ = evaluate_consumer_group_lag(
        ACCESS,
        ["savings"],
        previous={"savings": 498},
        runner=_lag_runner({"savings": 498}),
    )
    assert report.outcome is EnumConditionOutcome.OK
    assert report.alarms == ()


@pytest.mark.unit
def test_shrinking_lag_raises_nothing() -> None:
    """A backlog being worked off is the healthy case, at any magnitude."""
    report, _ = evaluate_consumer_group_lag(
        ACCESS,
        ["savings"],
        previous={"savings": 900},
        runner=_lag_runner({"savings": 12}),
    )
    assert report.outcome is EnumConditionOutcome.OK


@pytest.mark.unit
def test_the_first_sample_is_indeterminate_not_ok() -> None:
    """With one sample, growth is not a question that has been answered."""
    report, sample = evaluate_consumer_group_lag(
        ACCESS, ["savings"], previous=None, runner=_lag_runner({"savings": 498})
    )
    assert report.outcome is EnumConditionOutcome.INDETERMINATE
    assert "baseline" in report.evidence
    # The baseline is still returned, so the NEXT run can compare.
    assert sample == {"savings": 498}


@pytest.mark.unit
def test_an_unreadable_group_is_indeterminate_not_a_group_at_zero() -> None:
    """Operating Rule 16: an empty result is not evidence of absence."""
    report, _ = evaluate_consumer_group_lag(
        ACCESS,
        ["savings", "delegation"],
        previous={"savings": 1, "delegation": 1},
        runner=_lag_runner({"savings": 1}),
    )
    assert report.outcome is EnumConditionOutcome.INDETERMINATE
    assert "UNREADABLE" in report.evidence


@pytest.mark.unit
def test_a_green_lag_reading_names_the_fact_it_read() -> None:
    """A green here must not be quotable as node liveness (OMN-18881).

    A group can read Stable with zero lag while the node behind it refuses
    every message, so the evidence names TOTAL-LAG explicitly rather than
    saying the lane is fine.
    """
    report, _ = evaluate_consumer_group_lag(
        ACCESS, ["savings"], previous={"savings": 5}, runner=_lag_runner({"savings": 5})
    )
    assert "TOTAL-LAG" in report.evidence


# ---------------------------------------------------------------------------
# Edge-triggering, and the whole-run record
# ---------------------------------------------------------------------------


def _run(
    tmp_path: Path,
    *,
    result: EnumLabPassResult,
    restarts: str,
    lag: int,
    ledger: Path,
) -> ModelAlarmRun:
    config = ModelAlarmConfig(
        repo="o/r",
        lane=EnumLabLane.COMPOSE_DEV,
        ready_url="http://lane/ready",
        agent_url="http://lane:8098",
        broker_container="broker",
        broker_address="localhost:19092",
        docker_command=("docker",),
        container_restart_bounds={"savings-writer": 2},
        consumer_groups=("savings",),
    )

    def runner(
        argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if "inspect" in argv:
            return _completed(restarts)
        return _completed(f"TOTAL-LAG {lag}\n")

    return run_once(
        config,
        state_dir=tmp_path,
        ledger_path=ledger,
        sha=SHA,
        receipt_reader=lambda repo, lane, sha: _receipt(
            result, ok=result is EnumLabPassResult.PASS
        ),
        runner=runner,
        posting_channel=CHANNEL,
    )


@pytest.mark.unit
def test_the_same_failure_raises_exactly_one_alarm_across_two_runs(
    tmp_path: Path,
) -> None:
    """AC2: exactly one. Falsifier: it fires repeatedly for one condition.

    Level-triggering would re-raise this every tick, and an alarm that
    re-fires hourly is one nobody reads by the second day.
    """
    ledger = tmp_path / "ledger.md"
    ledger.write_text("| no consent here |\n", encoding="utf-8")

    first = _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=5,
        ledger=ledger,
    )
    second = _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=5,
        ledger=ledger,
    )

    receipt_alarms = [
        a for a in first.raised if a.condition is EnumAlarmCondition.LAB_PASS_RECEIPT
    ]
    assert len(receipt_alarms) == 1
    assert receipt_alarms[0].subject == SHA
    assert [
        a for a in second.raised if a.condition is EnumAlarmCondition.LAB_PASS_RECEIPT
    ] == []


@pytest.mark.unit
def test_a_recovered_subject_can_alarm_again(tmp_path: Path) -> None:
    """Edge-triggering must not become permanent suppression."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text("| none |\n", encoding="utf-8")

    _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=5,
        ledger=ledger,
    )
    _run(
        tmp_path,
        result=EnumLabPassResult.PASS,
        restarts="0 running",
        lag=5,
        ledger=ledger,
    )
    again = _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=5,
        ledger=ledger,
    )

    assert [
        a.subject
        for a in again.raised
        if a.condition is EnumAlarmCondition.LAB_PASS_RECEIPT
    ] == [SHA]


@pytest.mark.unit
def test_an_indeterminate_does_not_clear_an_active_alarm(tmp_path: Path) -> None:
    """ "I could not tell" is not "it recovered".

    Clearing on INDETERMINATE would re-raise the identical alarm on the next
    readable tick, which is the duplicate edge-triggering exists to prevent.
    """
    state = ModelAlarmState(active={"container_restarts:w": "t0"})
    from scripts.lab_alarm import select_new_alarms

    fresh = select_new_alarms(
        [
            ModelConditionReport(
                condition=EnumAlarmCondition.CONTAINER_RESTARTS,
                outcome=EnumConditionOutcome.INDETERMINATE,
                evidence="docker unreachable",
            )
        ],
        state,
        now="t1",
    )
    assert fresh == ()
    assert "container_restarts:w" in state.active


@pytest.mark.unit
def test_a_healthy_run_raises_nothing_and_records_all_three_conditions(
    tmp_path: Path,
) -> None:
    """AC5. Falsifier: a silent run is indistinguishable from one that never ran."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text("| none |\n", encoding="utf-8")

    # First run establishes the lag baseline; the second is the healthy one.
    _run(
        tmp_path,
        result=EnumLabPassResult.PASS,
        restarts="0 running",
        lag=7,
        ledger=ledger,
    )
    run = _run(
        tmp_path,
        result=EnumLabPassResult.PASS,
        restarts="0 running",
        lag=7,
        ledger=ledger,
    )

    assert run.raised == ()
    assert {report.condition for report in run.reports} == set(EnumAlarmCondition)
    assert all(report.evidence.strip() for report in run.reports)

    records = [
        json.loads(line)
        for line in (tmp_path / "alarm-runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 2
    assert sorted(records[-1]["conditions_evaluated"]) == sorted(
        c.value for c in EnumAlarmCondition
    )
    assert records[-1]["raised"] == []


@pytest.mark.unit
def test_a_run_that_skipped_a_condition_cannot_be_constructed() -> None:
    """The record model refuses to describe a partial run as a run."""
    with pytest.raises(ValueError, match="consumer_group_lag"):
        ModelAlarmRun(
            started_at="t0",
            finished_at="t1",
            reports=(
                ModelConditionReport(
                    condition=EnumAlarmCondition.LAB_PASS_RECEIPT,
                    outcome=EnumConditionOutcome.OK,
                    evidence="fine",
                ),
                ModelConditionReport(
                    condition=EnumAlarmCondition.CONTAINER_RESTARTS,
                    outcome=EnumConditionOutcome.OK,
                    evidence="fine",
                ),
            ),
            raised=(),
            posting="disabled",
        )


@pytest.mark.unit
def test_a_condition_reporting_no_evidence_is_refused() -> None:
    """An OK with no evidence is indistinguishable from a condition never run."""
    with pytest.raises(ValueError, match="evidence"):
        ModelConditionReport(
            condition=EnumAlarmCondition.CONTAINER_RESTARTS,
            outcome=EnumConditionOutcome.OK,
            evidence="   ",
        )


# ---------------------------------------------------------------------------
# AC6 — nothing is sent, and the consent gate is a tested behaviour
# ---------------------------------------------------------------------------

_CONSENT = (
    '2026-09-21T00:00Z | OPERATOR-CONSENT | lane=x | "go ahead" | '
    "APPROVED SCOPE: post lab alarms to #onex-lab-alarms | "
    "OUT OF SCOPE: every other channel | durable authorization evidence"
)


@pytest.mark.unit
def test_no_consent_row_means_posting_is_disabled(tmp_path: Path) -> None:
    """AC6 falsifier: a posting path reachable with no resolvable row."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text("| 2026-09-21 | OMN-1 | CLAIM | unrelated |\n", encoding="utf-8")
    assert resolve_posting_consent(ledger, channel=CHANNEL) is None

    run = _run(
        tmp_path,
        result=EnumLabPassResult.PASS,
        restarts="0 running",
        lag=1,
        ledger=ledger,
    )
    assert run.posting.startswith("disabled")


@pytest.mark.unit
def test_a_consent_row_naming_the_channel_resolves_by_file_and_line(
    tmp_path: Path,
) -> None:
    """The positive control, without which the refusal proves nothing."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text(f"| header |\n{_CONSENT}\n", encoding="utf-8")

    consent = resolve_posting_consent(ledger, channel=CHANNEL)
    assert consent is not None
    assert consent.line == 2
    assert consent.citation.endswith(":2")


@pytest.mark.unit
def test_a_consent_row_missing_its_out_of_scope_list_does_not_resolve(
    tmp_path: Path,
) -> None:
    """The OUT OF SCOPE half is what bounds the grant, so it is required.

    A row missing it looks identical to a valid one to the next reader, which
    is exactly why it cannot be allowed to resolve.
    """
    ledger = tmp_path / "ledger.md"
    ledger.write_text(
        '2026-09-21T00:00Z | OPERATOR-CONSENT | lane=x | "go" | '
        "APPROVED SCOPE: post to #onex-lab-alarms | OUT OF SCOPE: |\n",
        encoding="utf-8",
    )
    assert resolve_posting_consent(ledger, channel=CHANNEL) is None


@pytest.mark.unit
def test_a_consent_row_for_another_channel_does_not_resolve(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ledger.write_text(_CONSENT + "\n", encoding="utf-8")
    assert resolve_posting_consent(ledger, channel="#some-other-channel") is None


@pytest.mark.unit
def test_an_unreadable_ledger_does_not_resolve(tmp_path: Path) -> None:
    """A grant that cannot be read is not a grant."""
    assert resolve_posting_consent(tmp_path / "absent.md", channel=CHANNEL) is None


@pytest.mark.unit
def test_the_live_ledger_authorizes_no_posting() -> None:
    """Today's real state, asserted rather than assumed.

    The alarm ships with its posting arm unbuilt and unauthorized. If someone
    later appends a consent row, this test turns red and the sender becomes a
    deliberate decision rather than a silent one.
    """
    omni_home = os.environ.get("OMNI_HOME")
    if not omni_home:
        pytest.skip("OMNI_HOME is unset, so the shared ledger cannot be located")
    ledger = Path(omni_home) / "docs" / "tracking" / "ROLLING_WORK_LEDGER.md"
    if not ledger.is_file():
        pytest.skip("the shared registry ledger is not present on this host")
    assert resolve_posting_consent(ledger, channel=CHANNEL) is None


@pytest.mark.unit
def test_the_alarm_module_imports_no_network_client() -> None:
    """There is no sender, proven structurally rather than by reading it.

    The alarm writes a durable local artifact and sends nothing. This asserts
    the absence mechanically, so adding an HTTP or chat client turns a test
    red instead of quietly shipping a posting path with no consent row behind
    it.
    """
    source = (REPO_ROOT / "scripts" / "lab_alarm.py").read_text(encoding="utf-8")
    for forbidden in ("requests", "httpx", "urllib.request", "slack_sdk", "webhook"):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source


# ---------------------------------------------------------------------------
# The transport seam, the config, and the timer's own declaration (AC1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_runner_rewrites_docker_through_the_configured_transport() -> None:
    """The lane is not on this host, so a bare docker reads nothing.

    Falsifier: docker is invoked locally, which on the real timer host makes
    every container permanently unreadable.
    """
    seen: list[list[str]] = []

    def capture(
        argv: Sequence[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return _completed("0 running")

    from scripts import lab_alarm

    original = lab_alarm._run_read_only
    lab_alarm._run_read_only = capture  # type: ignore[assignment]
    try:
        runner = make_runner(["ssh", "lab-host", "docker"])
        runner(["docker", "inspect", "-f", "{{.RestartCount}}", "c"], timeout=1.0)
    finally:
        lab_alarm._run_read_only = original  # type: ignore[assignment]

    assert seen[0][:3] == ["ssh", "lab-host", "docker"]
    assert seen[0][3] == "inspect"


@pytest.mark.unit
def test_expand_env_raises_on_an_unset_variable(tmp_path: Path) -> None:
    """Operating Rule 8: a silently empty host is a debuggable afternoon."""
    with pytest.raises(ValueError, match="DEFINITELY_UNSET_OMN18867"):
        expand_env("http://${DEFINITELY_UNSET_OMN18867}:8085/ready", source=tmp_path)


@pytest.mark.unit
def test_the_shipped_config_hardcodes_no_machine_address() -> None:
    """Operating Rule 6, asserted on the artifact rather than trusted."""
    raw = CONFIG.read_text(encoding="utf-8")
    assert "192.168." not in raw
    assert "/Users/" not in raw
    payload = json.loads(raw)
    assert "${ONEX_INFRA_HOST}" in payload["ready_url"]
    # The container that crash-looped unobserved for nine days carries the
    # tightest bound, so the alarm is actually armed against its own case.
    bounds = payload["container_restart_bounds"]
    assert (
        bounds["omnimarket-projection-savings-writer"]
        < bounds["omnibase-infra-redpanda"]
    )


@pytest.mark.unit
def test_the_shipped_config_loads_and_declares_every_condition_a_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config that declares no subject would make the alarm permanently quiet."""
    monkeypatch.setenv("ONEX_INFRA_HOST", "lab.invalid")
    monkeypatch.setenv("ONEX_RUNTIME_SSH_HOST", "user@lab.invalid")
    config = ModelAlarmConfig.load(CONFIG)
    assert config.lane is EnumLabLane.COMPOSE_DEV
    assert config.container_restart_bounds
    assert config.consumer_groups
    assert config.docker_command[0] == "ssh"


@pytest.mark.unit
def test_the_launch_agent_is_declared_hourly_and_not_daily() -> None:
    """AC1's mechanically checkable half.

    A ``StartCalendarInterval`` carrying an ``Hour`` key is a DAILY job that
    still reads as periodic, which would leave the detection bound this ticket
    is buying at 24 hours rather than one. The live readback -- a non-zero PID
    column AND a fresh log entry -- cannot be taken from a test and is the
    installer's job; this pins everything that can be.
    """
    plist = PLIST_TEMPLATE.read_text(encoding="utf-8")
    assert "<key>Minute</key>" in plist
    assert "<key>Hour</key>" not in plist
    assert "ai.omninode.lab-alarm" in plist
    # RunAtLoad and KeepAlive both false: installing must not fire a run, and
    # a restarted-on-exit reader would destroy the two-samples-straddling-
    # real-time property the growth condition rests on.
    assert plist.count("<false/>") >= 2
    assert "@BREW_PYTHON@" in plist
    assert "@OMNI_HOME@" in plist
    # Operating Rule 6: the template carries placeholders, never this machine.
    assert "/Users/" not in plist
    assert "192.168." not in plist


@pytest.mark.unit
def test_the_installer_prints_both_halves_of_the_omn17173_readback() -> None:
    """A loaded-but-bootout'd job looks installed and does nothing.

    Neither the label nor the log alone proves the timer fires, so the
    installer must direct the operator at both. Falsifier: an installer that
    prints one and lets a bootout'd job read as installed.
    """
    body = INSTALLER.read_text(encoding="utf-8")
    assert "launchctl list" in body
    assert "launchd.out.log" in body
    assert "bootstrap" in body
    assert "OMN-17173" in body
    assert "/Users/" not in body


@pytest.mark.unit
def test_the_alarm_runs_on_the_brew_interpreter_with_no_virtualenv() -> None:
    """Operating Rule 11: launchd has a restricted PATH and no login shell.

    The agent names a literal brew interpreter, and the alarm is stdlib-only
    plus the stdlib-only lab-pass receipt module, so no virtual environment
    has to stay healthy for the timer to keep working.
    """
    installer = INSTALLER.read_text(encoding="utf-8")
    assert "/opt/homebrew/bin/python3.13" in installer
    assert "/usr/local/bin/python3.13" in installer
    source = (REPO_ROOT / "scripts" / "lab_alarm.py").read_text(encoding="utf-8")
    assert "import yaml" not in source
    assert "import pydantic" not in source


@pytest.mark.unit
def test_the_alarm_is_importable_by_a_bare_interpreter() -> None:
    """The timer runs the file by path, not as an installed console script.

    Falsifier: it imports only because pytest put the repo root on the path,
    and the launch agent's direct invocation dies at import time every hour
    with nothing but a traceback in a log nobody reads.
    """
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "lab_alarm.py"),
            "--help",
        ],
        capture_output=True,
        check=False,
        cwd=str(Path(__file__).parent),
        env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert "--state-dir" in completed.stdout.decode()


# ---------------------------------------------------------------------------
# The exit code carries the verdict too
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_exit_code_carries_the_verdict(tmp_path: Path) -> None:
    """A monitor that prints a finding and exits 0 is the shape this epic is about.

    The durable artifact is what a person reads; the exit status is what every
    other consumer reads, launchd's own accounting included. Falsifier: an
    alarming run exits 0, or an INDETERMINATE run does.
    """
    from scripts.lab_alarm import EXIT_ALARM, EXIT_INDETERMINATE, EXIT_OK, exit_code

    ledger = tmp_path / "ledger.md"
    ledger.write_text("| none |\n", encoding="utf-8")

    # Run 1 has no previous lag sample, so the lag condition is INDETERMINATE.
    baseline = _run(
        tmp_path,
        result=EnumLabPassResult.PASS,
        restarts="0 running",
        lag=3,
        ledger=ledger,
    )
    assert exit_code(baseline) == EXIT_INDETERMINATE

    healthy = _run(
        tmp_path,
        result=EnumLabPassResult.PASS,
        restarts="0 running",
        lag=3,
        ledger=ledger,
    )
    assert exit_code(healthy) == EXIT_OK

    alarming = _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=3,
        ledger=ledger,
    )
    assert exit_code(alarming) == EXIT_ALARM


@pytest.mark.unit
def test_the_delivered_sha_prefers_the_agent_over_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured 2026-09-21: /ready alone reports unresolved on a healthy lane.

    This lane's /ready answers healthy and carries a package version with no
    commit anywhere in its payload, while the deploy agent answered with a
    40-hex sha on the same host at the same moment. A resolver on /ready alone
    would leave the receipt condition permanently INDETERMINATE, which is the
    alarm half-blind rather than the alarm working.
    """
    import scripts.ci.lab_pass_receipt as receipt_module
    from scripts.lab_alarm import resolve_delivered_sha

    monkeypatch.setattr(
        receipt_module, "read_agent_loaded_code_sha", lambda url, **kw: "b" * 40
    )
    monkeypatch.setattr(receipt_module, "read_ready_revision", lambda url, **kw: "")
    sha, surface = resolve_delivered_sha(
        agent_url="http://lane:8098", ready_url="http://lane/ready"
    )
    assert sha == "b" * 40
    assert "deploy-agent" in surface


@pytest.mark.unit
def test_every_silent_sha_surface_is_named_rather_than_reported_as_a_bare_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader must be able to tell WHICH surface declined."""
    import scripts.ci.lab_pass_receipt as receipt_module
    from scripts.lab_alarm import resolve_delivered_sha

    monkeypatch.setattr(
        receipt_module, "read_agent_loaded_code_sha", lambda url, **kw: ""
    )
    monkeypatch.setattr(receipt_module, "read_ready_revision", lambda url, **kw: "")
    sha, surface = resolve_delivered_sha(agent_url="a", ready_url="b")
    assert sha == ""
    assert "deploy agent" in surface and "/ready" in surface


@pytest.mark.unit
def test_an_exited_container_inside_its_restart_bound_is_recorded_not_graded() -> None:
    """A measured limit, pinned so it stays deliberate rather than accidental.

    On 2026-09-21 ``omnimarket-projection-api`` read ``exited`` with a restart
    count of 0 on the real lane. This condition grades RESTARTS, so that is
    inside its bound and correctly not an alarm -- grading status would need
    its own per-container declaration, since the migration gate exits on
    purpose every boot and a writer does not. The status still travels in the
    evidence, so a reader can see what this condition chose not to grade.
    """
    report = evaluate_container_restarts(
        {"projection-api": 5},
        runner=_docker_runner({"projection-api": _completed("0 exited")}),
    )
    assert report.outcome is EnumConditionOutcome.OK
    assert "exited" in report.evidence
