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
CHANNEL = "#omninode-notifications"

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
        env_file=tmp_path / "absent.env",
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
    "APPROVED SCOPE: post lab alarms to #omninode-notifications | "
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
        "APPROVED SCOPE: post to #omninode-notifications | OUT OF SCOPE: |\n",
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
def test_the_live_ledger_authorizes_the_consented_channel() -> None:
    """The real grant, resolved from the real ledger rather than asserted.

    An operator ruling on 2026-09-21 authorized this alarm to post to
    #omninode-notifications using the existing lab bot token by reference.
    This reads that row back through the same resolver the alarm uses, so the
    authorization is proven by the mechanism and not by a dispatch message --
    no agent message is consent.
    """
    omni_home = os.environ.get("OMNI_HOME")
    if not omni_home:
        pytest.skip("OMNI_HOME is unset, so the shared ledger cannot be located")
    ledger = Path(omni_home) / "docs" / "tracking" / "ROLLING_WORK_LEDGER.md"
    if not ledger.is_file():
        pytest.skip("the shared registry ledger is not present on this host")

    consent = resolve_posting_consent(ledger, channel=CHANNEL)
    assert consent is not None, "the operator consent row no longer resolves"
    assert consent.approved_by.startswith("operator")
    assert consent.channel == CHANNEL

    # And the control that makes the line above mean something: the same live
    # ledger authorizes NO other channel.
    assert resolve_posting_consent(ledger, channel="#not-approved-anywhere") is None


@pytest.mark.unit
def test_a_ruling_shaped_consent_row_resolves(tmp_path: Path) -> None:
    """The shape the real authorization was actually recorded in.

    Rule 18's canonical row carries two labelled scope lists; the live row is
    a dated RULING carrying the same four substantive facts instead. The
    resolver requires the substance, and the refusal controls below are what
    keep that from being a rubber stamp.
    """
    ledger = tmp_path / "ledger.md"
    ledger.write_text(
        "2026-09-21T14:46:45Z | RULING | lane=foreground | OPERATOR-CONSENT Slack "
        "channel | OMN-18867 durable lab alarm may post to #omninode-notifications "
        "using the existing bot token by reference (SLACK_BOT_TOKEN, never printed); "
        "no new Slack app, no webhook | approved_by=operator at 2026-09-21 "
        "('channel approved')\n",
        encoding="utf-8",
    )
    consent = resolve_posting_consent(ledger, channel=CHANNEL)
    assert consent is not None
    assert consent.approved_by == "operator"
    assert consent.line == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "why"),
    [
        (
            "2026-09-21 | RULING | OPERATOR-CONSENT | may post to "
            "#omninode-notifications; no new Slack app",
            "no approver",
        ),
        (
            "2026-09-21 | RULING | OPERATOR-CONSENT | may post somewhere; "
            "no new Slack app | approved_by=operator",
            "no channel",
        ),
        (
            "2026-09-21 | RULING | OPERATOR-CONSENT | may post to "
            "#omninode-notifications | approved_by=operator",
            "no exclusion clause",
        ),
        (
            "2026-09-21 | RULING | may post to #omninode-notifications; "
            "no new Slack app | approved_by=operator",
            "not a consent row at all",
        ),
    ],
)
def test_a_row_missing_any_substantive_fact_refuses(
    tmp_path: Path, row: str, why: str
) -> None:
    """The controls that keep the widened resolver from being a rubber stamp."""
    ledger = tmp_path / "ledger.md"
    ledger.write_text(row + "\n", encoding="utf-8")
    assert resolve_posting_consent(ledger, channel=CHANNEL) is None, why


@pytest.mark.unit
def test_nothing_is_sent_without_a_resolvable_consent_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC6 falsifier: a posting path reachable with no row resolvable by line."""
    from scripts import lab_alarm

    sent: list[str] = []
    monkeypatch.setattr(
        lab_alarm, "post_alarm", lambda *a, **k: sent.append("sent") or "ts"
    )
    ledger = tmp_path / "ledger.md"
    ledger.write_text("| nothing authorizing here |\n", encoding="utf-8")

    run = _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=1,
        ledger=ledger,
    )
    assert run.raised, "the fixture must actually raise, or this proves nothing"
    assert sent == []
    assert run.posting.startswith("disabled")


@pytest.mark.unit
def test_the_token_never_reaches_the_run_record_or_the_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential is referenced by name and never recorded.

    Falsifier: the token value appears in the durable artifact, the rendered
    output, or a delivery id -- any of which turns a local log into a secret
    store.
    """
    from scripts import lab_alarm
    from scripts.lab_alarm import render

    secret = "xoxb-THIS-MUST-NEVER-BE-RECORDED"
    env_file = tmp_path / "bot.env"
    env_file.write_text(f"{lab_alarm.SLACK_TOKEN_VAR}={secret}\n", encoding="utf-8")
    assert lab_alarm.read_secret(env_file, lab_alarm.SLACK_TOKEN_VAR) == secret

    monkeypatch.setattr(lab_alarm, "post_alarm", lambda *a, **k: "1758470000.001")
    ledger = tmp_path / "ledger.md"
    ledger.write_text(_CONSENT + "\n", encoding="utf-8")

    run = _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=1,
        ledger=ledger,
    )
    recorded = (tmp_path / "alarm-runs.jsonl").read_text(encoding="utf-8")
    assert secret not in recorded
    assert secret not in render(run)
    assert secret not in run.posting


@pytest.mark.unit
def test_a_failed_delivery_is_recorded_rather_than_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alarm that believes it delivered and did not is this ticket's defect."""
    from scripts import lab_alarm
    from scripts.lab_alarm import PostingError

    def boom(*args: object, **kwargs: object) -> str:
        raise PostingError(
            "delivery to #omninode-notifications refused: channel_not_found"
        )

    monkeypatch.setattr(lab_alarm, "post_alarm", boom)
    ledger = tmp_path / "ledger.md"
    ledger.write_text(_CONSENT + "\n", encoding="utf-8")

    run = _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=1,
        ledger=ledger,
    )
    assert "FAILED" in run.posting
    assert "channel_not_found" in run.posting
    assert "delivered 0/" in run.posting


@pytest.mark.unit
def test_a_repeat_of_the_same_condition_delivers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edge-triggering IS the channel's deduplication.

    Falsifier: a condition that stays bad posts once an hour, which is how an
    alarm gets muted and becomes coverage nobody has.
    """
    from scripts import lab_alarm

    posts: list[str] = []
    monkeypatch.setattr(
        lab_alarm,
        "post_alarm",
        lambda alarm, **k: (posts.append(alarm.subject), "ts")[1],
    )
    ledger = tmp_path / "ledger.md"
    ledger.write_text(_CONSENT + "\n", encoding="utf-8")

    _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=1,
        ledger=ledger,
    )
    _run(
        tmp_path,
        result=EnumLabPassResult.FAIL,
        restarts="0 running",
        lag=1,
        ledger=ledger,
    )
    assert posts.count(SHA) == 1


@pytest.mark.unit
def test_the_token_is_never_taken_on_a_command_line() -> None:
    """A token on argv reaches every process listing on the host.

    The alarm's parser declares a path to an env file and no token option, so
    adding one turns this red rather than passing review.
    """
    from scripts.lab_alarm import build_parser

    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert "--env-file" in options
    assert not any("token" in option.lower() for option in options)


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


def _render_lab_alarm_plist(
    tmp_path: Path, *, drop_env: str | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the real installer's ``--render-only`` mode against a fixture env.

    Never touches launchd, plutil or the real ``~/Library/LaunchAgents`` --
    ``--render-only`` writes the substituted plist to an explicit target and
    exits before any of that, so this is safe on a CI runner with no brew
    interpreter (see the installer's own fallback for a missing one).
    """
    fake_omni_home = tmp_path / "omni_home"
    (fake_omni_home / "omnibase_infra" / "config").mkdir(parents=True)
    (fake_omni_home / "omnibase_infra" / "config" / "lab_alarm.json").write_text("{}")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "OMNI_HOME": str(fake_omni_home),
        "ONEX_INFRA_HOST": "lab.fixture.invalid",
        "ONEX_RUNTIME_SSH_HOST": "fixture-user@lab.fixture.invalid",
    }
    if drop_env is not None:
        del env[drop_env]
    return subprocess.run(
        [str(INSTALLER), "--render-only", str(tmp_path / "rendered.plist")],
        capture_output=True,
        check=False,
        env=env,
    )


@pytest.mark.unit
def test_a_render_carries_the_lab_host_and_ssh_target_from_the_installing_env(
    tmp_path: Path,
) -> None:
    """AC1's env-var half.

    The plist template's ``EnvironmentVariables`` carried ``OMNI_HOME`` and
    ``PATH`` but not the two vars ``config/lab_alarm.json`` needs to even
    load. Falsifier: a rendered plist that still carries an ``@..@``
    placeholder, which is exactly what crashed the agent at config load on
    the first real launchd fire -- measured live during the OMN-18867 AC1
    install, 2026-09-21: a manual shell run succeeded on the operator's
    inherited env while ``launchctl kickstart`` on that same installed plist
    died before evaluating any condition, because launchd never sources
    ``~/.zshrc``.
    """
    completed = _render_lab_alarm_plist(tmp_path)
    assert completed.returncode == 0, completed.stderr.decode()
    rendered = (tmp_path / "rendered.plist").read_text(encoding="utf-8")
    assert "lab.fixture.invalid" in rendered
    assert "fixture-user@lab.fixture.invalid" in rendered
    assert "@ONEX_INFRA_HOST@" not in rendered
    assert "@ONEX_RUNTIME_SSH_HOST@" not in rendered


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["ONEX_INFRA_HOST", "ONEX_RUNTIME_SSH_HOST"])
def test_the_installer_refuses_rather_than_render_an_empty_lab_host(
    tmp_path: Path, missing: str
) -> None:
    """Falsifier: the installer renders a blank instead of refusing.

    A silently empty ``${ONEX_INFRA_HOST}`` or ``${ONEX_RUNTIME_SSH_HOST}``
    is Operating Rule 8's exact failure mode -- a default the agent then
    fires against -- and the two-cluster / two-host confusion this repo has
    already hit once makes an empty lab address worse than a loud refusal.
    """
    completed = _render_lab_alarm_plist(tmp_path, drop_env=missing)
    assert completed.returncode != 0
    assert missing in completed.stderr.decode()
    assert not (tmp_path / "rendered.plist").exists()


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
