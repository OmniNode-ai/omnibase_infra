# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline branch coverage for provider-rung failures (OMN-17427)."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import provider_rung_canary_probe as probe

pytestmark = pytest.mark.unit

BACKEND = "fake-provider"
ENDPOINT = "https://provider.invalid/chat/completions"
ARGS = ["--container", "fake", "--user", "fake", "--contract-path", "/fake.yaml"]


@pytest.fixture(autouse=True)
def _no_subprocesses(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        pytest.fail("the offline canary tests must not launch a subprocess")

    monkeypatch.setattr(probe.subprocess, "run", refuse)


def _observation() -> dict[str, Any]:
    return {
        "runtime_binds_contract": True,
        "declared": [
            {
                "backend_id": BACKEND,
                "endpoint_url": ENDPOINT,
                "ref_name": "fake-provider-ref",
                "model_name": "fake-model",
            }
        ],
        "probes": [
            {
                "backend_ids": [BACKEND],
                "endpoint_url": ENDPOINT,
                "endpoint_host": "provider.invalid",
                "resolved": True,
                "request_sent": True,
                "http_status": 200,
                "body_is_chat_completion": True,
                "model_echo": "fake-model",
            }
        ],
        "controls": [
            {
                "endpoint_url": ENDPOINT,
                "endpoint_host": "provider.invalid",
                "http_status": 401,
            }
        ],
    }


def _assert_failure(record: probe.Record, verdict: str, check_name: str) -> None:
    assert record.verdict == "fail"
    assert record.exit_code == probe.EXIT_FINDINGS
    assert record.rungs[0]["verdict"] == verdict
    assert "0 of 1 probed backends LIVE" in record.detail
    assert check_name in {check.name for check in record.checks if not check.ok}


def test_pin_only_rung_is_a_typed_failure() -> None:
    obs = _observation()
    # A backend assignment contains no evidence of an accepted request.
    obs["probes"][0] = {
        "backend_ids": [BACKEND],
        "endpoint_url": ENDPOINT,
    }
    _assert_failure(probe.grade(obs), probe.PROBE_ERROR, f"rung/{BACKEND}")


def test_rung_with_no_attempts_is_a_typed_failure() -> None:
    obs = _observation()
    obs["probes"] = []
    _assert_failure(
        probe.grade(obs), probe.PROBE_ERROR, "every_credentialed_backend_probed"
    )


def test_rung_refused_by_budget_is_a_typed_failure() -> None:
    obs = _observation()
    obs["probes"][0] = {
        "backend_ids": [BACKEND],
        "endpoint_url": ENDPOINT,
        "resolved": True,
        "request_sent": False,
        "exception": "BudgetRefused",
        "exception_family": "budget",
    }
    _assert_failure(probe.grade(obs), probe.PROBE_ERROR, f"rung/{BACKEND}")


@pytest.mark.parametrize(
    ("field", "value", "verdict"),
    [
        ("endpoint_url", 17, probe.SKIPPED_NO_ENDPOINT),
        ("endpoint_url", "  ", probe.SKIPPED_NO_ENDPOINT),
        ("ref_name", [], probe.SKIPPED_NO_REF),
        ("ref_name", "  ", probe.SKIPPED_NO_REF),
    ],
)
def test_malformed_provider_row_cannot_report_ok(
    field: str, value: Any, verdict: str
) -> None:
    obs = _observation()
    obs["declared"][0][field] = value
    record = probe.grade(obs)
    assert record.verdict == "fail"
    assert record.exit_code == probe.EXIT_FINDINGS
    assert record.rungs[0]["disposition"] == verdict
    assert record.rungs[0]["verdict"] == verdict
    coverage = next(
        c for c in record.checks if c.name == "every_credentialed_backend_probed"
    )
    assert coverage.ok is False
    assert f"probed but not credentialed ['{BACKEND}']" in coverage.evidence


@pytest.mark.parametrize(
    ("fact", "verdict"),
    [
        ({"http_status": 429}, probe.QUOTA_DEAD),
        ({"http_status": 403}, probe.AUTH_DEAD),
        ({"exception_family": "transport"}, probe.UNREACHABLE),
        ({"http_status": 200}, probe.PROTOCOL_ERROR),
        ({"http_status": "200"}, probe.PROBE_ERROR),
        ({"resolved": False, "request_sent": False}, probe.UNRESOLVED),
    ],
)
def test_zero_accepted_completions_never_report_live(
    fact: dict[str, Any], verdict: str
) -> None:
    obs = _observation()
    obs["probes"][0] = {
        "backend_ids": [BACKEND],
        "endpoint_url": ENDPOINT,
        **fact,
    }
    _assert_failure(probe.grade(obs), verdict, f"rung/{BACKEND}")


def test_duplicate_probe_fails_even_when_both_requests_are_live() -> None:
    obs = _observation()
    obs["probes"].append(copy.deepcopy(obs["probes"][0]))
    record = probe.grade(obs)
    assert record.rungs[0]["verdict"] == probe.LIVE
    assert record.exit_code == probe.EXIT_FINDINGS
    assert record.failures == [
        f"every_credentialed_backend_probed: expected 1 ['{BACKEND}']; "
        f"not probed []; probed but not credentialed []; probed twice ['{BACKEND}']"
    ]


def test_malformed_controls_fail_the_wrong_key_coverage_check() -> None:
    obs = _observation()
    obs["controls"] = {"http_status": 401}
    record = probe.grade(obs)
    assert record.controls == []
    assert record.exit_code == probe.EXIT_FINDINGS
    assert record.failures == [
        "wrong_key_control_per_endpoint: endpoints 1; controlled 0; "
        f"missing ['{ENDPOINT}']"
    ]


@pytest.mark.parametrize(
    ("host", "rule"),
    [
        (None, {"match_endpoint_host": "provider.invalid"}),
        ("provider.invalid", {"match_endpoint_host": 17}),
        ("provider.invalid", {"match_endpoint_host": ""}),
        ("provider.invalid", {"match_endpoint_host": "provider.invalid"}),
        (
            "provider.invalid",
            {
                "match_endpoint_host": "provider.invalid",
                "codes": [None, {"code": "different"}],
            },
        ),
        ("provider.invalid", "malformed-rule"),
    ],
)
def test_invalid_or_unmatched_quota_rules_preserve_quota_dead(
    host: Any, rule: Any
) -> None:
    obs = _observation()
    obs["quota_policy"] = [rule]
    obs["probes"][0].update(
        {"http_status": 429, "endpoint_host": host, "provider_error_code": "limit"}
    )
    record = probe.grade(obs)
    _assert_failure(record, probe.QUOTA_DEAD, f"rung/{BACKEND}")
    assert record.rungs[0]["quota_class"] is None


@pytest.mark.parametrize(
    "error", [OSError("fake failure"), subprocess.TimeoutExpired("fake-docker", 1)]
)
def test_docker_start_failure_is_an_input_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(probe.subprocess, "run", fail)
    with pytest.raises(probe.ProbeInputError, match=type(error).__name__) as caught:
        probe._docker("fake-docker", ["inspect", "fake"], stdin=None, timeout=1)
    assert caught.value.__cause__ is error


def test_unreadable_observer_cannot_execute_or_report_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []

    def inspect_only(docker_bin: str, args: list[str], **kwargs: Any) -> str:
        calls.append(args)
        assert args[0] == "inspect"
        return "fake-image|running|fake-start"

    monkeypatch.setattr(probe, "_docker", inspect_only)
    monkeypatch.setattr(probe, "OBSERVER", tmp_path / "missing.py")
    assert probe.main(ARGS) == probe.EXIT_INPUT
    assert len(calls) == 1
    out = capsys.readouterr()
    assert "observer not readable" in out.err
    assert "GREEN" not in out.out


@pytest.mark.parametrize("payload", ["not JSON", "[]", '{"error":"fake error"}'])
def test_malformed_replay_is_an_input_error_without_a_record(
    payload: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(payload, encoding="utf-8")
    record = tmp_path / "record.json"
    assert probe.main([*ARGS, "--replay", str(replay), "--record", str(record)]) == (
        probe.EXIT_INPUT
    )
    assert not record.exists()
    out = capsys.readouterr()
    assert "could not run" in out.err
    assert "GREEN" not in out.out


@pytest.mark.parametrize("accepted", [False, True])
def test_replay_writes_consistent_record_summary_and_exit_code(
    accepted: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    obs = _observation()
    if not accepted:
        obs["probes"][0]["http_status"] = 429
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(obs), encoding="utf-8")
    record_path = tmp_path / "record.json"
    summary_path = tmp_path / "summary.md"
    summary_path.write_text("existing summary\n", encoding="utf-8")
    rc = probe.main(
        [
            *ARGS,
            "--replay",
            str(replay),
            "--record",
            str(record_path),
            "--summary",
            str(summary_path),
        ]
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    summary = summary_path.read_text(encoding="utf-8")
    out = capsys.readouterr()
    verdict = probe.LIVE if accepted else probe.QUOTA_DEAD
    assert rc == (probe.EXIT_OK if accepted else probe.EXIT_FINDINGS)
    assert record["verdict"] == ("pass" if accepted else "fail")
    assert record["rungs"][0]["verdict"] == verdict
    assert summary.startswith("existing summary\n")
    assert f"| {BACKEND} | {verdict} |" in summary
    assert "| provider.invalid | AUTH_DEAD | 401 |" in summary
    assert ("Failures:" in summary) is not accepted
    if accepted:
        assert record["failures"] == []
        assert "GREEN" in out.out
        assert out.err == ""
    else:
        assert record["failures"][0].startswith(f"rung/{BACKEND}: QUOTA_DEAD")
        assert "QUOTA_DEAD" in out.err
        assert "RED" in out.err
        assert "GREEN" not in out.out


def test_replay_can_report_ok_without_writing_a_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(_observation()), encoding="utf-8")
    assert probe.main([*ARGS, "--replay", str(replay)]) == probe.EXIT_OK
    assert "GREEN" in capsys.readouterr().out
