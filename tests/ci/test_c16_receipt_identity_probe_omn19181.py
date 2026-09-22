# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C16 receipt-identity producer (OMN-19181).

Every test here runs against RECORDED observations under
``tests/fixtures/omn19181/`` -- captured read-only from the .201 dev lane's
gateway and orchestrator-projection rows, provenance in each file -- and performs no
network or database I/O, so the verdict is falsifiable on a laptop and in CI
rather than only on the lab.

The three captured fixtures are real: a completed canary run with a typed dead
run (PASS), a run that died on a provider 429 with NO class or code, and a run
still reading ``published`` two days after submission. The remaining cases are
single-field mutations of the passing capture, each named for the one thing it
changes, so a red here always points at one clause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c16_receipt_identity_probe as probe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn19181"
CAPTURED = (
    "lane_healthy_typed_death.json",
    "lane_dead_run_untyped_cause.json",
    "lane_dead_run_never_terminal.json",
)


def _load(name: str = "lane_healthy_typed_death.json") -> dict[str, Any]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _outcomes(payload: dict[str, Any]) -> dict[str, str]:
    record = probe.grade(probe.observations_from_replay(payload))
    return {r.probe: r.outcome for r in record.results}


@pytest.mark.unit
def test_the_three_probes_are_the_criterions_own_and_all_are_graded() -> None:
    assert probe.PROBES == ("R-DELEG-11", "R-DELEG-12", "R-DELEG-26")
    record = probe.grade(probe.observations_from_replay(_load()))
    assert tuple(r.probe for r in record.results) == probe.PROBES


@pytest.mark.unit
@pytest.mark.parametrize("name", CAPTURED)
def test_each_captured_lane_record_grades_as_its_provenance_says(name: str) -> None:
    payload = _load(name)
    assert _outcomes(payload) == payload["expected"]


@pytest.mark.unit
def test_healthy_lane_data_can_pass_and_exits_zero() -> None:
    """The control: a check that cannot pass is worse than one that cannot fail."""
    record = probe.grade(probe.observations_from_replay(_load()))
    assert record.verdict == "pass"
    assert record.exit_code == probe.EXIT_OK


@pytest.mark.unit
def test_skip_skip_pass_is_red_and_never_a_pass() -> None:
    """AC4's falsifier: two probes whose subject never materialised, one PASS."""
    payload = _load()
    healthy = payload["observations"]["healthy"]
    healthy["receipt"] = None
    healthy["status"] = {"status": "published"}
    healthy["error"] = "budget spent before a terminal status"
    record = probe.grade(probe.observations_from_replay(payload))
    outcomes = {r.probe: r.outcome for r in record.results}
    assert outcomes == {
        "R-DELEG-11": "SKIP",
        "R-DELEG-12": "SKIP",
        "R-DELEG-26": "PASS",
    }
    assert record.verdict == "fail"
    assert record.exit_code == probe.EXIT_FINDINGS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("side", "key", "value"),
    [
        ("receipt", "route", "cloud-gemini-pro"),
        ("receipt", "provider", "gemini"),
        ("receipt", "terminal_model_used", "claude-opus-4-6"),
    ],
)
def test_a_receipt_naming_a_route_the_orchestrator_did_not_record_fails(
    side: str, key: str, value: str
) -> None:
    payload = _load()
    payload["observations"]["healthy"][side][key] = value
    result = probe.grade(probe.observations_from_replay(payload)).results[1]
    assert result.probe == "R-DELEG-12"
    assert result.outcome == "FAIL"
    assert any(repr(value) in reason for reason in result.reasons)


@pytest.mark.unit
def test_two_blank_routes_are_not_an_identity() -> None:
    payload = _load()
    state = payload["observations"]["healthy_state"]["payload"]
    payload["observations"]["healthy"]["receipt"]["route"] = None
    state["routing_decision"]["route"] = None
    state["inference_route"] = None
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_path",
    [
        ("routing_decision", "route"),
        ("inference_route",),
        ("routing_decision", "provider"),
        ("inference_model_used",),
    ],
)
def test_each_independent_observation_is_compared_not_just_one(
    field_path: tuple[str, ...],
) -> None:
    """A re-route whose answering attempt disagrees with the receipt is caught."""
    payload = _load()
    target = payload["observations"]["healthy_state"]["payload"]
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = "cloud-gemini-pro"
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"


@pytest.mark.unit
def test_an_unterminated_projection_fails_and_an_unreadable_one_skips() -> None:
    payload = _load()
    payload["observations"]["healthy_state"].update(
        state="INFERENCE_PENDING", payload=None
    )
    assert _outcomes(payload)["R-DELEG-12"] == "FAIL"
    payload = _load()
    payload["observations"]["healthy_state"]["error"] = (
        "read failed: InsufficientPrivilege"
    )
    assert _outcomes(payload)["R-DELEG-12"] == "SKIP"


@pytest.mark.unit
@pytest.mark.parametrize("passed", [None, "skipped", 1])
def test_a_rule_without_a_boolean_verdict_on_a_completed_run_fails(passed: Any) -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["rule_evaluations"][0]["passed"] = (
        passed
    )
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
def test_a_completed_run_with_no_evaluated_rule_fails() -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["rule_evaluations"] = []
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
def test_a_failed_blocking_rule_on_a_completed_run_fails() -> None:
    payload = _load()
    payload["observations"]["healthy"]["receipt"]["rule_evaluations"][0]["passed"] = (
        False
    )
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
@pytest.mark.parametrize("surface", ["status", "receipt"])
def test_a_dying_run_that_reports_success_fails(surface: str) -> None:
    payload = _load()
    payload["observations"]["dying"][surface]["status"] = "completed"
    assert _outcomes(payload)["R-DELEG-11"] == "FAIL"


@pytest.mark.unit
def test_a_dying_submission_refused_at_ingress_skips_and_is_red() -> None:
    payload = _load()
    dying = payload["observations"]["dying"]
    dying.update(
        submit_status=400, status=None, receipt=None, error="submit answered 400"
    )
    record = probe.grade(probe.observations_from_replay(payload))
    assert {r.probe: r.outcome for r in record.results}["R-DELEG-26"] == "SKIP"
    assert record.exit_code == probe.EXIT_FINDINGS


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("terminal_failure_class", "provider said no"),
        ("terminal_failure_code", "required_bar_missing"),
    ],
)
def test_a_cause_outside_the_typed_grammar_fails(key: str, value: str) -> None:
    payload = _load()
    payload["observations"]["dying"]["receipt"][key] = value
    payload["observations"]["dying"]["status"][key] = value
    assert _outcomes(payload)["R-DELEG-26"] == "FAIL"


@pytest.mark.unit
def test_status_and_receipt_disagreeing_on_the_cause_fails() -> None:
    payload = _load()
    payload["observations"]["dying"]["status"]["terminal_failure_code"] = None
    assert _outcomes(payload)["R-DELEG-26"] == "FAIL"


@pytest.mark.unit
def test_the_dying_run_differs_from_the_healthy_one_only_in_task_class() -> None:
    diff = {
        key
        for key in probe.HEALTHY_PAYLOAD
        if probe.HEALTHY_PAYLOAD[key] != probe.DYING_PAYLOAD[key]
    }
    assert diff == {"task_type"}
    assert set(probe.HEALTHY_PAYLOAD) == set(probe.DYING_PAYLOAD)


@pytest.mark.unit
def test_the_header_form_is_pinned() -> None:
    assert probe.HEADER_NAME == "x-api-key"


@pytest.mark.unit
def test_the_record_carries_no_credential_tenant_or_endpoint(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(_load()), encoding="utf-8")
    out = tmp_path / "record.json"
    assert probe.main(["--replay", str(replay), "--record", str(out)]) == probe.EXIT_OK
    text = out.read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["criterion"] == "C16"
    assert [p["probe"] for p in record["probes"]] == list(probe.PROBES)
    for forbidden in ("tenant_id", "endpoint_url", "api_key", "prompt"):
        assert forbidden not in text


@pytest.mark.unit
def test_a_missing_secret_is_an_input_failure_and_still_writes_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMN19181_TEST_UNSET", raising=False)
    out = tmp_path / "record.json"
    code = probe.main(
        [
            "--base-url",
            "http://127.0.0.1:9",
            "--credential-env",
            "OMN19181_TEST_UNSET",
            "--projection-dsn-env",
            "OMN19181_TEST_UNSET",
            "--runner-identity",
            "test",
            "--record",
            str(out),
        ]
    )
    assert code == probe.EXIT_INPUT
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "could_not_run"
