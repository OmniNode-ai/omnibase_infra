# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C14 customer-local routing producer (OMN-19180).

Every test grades a RECORDED ``{collection, run}`` from
``tests/fixtures/omn19180/`` -- a real run of the probe on a no-checkout
install, with machine paths normalised -- and mutates one fact at a time. No
network, no installed product, so the grader's verdict is falsifiable on a
laptop and in CI rather than only on the lab.

The recorded run is GREEN, which is the positive control the criterion needs:
the grader can pass on a healthy install. Every other test flips exactly one
fact and requires the grader to go red naming it, or -- for a blind instrument
or a machine that is not a customer's -- to refuse to grade at all.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c14_customer_local_routing_probe as probe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn19180"


def _green() -> dict[str, Any]:
    payload = json.loads((FIXTURES / "green.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return copy.deepcopy(payload)


def _grade(payload: dict[str, Any]) -> probe.Record:
    return probe.grade(payload["collection"], payload.get("run"))


def _row(record: probe.Record, name: str) -> probe.RowResult:
    return next(row for row in record.rows if row.name == name)


@pytest.mark.unit
def test_the_recorded_healthy_install_grades_pass() -> None:
    """The positive control: a real no-checkout run where every row held."""
    record = _grade(_green())
    assert record.verdict == "PASS", [row.findings for row in record.rows]
    assert record.exit_code == probe.EXIT_OK
    assert [row.name for row in record.rows] == [
        "row1_local_answer_ends_chain",
        "row2_no_unbindable_tier",
        "row4_reviewer_leg_unmetered",
    ]


# --- row 2: every customer-visible class, every tier, every backend ----------


@pytest.mark.unit
def test_a_tier_naming_a_deleted_backend_is_red() -> None:
    payload = _green()
    payload["collection"]["walk"]["research"][0]["candidates"].append("local-retired")
    record = _grade(payload)
    row = _row(record, "row2_no_unbindable_tier")
    assert not row.ok
    assert any("'local-retired'" in f and "deleted_backend" in f for f in row.findings)
    assert record.exit_code == probe.EXIT_FINDINGS


@pytest.mark.unit
def test_a_tier_naming_an_unreachable_local_backend_is_red() -> None:
    payload = _green()
    payload["collection"]["backends"]["local-coder"]["classification"] = (
        probe.UNREACHABLE
    )
    row = _row(_grade(payload), "row2_no_unbindable_tier")
    assert not row.ok
    assert any("local-coder" in f and "unreachable" in f for f in row.findings)


@pytest.mark.unit
def test_a_tier_whose_backend_has_no_endpoint_on_this_machine_is_red() -> None:
    payload = _green()
    payload["collection"]["backends"]["local-heavy-reasoning"]["classification"] = (
        probe.NO_ENDPOINT
    )
    row = _row(_grade(payload), "row2_no_unbindable_tier")
    assert not row.ok
    assert any("no_endpoint_on_this_machine" in f for f in row.findings)


@pytest.mark.unit
def test_a_tier_that_serves_the_class_with_nothing_is_red() -> None:
    payload = _green()
    payload["collection"]["walk"]["test"][0]["candidates"] = []
    row = _row(_grade(payload), "row2_no_unbindable_tier")
    assert not row.ok
    assert any(probe.TIER_SERVES_NOTHING in f for f in row.findings)


@pytest.mark.unit
def test_an_undeclared_tier_in_a_class_order_is_red() -> None:
    payload = _green()
    payload["collection"]["walk"]["review"].append(
        {"tier": "ghost", "declared": False, "candidates": []}
    )
    row = _row(_grade(payload), "row2_no_unbindable_tier")
    assert not row.ok
    assert any(probe.TIER_UNDECLARED in f for f in row.findings)


@pytest.mark.unit
def test_an_empty_enumeration_is_red_never_a_vacuous_pass() -> None:
    """AC5: zero customer-visible classes cannot read as 'none are unbindable'."""
    payload = _green()
    payload["collection"]["public_task_classes"] = []
    payload["collection"]["walk"] = {}
    row = _row(_grade(payload), "row2_no_unbindable_tier")
    assert not row.ok
    assert any("ZERO customer-visible classes" in f for f in row.findings)


@pytest.mark.unit
def test_a_class_missing_from_the_walk_is_red_not_skipped() -> None:
    """AC5: an enumeration that silently shrinks is the failure, not a skip."""
    payload = _green()
    del payload["collection"]["walk"]["summarization"]
    row = _row(_grade(payload), "row2_no_unbindable_tier")
    assert not row.ok
    assert any(
        "absent from the walk" in f and "summarization" in f for f in row.findings
    )


@pytest.mark.unit
def test_the_loader_and_the_shipped_file_must_agree_on_the_public_set() -> None:
    payload = _green()
    payload["collection"]["public_task_classes_from_contract_yaml"].append("escalation")
    row = _row(_grade(payload), "row2_no_unbindable_tier")
    assert not row.ok
    assert any("disagrees with the shipped contract" in f for f in row.findings)


# --- row 1: the ladder on the run's own receipt --------------------------------


def _receipt(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["run"]["receipt_result"]
    assert isinstance(result, dict)
    return result


@pytest.mark.unit
def test_a_ladder_that_climbs_past_local_is_red() -> None:
    payload = _green()
    receipt = _receipt(payload)
    local = dict(receipt["attempts"][0])
    local.update(acceptance_decision="climb", acceptance_reason="quality_bar_not_met")
    cloud = dict(receipt["attempts"][0])
    cloud.update(tier="cheap_cloud", backend_id="cloud-glm", cost_usd=0.0001)
    receipt["attempts"] = [local, cloud]
    receipt["attempts_count"] = 2
    receipt["escalation_count"] = 1
    row = _row(_grade(payload), "row1_local_answer_ends_chain")
    assert not row.ok
    assert any("left the local tier" in f for f in row.findings)
    assert any("escalation_count" in f for f in row.findings)


@pytest.mark.unit
def test_a_rung_after_an_accepted_local_answer_is_red() -> None:
    payload = _green()
    receipt = _receipt(payload)
    receipt["attempts"] = [receipt["attempts"][0], dict(receipt["attempts"][0])]
    receipt["attempts_count"] = 2
    row = _row(_grade(payload), "row1_local_answer_ends_chain")
    assert not row.ok
    assert any("AFTER an accepted local answer" in f for f in row.findings)


@pytest.mark.unit
def test_an_empty_ladder_is_red() -> None:
    """The OMN-18889 shape: a ladder dropped on the way to the record."""
    payload = _green()
    receipt = _receipt(payload)
    receipt["attempts"] = []
    row = _row(_grade(payload), "row1_local_answer_ends_chain")
    assert not row.ok
    assert any("EMPTY attempt ladder" in f for f in row.findings)


@pytest.mark.unit
def test_an_accept_with_no_typed_reason_is_red() -> None:
    payload = _green()
    _receipt(payload)["attempts"][0]["acceptance_reason"] = None
    row = _row(_grade(payload), "row1_local_answer_ends_chain")
    assert not row.ok
    assert any("no typed reason" in f for f in row.findings)


@pytest.mark.unit
def test_a_failed_run_is_red() -> None:
    payload = _green()
    payload["run"]["exit_code"] = 1
    _receipt(payload)["status"] = "failed"
    row = _row(_grade(payload), "row1_local_answer_ends_chain")
    assert not row.ok
    assert any("did not complete" in f for f in row.findings)


@pytest.mark.unit
def test_no_recorded_run_is_red_never_a_skip() -> None:
    payload = _green()
    payload["run"] = None
    record = _grade(payload)
    assert not _row(record, "row1_local_answer_ends_chain").ok
    assert not _row(record, "row4_reviewer_leg_unmetered").ok
    assert record.verdict == "FAIL"


@pytest.mark.unit
def test_an_egress_record_with_no_post_is_an_unproven_zero() -> None:
    """The instrument must have seen the accepted rung's own call."""
    payload = _green()
    payload["run"]["egress"] = [
        entry for entry in payload["run"]["egress"] if entry["method"] != "POST"
    ]
    row = _row(_grade(payload), "row1_local_answer_ends_chain")
    assert not row.ok
    assert any("unproven zero" in f for f in row.findings)


@pytest.mark.unit
def test_more_inference_calls_than_rungs_is_red() -> None:
    payload = _green()
    post = next(e for e in payload["run"]["egress"] if e["method"] == "POST")
    payload["run"]["egress"].append(dict(post))
    row = _row(_grade(payload), "row1_local_answer_ends_chain")
    assert not row.ok
    assert any("does not account for" in f for f in row.findings)


@pytest.mark.unit
def test_a_later_call_to_a_metered_provider_is_red_on_rows_1_and_4() -> None:
    payload = _green()
    payload["run"]["egress"].append(
        {
            "method": "CONNECT",
            "target": "generativelanguage.googleapis.com:443",
            "private": False,
        }
    )
    record = _grade(payload)
    assert not _row(record, "row1_local_answer_ends_chain").ok
    row4 = _row(record, "row4_reviewer_leg_unmetered")
    assert not row4.ok
    assert any("metered" in f for f in row4.findings)


# --- row 4: the reviewer leg -----------------------------------------------------


@pytest.mark.unit
def test_the_shipped_metered_reviewer_default_is_red() -> None:
    """The state measured on the published release: the judge rides Gemini."""
    payload = _green()
    payload["collection"]["judge"] = {
        "backend_id": "cloud-glm-judge",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "endpoint_private": False,
        "model_id": "gemini-2.5-flash",
        "provider": "gemini",
        "secret_ref": "llm.gemini.api_key",
    }
    row = _row(_grade(payload), "row4_reviewer_leg_unmetered")
    assert not row.ok
    assert any("metered provider" in f for f in row.findings)
    assert any("llm.gemini.api_key" in f for f in row.findings)


@pytest.mark.unit
def test_an_unresolvable_reviewer_is_red() -> None:
    payload = _green()
    payload["collection"]["judge"] = {"error": "ProtocolConfigurationError: no backend"}
    row = _row(_grade(payload), "row4_reviewer_leg_unmetered")
    assert not row.ok


@pytest.mark.unit
def test_a_local_reviewer_that_does_not_serve_its_model_is_red() -> None:
    payload = _green()
    payload["collection"]["judge"]["classification"] = probe.MODEL_NOT_SERVED
    row = _row(_grade(payload), "row4_reviewer_leg_unmetered")
    assert not row.ok


# --- the machine and the controls: exit 2, never a verdict -------------------------


@pytest.mark.unit
def test_a_blind_classifier_refuses_to_grade() -> None:
    payload = _green()
    payload["collection"]["controls"][0]["classification"] = "bindable"
    record = _grade(payload)
    assert record.verdict == "COULD_NOT_RUN"
    assert record.exit_code == probe.EXIT_INPUT


@pytest.mark.unit
def test_a_missing_control_refuses_to_grade() -> None:
    payload = _green()
    payload["collection"]["controls"] = []
    assert _grade(payload).exit_code == probe.EXIT_INPUT


@pytest.mark.unit
def test_a_checkout_is_not_a_customer_machine() -> None:
    payload = _green()
    payload["collection"]["modules"]["omnimarket"]["git_ancestor"] = "/src/omnimarket"
    record = _grade(payload)
    assert record.exit_code == probe.EXIT_INPUT
    assert "checkout" in record.detail


@pytest.mark.unit
def test_a_package_outside_site_packages_is_not_a_customer_machine() -> None:
    payload = _green()
    payload["collection"]["modules"]["omnibase_infra"]["in_site_packages"] = False
    assert _grade(payload).exit_code == probe.EXIT_INPUT


@pytest.mark.unit
def test_an_edited_routing_contract_is_not_the_shipped_one() -> None:
    payload = _green()
    payload["collection"]["bindings"]["DELEGATION_ROUTING_TIERS_PATH"][
        "matches_shipped"
    ] = False
    record = _grade(payload)
    assert record.exit_code == probe.EXIT_INPUT
    assert "byte-identical" in record.detail


@pytest.mark.unit
def test_a_non_isolated_interpreter_is_not_a_customer_machine() -> None:
    payload = _green()
    payload["collection"]["isolated_interpreter"] = False
    assert _grade(payload).exit_code == probe.EXIT_INPUT


@pytest.mark.unit
def test_a_collector_failure_refuses_to_grade() -> None:
    record = _grade(
        {"collection": {"collect_error": "ImportError: omnimarket"}, "run": {}}
    )
    assert record.exit_code == probe.EXIT_INPUT


# --- the private-address rule and the entry point ------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("url", "private"),
    [
        ("http://127.0.0.1:8000/v1/chat/completions", True),
        ("http://10.4.0.9:8000/v1/models", True),
        ("https://8.8.8.8/v1/chat/completions", False),
        ("not-a-url", False),
    ],
)
def test_endpoint_is_private_uses_addresses_not_names(url: str, private: bool) -> None:
    assert probe.endpoint_is_private(url) is private


@pytest.mark.unit
def test_replay_entry_point_writes_the_record_and_returns_the_verdict(
    tmp_path: Path,
) -> None:
    payload = _green()
    payload["collection"]["walk"]["research"][0]["candidates"].append("local-retired")
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(payload), encoding="utf-8")
    record_path = tmp_path / "record.json"
    summary = tmp_path / "summary.md"
    code = probe.main(
        [
            "--replay",
            str(replay),
            "--record",
            str(record_path),
            "--summary",
            str(summary),
        ]
    )
    assert code == probe.EXIT_FINDINGS
    written = json.loads(record_path.read_text(encoding="utf-8"))
    assert written["criterion"] == "C14"
    assert written["verdict"] == "FAIL"
    assert "row2_no_unbindable_tier" in written["detail"]
    assert "C14" in summary.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_live_mode_refuses_without_a_customer_interpreter() -> None:
    assert probe.main([]) == probe.EXIT_INPUT


@pytest.mark.unit
def test_a_customer_command_sees_only_the_allowlisted_environment() -> None:
    env = probe.customer_environment(
        {
            "HOME": "/opt/customer/home",
            "PATH": "/opt/customer/venv/bin:/usr/bin",
            "PYTHONPATH": "/src/omnimarket/src",
            "OMNI_HOME": "/src",
            "BIFROST_OVERLAY_PATH": "/opt/customer/home/overlay.yaml",
            "GEMINI_API_KEY": "not-a-real-key",
        }
    )
    assert set(env) == {"HOME", "PATH", "BIFROST_OVERLAY_PATH"}
