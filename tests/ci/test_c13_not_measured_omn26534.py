# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A lab model that did not answer grades NOT_MEASURED, never a product FAIL (OMN-26534).

``c13-timeout-receipt-run-45164398043.json`` is the receipt the failed scheduled
run 45164398043 wrote: its local rung ended ``failure_class`` ``timeout``,
``error_message`` ``request timed out``, and the grader called that FAIL, as if
the product had broken. Each test starts from the recorded passing session of
OMN-19179 and swaps in that receipt, or breaks one fact of it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c13_customer_local_probe as probe

REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_OBSERVATIONS = (
    REPO_ROOT / "tests" / "fixtures" / "omn19179" / "observations_lab_pass.json"
)
TIMEOUT_RECEIPT = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn26534"
    / "c13-timeout-receipt-run-45164398043.json"
)


def _observations(
    receipt: dict[str, Any] | None = None, **attempt: Any
) -> dict[str, Any]:
    """The passing session with the failed-run receipt: the local rung got no answer."""
    obs: dict[str, Any] = json.loads(PASS_OBSERVATIONS.read_text(encoding="utf-8"))
    recorded = (
        receipt or json.loads(TIMEOUT_RECEIPT.read_text(encoding="utf-8"))["receipt"]
    )
    recorded = copy.deepcopy(recorded)
    recorded["attempts"][0].update(attempt)
    obs["run_files"]["receipt.json"] = json.dumps(recorded)
    obs["run_files"]["result.txt"] = ""
    obs["steps"]["configured"]["returncode"] = 1
    return obs


@pytest.mark.unit
@pytest.mark.live_contact(
    "tests/fixtures/omn26534/c13-timeout-receipt-run-45164398043.json"
)
def test_a_timed_out_local_attempt_is_not_measured_with_a_lab_cause(
    recorded_response: dict[str, Any],
) -> None:
    """Replay the receipt run 45164398043 wrote: the lab server timed out."""
    receipt = recorded_response["receipt"]
    assert receipt["attempts"][0]["failure_class"] == "timeout"
    assert receipt["attempts"][0]["error_message"] == "request timed out"
    record = probe.grade(_observations(receipt), as_of="2026-11-23T11:21:00Z")
    assert record.verdict == "NOT_MEASURED", [c.as_dict() for c in record.clauses]
    assert record.as_dict()["cause_class"] == "lab-model"
    summary = probe.render_summary(record)
    assert "NOT_MEASURED" in summary
    assert "lab-model" in summary


@pytest.mark.unit
@pytest.mark.parametrize("failure_class", ["timeout", "provider_call_failed"])
def test_both_lab_failure_classes_are_not_measured(failure_class: str) -> None:
    record = probe.grade(_observations(failure_class=failure_class))
    assert (record.verdict, record.cause_class) == ("NOT_MEASURED", "lab-model")


@pytest.mark.unit
def test_not_measured_has_its_own_exit_and_writes_the_cause() -> None:
    record = probe.grade(_observations())
    assert probe.exit_code(record) == probe.EXIT_NOT_MEASURED == 3
    assert record.as_dict()["verdict"] == "NOT_MEASURED"
    assert record.as_dict()["cause_class"] == "lab-model"
    failing = probe.grade(_observations(tier="cloud"))
    passing = probe.grade(json.loads(PASS_OBSERVATIONS.read_text(encoding="utf-8")))
    assert probe.exit_code(failing) == probe.EXIT_FINDINGS
    assert probe.exit_code(passing) == probe.EXIT_OK


@pytest.mark.unit
def test_a_pass_has_no_cause_and_a_product_fail_names_the_product() -> None:
    passing = json.loads(PASS_OBSERVATIONS.read_text(encoding="utf-8"))
    assert probe.grade(passing).cause_class is None
    receipt = json.loads(passing["run_files"]["receipt.json"])
    receipt["model"] = "not-a-served-model"
    passing["run_files"]["receipt.json"] = json.dumps(receipt)
    record = probe.grade(passing)
    assert (record.verdict, record.cause_class) == ("FAIL", "product")


@pytest.mark.unit
@pytest.mark.parametrize(
    "attempt",
    [
        {"tier": "cloud"},
        {"failure_class": "provider_auth_failed"},
        {"failure_class": "quality_gate_failed"},
        {"failure_class": None},
    ],
)
def test_a_failed_attempt_that_is_not_a_local_lab_failure_stays_fail(
    attempt: dict[str, Any],
) -> None:
    record = probe.grade(_observations(**attempt))
    assert (record.verdict, record.cause_class) == ("FAIL", "product")


@pytest.mark.unit
def test_a_cloud_rung_answering_after_a_local_timeout_is_a_product_fail() -> None:
    """The ladder left the local tier: a defect, not a lab cause."""
    obs = _observations()
    receipt = json.loads(obs["run_files"]["receipt.json"])
    receipt["attempts"].append(
        {"tier": "cloud", "failure_class": None, "acceptance_decision": "accept"}
    )
    obs["run_files"]["receipt.json"] = json.dumps(receipt)
    assert probe.grade(obs).verdict == "FAIL"


@pytest.mark.unit
def test_a_lab_timeout_does_not_hide_a_product_defect() -> None:
    """An external connect in the same session is still a product FAIL."""
    obs = _observations()
    obs["steps"]["configured"]["strace"] += (
        "4242 connect(9, {sa_family=AF_INET, sin_port=htons(443), "
        'sin_addr=inet_addr("34.117.59.81")}, 16) = -1 EINPROGRESS\n'
    )
    assert probe.grade(obs).verdict == "FAIL"
