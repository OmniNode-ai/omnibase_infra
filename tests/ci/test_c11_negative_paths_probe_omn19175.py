# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C11 negative-paths producer (OMN-19175).

Every test here runs against RECORDED observations under
``tests/fixtures/omn19175/`` and performs no network I/O, so the grader's
verdict is falsifiable on a laptop and in CI rather than only on the lab.

The live half -- whether the deployed platform actually emits those
observations -- is the workflow's job and is deliberately not asserted here.
What is asserted is the property the criterion turns on: four refusals, each
typed, each distinguishable from the other three, and a positive control in the
same invocation so an all-refused result cannot be produced by a dead endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c11_negative_paths_probe as probe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn19175"


def _load(name: str) -> dict[str, Any]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _observations(name: str) -> dict[str, probe.Observation]:
    return probe.observations_from_replay(_load(name))


@pytest.mark.unit
def test_the_four_cases_and_the_control_are_declared_up_front() -> None:
    """AC4's shape: the expectation is in the grader, never read off a response."""
    assert probe.CONTROL.name == "positive_control"
    assert [case.name for case in probe.CASES] == [
        "wrong_key",
        "absent_key",
        "wrong_tenant",
        "malformed",
    ]
    # Three coded arms with pre-stated, pairwise-distinct codes; the malformed
    # arm is graded on the 422 envelope and declares no code, because the
    # platform emits none and a producer must not invent one.
    coded = [case.expected_code for case in probe.CASES if case.expected_code]
    assert coded == [
        "auth.credential.invalid",
        "auth.credential.absent",
        "auth.tenant.forbidden",
    ]
    assert len(set(coded)) == len(coded)
    malformed = next(case for case in probe.CASES if case.name == "malformed")
    assert malformed.expected_code is None
    assert malformed.expected_missing_field == "runner_identity"


@pytest.mark.unit
def test_the_header_form_is_pinned() -> None:
    """Bearer returns the same invalid code with a different detail (OMN-19175).

    Two arms sending different header forms would collide on one code, so the
    producer pins one form and this pins the pin.
    """
    assert probe.HEADER_NAME == "x-api-key"


@pytest.mark.unit
def test_a_healthy_lane_grades_pass() -> None:
    record = probe.grade(_observations("lane_healthy.json"))
    assert record.verdict == "pass"
    assert record.exit_code == 0
    assert record.positive_control.ok is True
    assert [case.ok for case in record.cases] == [True, True, True, True]
    observed = [case.observed_code for case in record.cases]
    assert observed[:3] == [
        "auth.credential.invalid",
        "auth.credential.absent",
        "auth.tenant.forbidden",
    ]
    assert observed[3] is None


@pytest.mark.unit
def test_an_unreachable_endpoint_cannot_produce_a_pass() -> None:
    """AC4, falsified directly: everything refuses because nothing is serving."""
    record = probe.grade(_observations("lane_unreachable.json"))
    assert record.verdict == "fail"
    assert record.exit_code == 1
    assert record.positive_control.ok is False
    assert "positive_control" in record.detail


@pytest.mark.unit
def test_two_arms_collapsing_onto_one_code_is_refused() -> None:
    """The regression OMN-18042 landed the codes to prevent: a bare 401 for both."""
    record = probe.grade(_observations("codes_collapsed.json"))
    assert record.verdict == "fail"
    assert record.exit_code == 1
    assert any("auth.credential.invalid" in line for line in record.failures)


@pytest.mark.unit
def test_a_cross_tenant_read_that_succeeds_is_refused() -> None:
    """A 200 on another tenant's keys is the leak the case exists to catch."""
    record = probe.grade(_observations("cross_tenant_leak.json"))
    assert record.verdict == "fail"
    wrong_tenant = next(case for case in record.cases if case.name == "wrong_tenant")
    assert wrong_tenant.ok is False
    assert wrong_tenant.observed_status == 200


@pytest.mark.unit
def test_a_malformed_body_that_does_not_name_the_field_is_refused() -> None:
    """DECISION 3: graded on detail[].type == missing and detail[].loc.

    A 422 whose envelope names no offending field does not satisfy "a typed
    refusal naming the defect", and a producer must not synthesize a code to
    cover for it.
    """
    record = probe.grade(_observations("malformed_unnamed_field.json"))
    assert record.verdict == "fail"
    malformed = next(case for case in record.cases if case.name == "malformed")
    assert malformed.ok is False
    assert "runner_identity" in malformed.reason


@pytest.mark.unit
def test_a_malformed_arm_answered_by_auth_is_refused() -> None:
    """Auth resolves before validation, so a 401 here means the key never worked.

    Reporting that as a malformed refusal would grade a credential failure as a
    validation success -- a configuration failure wearing a product failure's
    clothes.
    """
    record = probe.grade(_observations("malformed_answered_by_auth.json"))
    assert record.verdict == "fail"
    malformed = next(case for case in record.cases if case.name == "malformed")
    assert malformed.ok is False
    assert malformed.observed_status == 401


@pytest.mark.unit
def test_the_record_round_trips_as_json_and_carries_an_as_of() -> None:
    record = probe.grade(_observations("lane_healthy.json"))
    payload = record.to_dict(
        base_url="http://host.docker.internal:8090", as_of="2026-09-22T00:00:00Z"
    )
    assert json.loads(json.dumps(payload)) == payload
    assert payload["version"] == 1
    assert payload["as_of"] == "2026-09-22T00:00:00Z"
    assert len(payload["cases"]) == 4
    assert payload["positive_control"]["pass"] is True


@pytest.mark.unit
def test_no_credential_value_can_reach_the_record() -> None:
    """The record is an artifact; a key in it would be a real exposure."""
    record = probe.grade(_observations("lane_healthy.json"))
    blob = json.dumps(
        record.to_dict(base_url="http://host.docker.internal:8090", as_of="x")
    )
    assert probe.SYNTHETIC_INVALID_KEY not in blob
    assert "x-api-key" not in blob.lower() or "value" not in blob.lower()


@pytest.mark.unit
def test_a_missing_credential_env_name_is_an_input_failure_not_a_verdict() -> None:
    """Exit 2, never exit 1: "I could not run" is not "the platform refused wrongly"."""
    with pytest.raises(probe.ProbeInputError):
        probe.resolve_credential("", environ={})
    with pytest.raises(probe.ProbeInputError):
        probe.resolve_credential("OMN19175_ABSENT_VAR", environ={})
    assert (
        probe.resolve_credential("OMN19175_PRESENT", environ={"OMN19175_PRESENT": "v"})
        == "v"
    )


@pytest.mark.unit
def test_every_arm_targets_a_route_the_runtime_gateway_serves() -> None:
    """OMN-26043: the separate HTTP API is gone, so the arms read the runtime's own routes.

    RULING 2026-11-02T15:02:01Z and RULING 2026-11-10T13:26:22Z moved the tenant
    routes inside the runtime. omnimarket's ``node_workflow_gateway_effect`` serves
    ``POST /v1/workflows``, ``GET /v1/workflows/contracts``,
    ``GET /v1/workflows/{id}/status`` and ``GET /v1/workflows/{id}/receipt`` on
    the lane's ingress port; the retired routes (``/v1/whoami``,
    ``/v1/tenants/<slug>/api-keys``, ``/v1/api-keys``) answered from onex-api.
    """
    served = (
        r"^/v1/workflows/contracts$",
        r"^/v1/workflows/[0-9a-f-]{36}/status$",
        r"^/v1/workflows/[0-9a-f-]{36}/receipt$",
    )
    import re

    for case in probe.CASES:
        assert any(re.fullmatch(pattern, case.path) for pattern in served), case.path
    paths = {case.name: case.path for case in probe.CASES}
    assert paths["wrong_key"] == paths["absent_key"]
    assert paths["malformed"].endswith("/receipt")
    assert probe.MALFORMED_MISSING_FIELD == "runner_identity"


@pytest.mark.unit
def test_the_workflow_resolves_the_lane_by_its_runtime_ingress() -> None:
    """OMN-26117: the probe reads the lane's ingress_url, never the retired gateway port.

    The dev lane's separate HTTP API on port 8090 is gone (RULING 2026-11-02T15:02:01Z,
    RULING 2026-11-10T13:26:22Z), so a lane resolved by ``gateway_url`` read a dead
    address and graded RED before any arm ran.
    """
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "chain-canary-c11-negative-paths.yml"
    ).read_text(encoding="utf-8")
    assert "require: ingress_url" in workflow
    assert "require: gateway_url" not in workflow
    assert "LANE_INGRESS_URL" in workflow
    assert "LANE_GATEWAY_URL" not in workflow
    assert "8090" not in workflow


@pytest.mark.unit
def test_the_probe_keeps_every_arm_status_and_code_the_criterion_names() -> None:
    """OMN-26117: no clause is weakened; the four arms and the control keep their expectations."""
    expected = {
        name: (case.expected_status, case.expected_code)
        for name, case in ((case.name, case) for case in probe.CASES)
    }
    assert expected == {
        "wrong_key": (401, "auth.credential.invalid"),
        "absent_key": (401, "auth.credential.absent"),
        "wrong_tenant": (403, "auth.tenant.forbidden"),
        "malformed": (422, None),
    }
    assert probe.CONTROL.path == "/health"
    assert probe.CONTROL.expected_status == 200
    assert probe.HEADER_NAME == "x-api-key"


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/c11_runtime_ingress_refusals_omn26117.json"
)
def test_recorded_runtime_ingress_answers_grade_as_the_declared_refusals(
    recorded_response: dict[str, Any],
) -> None:
    """OMN-26117: what the dev runtime's own ingress really answered grades PASS per arm.

    The recording is taken from the lane the probe now resolves; the two arms that
    need no credential and the positive control are graded by the probe's own
    ``grade_case`` against its declared expectations.
    """
    recorded = recorded_response["response"]["observations"]
    specs = {case.name: case for case in probe.CASES}
    for name in ("wrong_key", "absent_key"):
        observation = probe.Observation(
            status=recorded[name]["status"], body=recorded[name]["body"]
        )
        result = probe.grade_case(specs[name], observation)
        assert result.ok is True, result.reason
    control = probe.grade_case(
        probe.CONTROL,
        probe.Observation(
            status=recorded["positive_control"]["status"],
            body=recorded["positive_control"]["body"],
        ),
    )
    assert control.ok is True, control.reason
