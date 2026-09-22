# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C13 customer-local producer (OMN-19179).

Every test grades RECORDED observations under ``tests/fixtures/omn19179/`` and
performs no network I/O, so each failure branch is falsifiable here rather than
only on a hosted runner.

``observations_lab_pass.json`` is a real session, not a synthetic one: a clean
``ubuntu:24.04`` container with no OmniNode clone, the published packages
installed by pipx, a pinned llama.cpp build serving Qwen2.5-3B-Instruct on
loopback, and ``onex delegate`` traced by ``strace -f -e trace=connect``.
Each negative test below starts from that record and breaks exactly one fact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import c13_customer_local_probe as probe

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn19179"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "c13-customer-local-delegation.yml"
PRODUCER_STEP = "Drive one customer delegation and grade it"


def _pass_observations() -> dict[str, Any]:
    payload = json.loads(
        (FIXTURES / "observations_lab_pass.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _receipt(obs: dict[str, Any]) -> dict[str, Any]:
    receipt = json.loads(obs["run_files"]["receipt.json"])
    assert isinstance(receipt, dict)
    return receipt


def _set_receipt(obs: dict[str, Any], receipt: dict[str, Any]) -> None:
    obs["run_files"]["receipt.json"] = json.dumps(receipt)


def _failed(obs: dict[str, Any]) -> dict[str, list[str]]:
    record = probe.grade(obs, as_of="2026-09-22T00:00:00Z")
    return {c.name: c.reasons for c in record.clauses if not c.passed}


# --------------------------------------------------------------------------
# the recorded session passes, and passes for the right reasons
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_the_recorded_clean_container_session_grades_pass() -> None:
    record = probe.grade(_pass_observations(), as_of="2026-09-22T00:00:00Z")
    assert record.verdict == "PASS", [c.as_dict() for c in record.clauses]
    assert [c.name for c in record.clauses] == [
        "no_checkout",
        "three_files",
        "local_model",
        "zero_provider",
    ]


@pytest.mark.unit
def test_the_recorded_session_carries_both_positive_controls() -> None:
    """The pass is not an empty log: the model was seen, and so was the control."""
    obs = _pass_observations()
    configured = probe.count_kinds(
        probe.parse_connects(
            obs["steps"]["configured"]["strace"], model_port=obs["model_port"]
        )
    )
    control = probe.count_kinds(
        probe.parse_connects(
            obs["steps"]["outbound_control"]["strace"], model_port=obs["model_port"]
        )
    )
    assert configured.get("model", 0) >= 1
    assert configured.get("external", 0) == 0
    assert control.get("external", 0) >= 1
    assert obs["model_metrics"]["configured_tokens_predicted_delta"] >= 1
    assert obs["model_metrics"]["unconfigured_tokens_predicted_delta"] == 0


# --------------------------------------------------------------------------
# zero_provider: an unproven zero fails (ticket AC4)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_an_external_connect_in_the_configured_run_fails() -> None:
    obs = _pass_observations()
    obs["steps"]["configured"]["strace"] += (
        "4242 connect(9, {sa_family=AF_INET, sin_port=htons(443), "
        'sin_addr=inet_addr("34.117.59.81")}, 16) = -1 EINPROGRESS\n'
    )
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_an_external_connect_during_init_also_fails() -> None:
    obs = _pass_observations()
    obs["steps"]["init"]["strace"] += (
        "77 connect(3, {sa_family=AF_INET6, sin6_port=htons(443), sin6_flowinfo=htonl(0), "
        'inet_pton(AF_INET6, "2606:4700::6810:84e5", &sin6_addr), sin6_scope_id=0}, 28) = 0\n'
    )
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_a_name_lookup_fails_even_without_a_connect_behind_it() -> None:
    obs = _pass_observations()
    obs["steps"]["configured"]["strace"] += (
        "4242 connect(9, {sa_family=AF_INET, sin_port=htons(53), "
        'sin_addr=inet_addr("127.0.0.53")}, 16) = 0\n'
    )
    reasons = _failed(obs)["zero_provider"]
    assert any("name lookup" in r for r in reasons)


@pytest.mark.unit
def test_a_blind_tracer_is_an_unproven_zero_not_a_pass() -> None:
    """No instrumentation at all must not grade PASS."""
    obs = _pass_observations()
    for step in obs["steps"].values():
        step["strace"] = ""
    reasons = _failed(obs)["zero_provider"]
    assert any("positive control failed" in r for r in reasons)


@pytest.mark.unit
def test_a_model_connect_without_the_server_counting_tokens_fails() -> None:
    obs = _pass_observations()
    obs["model_metrics"]["configured_tokens_predicted_delta"] = 0
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_a_parser_that_cannot_see_the_outbound_control_fails() -> None:
    obs = _pass_observations()
    obs["steps"]["outbound_control"]["strace"] = ""
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_a_missing_outbound_control_fails() -> None:
    obs = _pass_observations()
    del obs["steps"]["outbound_control"]
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_an_unconfigured_run_that_succeeds_fails_the_negative_control() -> None:
    obs = _pass_observations()
    obs["steps"]["unconfigured"]["returncode"] = 0
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_an_unconfigured_run_that_reaches_the_model_fails_the_negative_control() -> (
    None
):
    obs = _pass_observations()
    obs["model_metrics"]["unconfigured_tokens_predicted_delta"] = 12
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_an_inet_connect_with_an_unreadable_address_is_external() -> None:
    connects = probe.parse_connects(
        "1 connect(3, {sa_family=AF_INET, sin_port=htons(443), ...}, 16) = 0\n",
        model_port=18731,
    )
    assert [c.kind for c in connects] == ["external"]


@pytest.mark.unit
def test_loopback_on_another_port_is_not_the_model() -> None:
    connects = probe.parse_connects(
        '1 connect(3, {sa_family=AF_INET, sin_port=htons(5432), sin_addr=inet_addr("127.0.0.1")}, 16) = 0\n'
        '1 connect(4, {sa_family=AF_INET, sin_port=htons(18731), sin_addr=inet_addr("127.0.0.1")}, 16) = 0\n'
        "1 <... connect resumed>) = 0\n",
        model_port=18731,
    )
    assert [c.kind for c in connects] == ["loopback_other", "model"]


# --------------------------------------------------------------------------
# no_checkout
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_a_package_imported_from_a_checkout_fails() -> None:
    obs = _pass_observations()
    receipt = _receipt(obs)
    packages = receipt["receipt"]["runtime_identity"]["packages"]
    packages["omnimarket"]["source"] = "workspace"
    packages["omnimarket"]["commit"] = "0" * 40
    _set_receipt(obs, receipt)
    assert "no_checkout" in _failed(obs)


@pytest.mark.unit
def test_a_vcs_or_path_install_fails() -> None:
    obs = _pass_observations()
    obs["direct_url"]["omnibase_infra"] = (
        '{"url": "file:///src/omnibase_infra", "dir_info": {"editable": true}}'
    )
    assert "no_checkout" in _failed(obs)


@pytest.mark.unit
def test_a_source_tree_on_the_machine_fails() -> None:
    obs = _pass_observations()
    obs["source_trees"] = ["/home/runner/work/omnibase_infra/omnibase_infra"]
    assert "no_checkout" in _failed(obs)


@pytest.mark.unit
def test_an_unrecorded_scan_is_not_a_clean_machine() -> None:
    obs = _pass_observations()
    obs["source_trees"] = None
    assert "no_checkout" in _failed(obs)


@pytest.mark.unit
def test_the_source_tree_scan_finds_a_pyproject_by_distribution_name(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "somewhere" / "omnimarket"
    tree.mkdir(parents=True)
    (tree / "pyproject.toml").write_text('[project]\nname = "omnimarket"\n')
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    (unrelated / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')
    assert probe.find_source_trees([tmp_path]) == [str(tree)]


# --------------------------------------------------------------------------
# three_files and local_model
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", probe.RUN_FILES)
def test_each_of_the_three_files_is_required(name: str) -> None:
    obs = _pass_observations()
    obs["run_files"][name] = None
    assert "three_files" in _failed(obs)


@pytest.mark.unit
def test_result_text_must_be_the_accepted_response() -> None:
    obs = _pass_observations()
    obs["run_files"]["result.txt"] = "a different answer"
    assert "three_files" in _failed(obs)


@pytest.mark.unit
def test_a_non_zero_cli_exit_fails_even_with_files_present() -> None:
    obs = _pass_observations()
    obs["steps"]["configured"]["returncode"] = 1
    assert "three_files" in _failed(obs)


@pytest.mark.unit
def test_a_model_the_server_does_not_serve_fails() -> None:
    obs = _pass_observations()
    receipt = _receipt(obs)
    receipt["model"] = "Qwen3.8-27B"
    _set_receipt(obs, receipt)
    assert "local_model" in _failed(obs)


@pytest.mark.unit
def test_a_non_loopback_endpoint_fails() -> None:
    obs = _pass_observations()
    receipt = _receipt(obs)
    receipt["endpoint"] = "http://10.0.0.5:8000/v1/chat/completions"
    _set_receipt(obs, receipt)
    assert "local_model" in _failed(obs)


@pytest.mark.unit
def test_an_attempt_above_the_local_tier_fails() -> None:
    obs = _pass_observations()
    receipt = _receipt(obs)
    attempts = receipt["receipt"]["result"]["attempts"]
    climbed = copy.deepcopy(attempts[0])
    climbed.update({"tier": "cheap_cloud", "acceptance_decision": "climb"})
    attempts.insert(0, climbed)
    _set_receipt(obs, receipt)
    assert "local_model" in _failed(obs)


@pytest.mark.unit
def test_a_non_local_routing_tier_fails() -> None:
    obs = _pass_observations()
    receipt = _receipt(obs)
    receipt["routing_tier"] = "cheap_cloud"
    _set_receipt(obs, receipt)
    assert "local_model" in _failed(obs)


# --------------------------------------------------------------------------
# exit codes and the workflow shape
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_exit_codes_distinguish_pass_fail_and_could_not_run(tmp_path: Path) -> None:
    passing = tmp_path / "pass.json"
    passing.write_text(json.dumps(_pass_observations()))
    failing_obs = _pass_observations()
    failing_obs["model_metrics"]["configured_tokens_predicted_delta"] = 0
    failing = tmp_path / "fail.json"
    failing.write_text(json.dumps(failing_obs))
    record = tmp_path / "record.json"

    assert (
        probe.main(["grade", "--observations", str(passing), "--record", str(record)])
        == 0
    )
    assert json.loads(record.read_text())["verdict"] == "PASS"
    assert (
        probe.main(["grade", "--observations", str(failing), "--record", str(record)])
        == 1
    )
    assert json.loads(record.read_text())["verdict"] == "FAIL"
    assert (
        probe.main(
            [
                "grade",
                "--observations",
                str(tmp_path / "absent.json"),
                "--record",
                str(record),
            ]
        )
        == 2
    )
    assert json.loads(record.read_text())["verdict"] == "COULD_NOT_RUN"


def _workflow() -> dict[Any, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.unit
def test_the_workflow_never_checks_out_a_repository() -> None:
    """A checkout of this repository IS a checkout the clause forbids."""
    steps = _workflow()["jobs"]["c13-customer-local"]["steps"]
    uses = [str(step.get("uses", "")) for step in steps]
    assert not any(u.startswith("actions/checkout") for u in uses)
    fetch = next(s for s in steps if s["name"].startswith("Fetch the probe scripts"))
    assert "raw.githubusercontent.com/${REPO}/${SHA}/" in fetch["run"]
    assert fetch["env"]["SHA"] == "${{ github.sha }}"


@pytest.mark.unit
def test_the_workflow_runs_on_a_hosted_image_not_a_lab_runner() -> None:
    runs_on = _workflow()["jobs"]["c13-customer-local"]["runs-on"]
    assert isinstance(runs_on, str)
    assert runs_on.startswith("ubuntu-")


@pytest.mark.unit
def test_the_producer_step_exit_code_is_the_verdict() -> None:
    job = _workflow()["jobs"]["c13-customer-local"]
    assert "continue-on-error" not in job
    step = next(s for s in job["steps"] if s["name"] == PRODUCER_STEP)
    assert "continue-on-error" not in step
    assert "if" not in step
    assert "|| true" not in step["run"]
    assert 'exit "${status}"' in step["run"]


@pytest.mark.unit
def test_the_workflow_has_a_schedule_and_no_pull_request_trigger() -> None:
    triggers = _workflow()[True]  # PyYAML reads the bare key `on` as True.
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers


@pytest.mark.unit
def test_the_model_and_server_are_verified_before_use() -> None:
    workflow = _workflow()
    env = workflow["env"]
    assert len(env["MODEL_SHA256"]) == 64
    assert len(env["LLAMA_SHA256"]) == 64
    fetch = next(
        s
        for s in workflow["jobs"]["c13-customer-local"]["steps"]
        if s["name"] == "Fetch and verify the pinned model and server build"
    )
    assert fetch["run"].count("sha256sum -c -") == 2
