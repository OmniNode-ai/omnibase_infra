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

from scripts.ci import c13_customer_local_probe as probe

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn19179"


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
    obs["steps"]["unconfigured"]["strace"] += (
        "4242 connect(9, {sa_family=AF_INET, sin_port=htons(18731), "
        'sin_addr=inet_addr("127.0.0.1")}, 16) = 0\n'
    )
    assert "zero_provider" in _failed(obs)


@pytest.mark.unit
def test_a_shared_servers_token_delta_alone_does_not_fail_the_negative_control() -> (
    None
):
    """On the lab customer machine the model server is shared, so its counter is not this run's alone.

    The negative control is graded on this process tree's own connects; the
    server-wide counter over the unconfigured run is recorded, not graded.
    """
    obs = _pass_observations()
    obs["model_metrics"]["unconfigured_tokens_predicted_delta"] = 12
    assert "zero_provider" not in _failed(obs)


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


@pytest.mark.unit
def test_the_customer_overlay_carries_the_declared_response_budget() -> None:
    """A reasoning model spends tokens before it answers.

    Measured 2026-09-23 on the lab customer machine: a 512-token budget let the
    served 27B model exhaust the budget before answering (quality 0.0), and
    8192 passed. The budget is an argument, and its default is the one that passed.
    """
    overlay = probe.bifrost_overlay_yaml("served-model", 8000, 8192)
    assert overlay.count("    max_tokens: 8192\n") == len(probe.LOCAL_BACKEND_IDS)
    assert overlay.count('    model_name: "served-model"\n') == len(
        probe.LOCAL_BACKEND_IDS
    )
    assert "http://127.0.0.1:8000/v1/chat/completions" in overlay


@pytest.mark.unit
def test_the_overlay_declares_every_shipped_local_backend() -> None:
    """The shipped routing sends prose classes to local-heavy-reasoning.

    Declaring only local-coder would leave the document class with no model
    and make the probe red for a configuration reason, not a C13 reason.
    """
    overlay = probe.bifrost_overlay_yaml("m", 8000, 8192)
    for backend_id in ("local-coder", "local-heavy-reasoning"):
        assert f"  - backend_id: {backend_id}\n" in overlay


@pytest.mark.unit
def test_the_customer_writes_the_documented_file_and_nothing_else() -> None:
    """OMN-16200: one documented file under HOME, no environment binding."""
    assert probe.OVERLAY_RELATIVE_PATH == ".omninode/delegation/bifrost_overrides.yaml"


# --------------------------------------------------------------------------
# a model the customer serves elsewhere in their own private network (OMN-19805)
# --------------------------------------------------------------------------

_PRIVATE_MODEL_HOST = "10.20.30.40"


def _served_from_private_host(obs: dict[str, Any]) -> dict[str, Any]:
    """The recorded session, as if its model answered from a private LAN host."""
    port = obs["model_port"]
    loopback = f'sin_port=htons({port}), sin_addr=inet_addr("127.0.0.1")'
    private = f'sin_port=htons({port}), sin_addr=inet_addr("{_PRIVATE_MODEL_HOST}")'
    for step in obs["steps"].values():
        if isinstance(step, dict) and step.get("strace"):
            step["strace"] = step["strace"].replace(loopback, private)
    receipt = _receipt(obs)
    receipt["endpoint"] = f"http://{_PRIVATE_MODEL_HOST}:{port}/v1/chat/completions"
    _set_receipt(obs, receipt)
    obs["model_host"] = _PRIVATE_MODEL_HOST
    return obs


@pytest.mark.unit
def test_a_session_served_by_the_declared_private_host_grades_pass() -> None:
    obs = _served_from_private_host(_pass_observations())
    assert _failed(obs) == {}
    connects = probe.parse_connects(
        obs["steps"]["configured"]["strace"],
        model_port=obs["model_port"],
        model_host=_PRIVATE_MODEL_HOST,
    )
    assert probe.count_kinds(connects).get("model", 0) >= 1


@pytest.mark.unit
def test_a_private_host_connect_the_session_did_not_declare_is_external() -> None:
    """The same strace, graded with no declared host, is a set of external calls."""
    obs = _served_from_private_host(_pass_observations())
    del obs["model_host"]
    failed = _failed(obs)
    assert "zero_provider" in failed
    assert "local_model" in failed


@pytest.mark.unit
def test_only_the_declared_host_and_port_is_the_model() -> None:
    connects = probe.parse_connects(
        '1 connect(3, {sa_family=AF_INET, sin_port=htons(8000), sin_addr=inet_addr("10.20.30.40")}, 16) = 0\n'
        '1 connect(4, {sa_family=AF_INET, sin_port=htons(8001), sin_addr=inet_addr("10.20.30.40")}, 16) = 0\n'
        '1 connect(5, {sa_family=AF_INET, sin_port=htons(8000), sin_addr=inet_addr("10.20.30.41")}, 16) = 0\n'
        '1 connect(6, {sa_family=AF_INET, sin_port=htons(8000), sin_addr=inet_addr("127.0.0.1")}, 16) = 0\n',
        model_port=8000,
        model_host=_PRIVATE_MODEL_HOST,
    )
    assert [c.kind for c in connects] == [
        "model",
        "external",
        "external",
        "loopback_other",
    ]


@pytest.mark.unit
def test_a_receipt_endpoint_on_another_private_host_fails() -> None:
    obs = _served_from_private_host(_pass_observations())
    receipt = _receipt(obs)
    receipt["endpoint"] = f"http://10.20.30.41:{obs['model_port']}/v1/chat/completions"
    _set_receipt(obs, receipt)
    assert "local_model" in _failed(obs)


@pytest.mark.unit
@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "10.20.30.40", "172.16.0.9"])
def test_loopback_and_private_model_hosts_are_accepted(host: str) -> None:
    assert probe.validate_model_host(host)


@pytest.mark.unit
@pytest.mark.parametrize("host", ["1.1.1.1", "8.8.8.8", "model.example.com", ""])
def test_a_public_or_named_model_host_is_refused_before_anything_runs(
    host: str,
) -> None:
    with pytest.raises(probe.ProbeInputError):
        probe.validate_model_host(host)


@pytest.mark.unit
def test_a_public_model_host_exits_could_not_run(tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    argv = [
        "run",
        "--customer-home",
        str(tmp_path / "home"),
        "--customer-bin",
        str(tmp_path / "bin"),
        "--workdir",
        str(tmp_path / "work"),
        "--trace-dir",
        str(tmp_path / "traces"),
        "--model-port",
        "8000",
        "--model-host",
        "8.8.8.8",
        "--served-model",
        "m",
        "--observations-out",
        str(tmp_path / "obs.json"),
        "--record",
        str(record),
    ]
    assert probe.main(argv) == 2
    assert "public internet" in json.loads(record.read_text())["reason"]


@pytest.mark.unit
def test_the_overlay_points_at_the_declared_private_host() -> None:
    overlay = probe.bifrost_overlay_yaml("m", 8000, 8192, _PRIVATE_MODEL_HOST)
    assert overlay.count(
        f'    endpoint_url: "http://{_PRIVATE_MODEL_HOST}:8000/v1/chat/completions"\n'
    ) == len(probe.LOCAL_BACKEND_IDS)


@pytest.mark.unit
def test_an_ipv6_model_host_is_bracketed_in_the_url() -> None:
    assert probe.model_base_url("::1", 8000) == "http://[::1]:8000"


@pytest.mark.unit
def test_the_vllm_token_counter_is_summed_across_label_sets() -> None:
    text = (
        "# HELP vllm:generation_tokens_total Number of generation tokens processed.\n"
        'vllm:generation_tokens_total{engine="0",model_name="m"} 1.5e+03\n'
        'vllm:generation_tokens_total{engine="1",model_name="m"} 20.0\n'
        'vllm:generation_tokens_total_created{engine="0",model_name="m"} 9e+09\n'
    )
    assert probe.tokens_predicted_from_metrics(text) == 1520


@pytest.mark.unit
def test_the_llamacpp_token_counter_still_reads() -> None:
    assert (
        probe.tokens_predicted_from_metrics("llamacpp:tokens_predicted_total 42\n")
        == 42
    )


@pytest.mark.unit
def test_a_server_with_no_token_counter_cannot_prove_the_positive_control() -> None:
    with pytest.raises(probe.ProbeInputError):
        probe.tokens_predicted_from_metrics("process_cpu_seconds_total 1\n")
