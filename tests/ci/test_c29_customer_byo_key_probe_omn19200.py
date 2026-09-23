# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline grading tests for the C29 customer-own-key producer (OMN-19200).

Every test here runs against RECORDED observations under
``tests/fixtures/omn19200/`` and performs no network I/O. Both fixtures are
real sessions taken on a clean container on 2026-09-22 (their provenance is in
each file): one where the key was registered the way the product says and the
delegation was refused, and one where a hand-minted reference let it reach the
provider. The failure branches a live run cannot produce on demand -- an
OmniNode lookup, an address nobody looked up, a blind tracer, a leaked key --
are derived from the working one, so each is falsifiable here and not only on a
hosted runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c29_customer_byo_key_probe as probe

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn19200"


def _load(name: str) -> dict[str, Any]:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _working() -> dict[str, Any]:
    return _load("observations_working_path.json")


def _reasons(record: probe.Record, clause: str) -> list[str]:
    return next(c for c in record.clauses if c.name == clause).reasons


def _dns_query_line(pid: int, name: str) -> str:
    """A strace sendmmsg line carrying one A query for ``name``, as -xx prints it."""
    labels = b"".join(bytes([len(p)]) + p.encode() for p in name.split("."))
    packet = (
        b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
        + labels
        + b"\x00\x00\x01\x00\x01"
    )
    escaped = "".join(f"\\x{b:02x}" for b in packet)
    return (
        f'{pid}   sendmmsg(12, [{{msg_hdr={{msg_name=NULL, msg_namelen=0, msg_iov=[{{iov_base="{escaped}", '
        f"iov_len={len(packet)}}}], msg_iovlen=1, msg_controllen=0, msg_flags=0}}, msg_len={len(packet)}}}], 1, MSG_NOSIGNAL) = 1"
    )


def _connect_line(pid: int, host: str, port: int) -> str:
    escaped = "".join(f"\\x{ord(c):02x}" for c in host)
    return (
        f"{pid}   connect(13, {{sa_family=AF_INET, sin_port=htons({port}), "
        f'sin_addr=inet_addr("{escaped}")}}, 16) = 0'
    )


@pytest.mark.unit
def test_a_session_that_reached_the_provider_on_a_stored_key_grades_pass() -> None:
    """The probe can pass on a healthy system: a real working session is PASS."""
    record = probe.grade(_working())
    assert record.verdict == "PASS", {c.name: c.reasons for c in record.clauses}
    names = next(c for c in record.clauses if c.name == "names_provider").evidence
    assert names["receipt_backend_id"] == "byok-glm"
    assert names["receipt_model"] == "glm-5.3-flash"
    steps = next(c for c in record.clauses if c.name == "no_omninode").evidence["steps"]
    assert steps["keyed"]["queried"] == ["api.z.ai"]
    assert steps["keyed"]["connect_kinds"]["provider"] >= 1


@pytest.mark.unit
def test_the_documented_registration_is_refused_today_and_grades_fail() -> None:
    """The real verdict on the published packages: the product's own instruction does not work.

    OMN-19205 owns the defect. When it lands, this fixture stays as the record of
    the day the criterion was red for it; the live producer turns green, not this.
    """
    record = probe.grade(_load("observations_documented_surface_refused.json"))
    assert record.verdict == "FAIL"
    names = _reasons(record, "names_provider")
    # The provider-only customer: the OMN-16200 gate words OMN-19205's refusal.
    assert any("No local model is declared" in r for r in names)
    # The customer wrote no configuration at all.
    evidence = next(c for c in record.clauses if c.name == "customer_key").evidence
    assert set(evidence["customer_env_keys"]) == {"HOME", "PATH", "LANG", "TERM"}
    # Refused before any provider was reached, so the zero is unproven, not green.
    assert any("positive control failed" in r for r in _reasons(record, "no_omninode"))
    # The instrument itself was fine: the OmniNode control was decoded.
    evidence = next(c for c in record.clauses if c.name == "no_omninode").evidence
    assert evidence["omninode_control_seen"] == [probe.OMNINODE_CONTROL_HOST]


@pytest.mark.unit
def test_an_omninode_lookup_anywhere_in_the_session_fails() -> None:
    obs = _working()
    obs["steps"]["keyed"]["strace"] += (
        "\n" + _dns_query_line(301, "api.omninode.ai") + "\n"
    )
    record = probe.grade(obs)
    assert record.verdict == "FAIL"
    assert any(
        "OmniNode name(s) looked up" in r for r in _reasons(record, "no_omninode")
    )


@pytest.mark.unit
def test_an_address_nobody_looked_up_fails() -> None:
    """An unattributable connect could be anyone's, ours included."""
    obs = _working()
    obs["steps"]["init"]["strace"] += (
        "\n" + _connect_line(302, "203.0.113.9", 443) + "\n"
    )
    record = probe.grade(obs)
    assert record.verdict == "FAIL"
    assert any("unattributed" in r for r in _reasons(record, "no_omninode"))


@pytest.mark.unit
def test_a_blind_tracer_cannot_produce_a_pass() -> None:
    """Positive control 1: no provider connect seen means the zero is unproven."""
    obs = _working()
    obs["steps"]["keyed"]["strace"] = ""
    record = probe.grade(obs)
    assert record.verdict == "FAIL"
    assert any("positive control failed" in r for r in _reasons(record, "no_omninode"))


@pytest.mark.unit
def test_a_decoder_that_cannot_see_an_omninode_lookup_cannot_produce_a_pass() -> None:
    """Positive control 2: the OmniNode control must be decoded and classified."""
    obs = _working()
    obs["steps"]["omninode_control"]["strace"] = ""
    record = probe.grade(obs)
    assert record.verdict == "FAIL"
    assert any(
        probe.OMNINODE_CONTROL_HOST in r for r in _reasons(record, "no_omninode")
    )
    del obs["steps"]["omninode_control"]
    assert probe.grade(obs).verdict == "FAIL"


@pytest.mark.unit
def test_a_keyless_delegation_that_succeeds_fails_the_negative_control() -> None:
    obs = _working()
    obs["steps"]["keyless"]["returncode"] = 0
    obs["steps"]["keyless"]["stdout"] = "{}"
    obs["steps"]["keyless"]["stderr"] = ""
    record = probe.grade(obs)
    reasons = _reasons(record, "no_omninode")
    assert record.verdict == "FAIL"
    assert any("negative control failed" in r and "succeeded" in r for r in reasons)
    assert any("no typed ONEX refusal code" in r for r in reasons)


@pytest.mark.unit
def test_the_keyless_refusal_code_is_recorded() -> None:
    evidence = next(
        c for c in probe.grade(_working()).clauses if c.name == "no_omninode"
    ).evidence
    assert evidence["keyless_typed_refusal"] is True
    assert evidence["keyless_refusal_code"] == "ONEX_CORE_041_INVALID_CONFIGURATION"


@pytest.mark.unit
@pytest.mark.parametrize("step", probe.SESSION_STEPS)
def test_an_untraced_customer_step_fails(step: str) -> None:
    obs = _working()
    del obs["steps"][step]
    assert probe.grade(obs).verdict == "FAIL"


@pytest.mark.unit
def test_a_key_variable_in_the_customer_environment_fails() -> None:
    obs = _working()
    obs["customer_env_keys"] = [*obs["customer_env_keys"], "OPENROUTER_API_KEY"]
    record = probe.grade(obs)
    assert record.verdict == "FAIL"
    assert any("OPENROUTER_API_KEY" in r for r in _reasons(record, "customer_key"))


@pytest.mark.unit
def test_a_key_not_answered_by_the_local_store_fails() -> None:
    obs = _working()
    receipt = json.loads(obs["run_files"]["receipt.json"])
    receipt["receipt"]["result"]["secret_source"] = "environment"
    obs["run_files"]["receipt.json"] = json.dumps(receipt)
    record = probe.grade(obs)
    assert record.verdict == "FAIL"
    assert any("environment" in r for r in _reasons(record, "customer_key"))


@pytest.mark.unit
def test_a_second_provider_registered_fails_one_provider_key() -> None:
    obs = _working()
    obs["registered_refs"] = [*obs["registered_refs"], "llm.openrouter.api_key"]
    record = probe.grade(obs)
    assert record.verdict == "FAIL"
    assert any("other than 'glm'" in r for r in _reasons(record, "customer_key"))


@pytest.mark.unit
def test_a_leaked_key_fails_and_an_unchecked_one_is_not_a_pass() -> None:
    obs = _working()
    obs["key_leaks"] = ["observations.steps.keyed.stderr"]
    assert probe.grade(obs).verdict == "FAIL"
    del obs["key_leaks"]
    assert probe.grade(obs).verdict == "FAIL"


@pytest.mark.unit
def test_a_receipt_that_does_not_name_the_provider_fails() -> None:
    obs = _working()
    receipt = json.loads(obs["run_files"]["receipt.json"])
    receipt["backend_id"] = "cloud-gemini-flash"
    receipt["endpoint"] = (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    receipt["model"] = ""
    obs["run_files"]["receipt.json"] = json.dumps(receipt)
    reasons = _reasons(probe.grade(obs), "names_provider")
    assert any("does not name provider" in r for r in reasons)
    assert any("not the provider's host" in r for r in reasons)
    assert any("names no model" in r for r in reasons)


@pytest.mark.unit
def test_a_checkout_installed_package_fails_clean_machine() -> None:
    obs = _working()
    receipt = json.loads(obs["run_files"]["receipt.json"])
    receipt["receipt"]["runtime_identity"]["packages"]["omnimarket"]["source"] = (
        "editable"
    )
    obs["run_files"]["receipt.json"] = json.dumps(receipt)
    assert _reasons(probe.grade(obs), "clean_machine")


@pytest.mark.unit
def test_the_dns_decoder_reads_an_answer_and_attributes_its_addresses() -> None:
    answer = bytes.fromhex(
        "abcd81800001000100000000"  # header: response, 1 question, 1 answer
        "03617069017a02616900"  # api.z.ai
        "00010001"  # A, IN
        "c00c000100010000003c0004"  # pointer to qname, A, IN, ttl, rdlength 4
        "cb00710a"  # 203.0.113.10
    )
    message = probe.parse_dns(answer)
    assert message is not None and message.is_response
    assert message.qname == "api.z.ai"
    assert message.addresses == (("api.z.ai", "203.0.113.10"),)
    escaped = "".join(f"\\x{b:02x}" for b in answer)
    trace = probe.parse_trace(
        f'77    <... recvfrom resumed>"{escaped}", 2048, 0, NULL, NULL) = {len(answer)}\n'
        + _connect_line(77, "203.0.113.10", 443)
    )
    rows = probe.classify_connects(trace, provider_host="api.z.ai", nameservers=[])
    assert [r["kind"] for r in rows] == ["provider"]


@pytest.mark.unit
def test_a_tls_record_is_never_read_as_dns() -> None:
    assert (
        probe.parse_dns(b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03" + b"\x00" * 40)
        is None
    )


@pytest.mark.unit
def test_omninode_names_are_classified_by_domain_not_by_prefix() -> None:
    assert probe.is_omninode("api.omninode.ai")
    assert probe.is_omninode("OMNINODE.AI.")
    assert not probe.is_omninode("notomninode.ai")
    assert not probe.is_omninode("api.z.ai")


@pytest.mark.unit
def test_the_registration_is_the_one_the_product_prints() -> None:
    """The probe registers the key exactly as `onex secret` documents, never another way."""
    for slug, spec in probe.PROVIDERS.items():
        assert spec.registration_ref == f"llm.{slug}.api_key"


@pytest.mark.unit
def test_a_run_with_no_key_supplied_is_an_input_failure_not_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("C29_TEST_KEY", raising=False)
    record = tmp_path / "record.json"
    code = probe.main(
        [
            "run",
            "--provider",
            "glm",
            "--key-env",
            "C29_TEST_KEY",
            "--customer-home",
            str(tmp_path / "home"),
            "--customer-bin",
            str(tmp_path / "bin"),
            "--workdir",
            str(tmp_path / "work"),
            "--trace-dir",
            str(tmp_path / "traces"),
            "--observations-out",
            str(tmp_path / "obs.json"),
            "--record",
            str(record),
        ]
    )
    assert code == probe.EXIT_INPUT
    assert json.loads(record.read_text())["verdict"] == "COULD_NOT_RUN"


@pytest.mark.unit
def test_the_replay_exit_code_is_the_verdict(tmp_path: Path) -> None:
    for name, expected in (
        ("observations_working_path.json", probe.EXIT_OK),
        ("observations_documented_surface_refused.json", probe.EXIT_FINDINGS),
    ):
        record = tmp_path / f"{name}.record.json"
        code = probe.main(
            ["grade", "--observations", str(FIXTURES / name), "--record", str(record)]
        )
        assert code == expected
        payload = json.loads(record.read_text())
        assert payload["criterion"] == "C29" and payload["as_of"]


@pytest.mark.unit
def test_redaction_replaces_the_key_and_records_where() -> None:
    leaks: list[str] = []
    out = probe.redact_key(
        {"a": ["x sk-test-123 y"], "b": "clean"}, "sk-test-123", "obs", leaks
    )
    assert out == {"a": [f"x {probe.REDACTED} y"], "b": "clean"}
    assert leaks == ["obs.a[0]"]


@pytest.mark.unit
def test_the_hosted_workflow_is_retired() -> None:
    """The job runs on the omnipc2 customer machine, from omninode_infra.

    Operator direction 2026-09-22: CI runs on our fleet, never GitHub-hosted,
    and the no-checkout machine is omnipc2, whose runner group admits only
    omninode_infra. A second copy of the job here would be a second verdict.
    """
    assert not (
        REPO_ROOT / ".github" / "workflows" / "c29-customer-byo-key-delegation.yml"
    ).exists()
