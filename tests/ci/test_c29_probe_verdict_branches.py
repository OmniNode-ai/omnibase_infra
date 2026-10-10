# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""C29 verdict boundaries with synthetic customer observations (OMN-17427)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c29_customer_byo_key_probe as probe

pytestmark = pytest.mark.unit

AS_OF = "2026-10-03T00:00:00+00:00"
CLAUSES = {"clean_machine", "customer_key", "names_provider", "no_omninode"}


@pytest.fixture
def observations(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake the tracer, leaving the actual clause graders and verdict intact."""
    traces = {
        "quiet": probe.Trace(),
        "provider": probe.Trace(
            connects=[probe.Connect("AF_INET", "203.0.113.10", 443)],
            queried=["api.z.ai"],
            answered={"203.0.113.10": "api.z.ai"},
            answer_names={"api.z.ai"},
        ),
        "control": probe.Trace(queried=[probe.OMNINODE_CONTROL_HOST]),
        "omninode": probe.Trace(
            connects=[probe.Connect("AF_INET", "203.0.113.11", 443)],
            answered={"203.0.113.11": probe.OMNINODE_CONTROL_HOST},
            answer_names={probe.OMNINODE_CONTROL_HOST},
        ),
        "other_named": probe.Trace(
            connects=[probe.Connect("AF_INET", "203.0.113.12", 443)],
            answered={"203.0.113.12": "example.org"},
        ),
        "unattributed": probe.Trace(
            connects=[probe.Connect("AF_INET", "203.0.113.13", 443)]
        ),
    }
    monkeypatch.setattr(probe, "parse_trace", traces.__getitem__)
    receipt = {
        "status": "success",
        "backend_id": "byok-glm",
        "endpoint": "https://api.z.ai/delegate",
        "model": "fake-model",
        "receipt": {
            "runtime_identity": {
                "packages": {
                    dist: {"version": "test", "source": "registry"}
                    for dist in probe.ONEX_DISTRIBUTIONS
                }
            },
            "result": {
                "secret_source": "store",
                "secret_ref": "llm.glm.api_key",
                "response": "fake response",
                "attempts": [
                    {
                        "backend_id": "byok-glm",
                        "model_id": "fake-model",
                        "acceptance_decision": "accept",
                    }
                ],
            },
        },
    }
    return {
        "as_of": AS_OF,
        "provider": "glm",
        "customer_env_keys": sorted(probe.ALLOWED_CUSTOMER_ENV_KEYS),
        "registered_refs": ["llm.glm.api_key"],
        "key_leaks": [],
        "source_trees": [],
        "workdir_git_ancestor": None,
        "steps": {
            "init": {"strace": "quiet"},
            "secret_set": {"returncode": 0, "stdin": "key", "strace": "quiet"},
            "keyless": {
                "returncode": 1,
                "stdout": json.dumps(
                    {"error_message": "[ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT]"}
                ),
                "strace": "quiet",
            },
            "keyed": {"returncode": 0, "strace": "provider"},
            "omninode_control": {"strace": "control"},
        },
        "run_files": {
            "receipt.json": json.dumps(receipt),
            "result.txt": "fake response",
        },
    }


def _assert_failure(
    observations: dict[str, Any], failed: set[str], reason: str
) -> None:
    record = probe.grade(observations)
    assert record.verdict == "FAIL"
    assert {c.name for c in record.clauses if not c.passed} == failed
    assert any(reason in r for c in record.clauses for r in c.reasons)
    payload = record.as_dict()
    assert payload["verdict"] == "FAIL"
    assert set(payload["clauses"]) == CLAUSES


def test_all_clauses_passing_produces_pass(observations: dict[str, Any]) -> None:
    record = probe.grade(observations)
    assert record.verdict == "PASS"
    assert {c.name for c in record.clauses} == CLAUSES
    assert all(c.passed and not c.reasons for c in record.clauses)
    assert record.as_of == AS_OF


@pytest.mark.parametrize(
    ("clause", "field", "value", "reason"),
    [
        ("clean_machine", "source_trees", None, "no source-tree scan"),
        ("clean_machine", "source_trees", ["/fake/checkout"], "source tree(s)"),
        ("clean_machine", "workdir_git_ancestor", "unrecorded", "not recorded"),
        ("clean_machine", "workdir_git_ancestor", "/fake/repo", "git work tree"),
        ("customer_key", "customer_env_keys", None, "keys were not recorded"),
        ("customer_key", "customer_env_keys", ["API_KEY"], "allowed set"),
        ("customer_key", "registered_refs", [], "no key is registered"),
        ("customer_key", "registered_refs", ["llm.gemini.api_key"], "other than"),
        ("customer_key", "key_leaks", None, "not checked for the key"),
        ("customer_key", "key_leaks", ["fake.stderr"], "appeared in captured output"),
    ],
)
def test_observation_failure_isolated_to_one_clause(
    observations: dict[str, Any], clause: str, field: str, value: Any, reason: str
) -> None:
    observations[field] = value
    _assert_failure(observations, {clause}, reason)


@pytest.mark.parametrize(
    ("clause", "path", "value", "reason"),
    [
        ("clean_machine", ("receipt", "runtime_identity"), None, "no runtime_identity"),
        (
            "clean_machine",
            ("receipt", "runtime_identity", "packages", "omnimarket"),
            None,
            "omnimarket absent",
        ),
        (
            "clean_machine",
            ("receipt", "runtime_identity", "packages", "omnimarket", "source"),
            "editable",
            "not the package index",
        ),
        (
            "clean_machine",
            ("receipt", "runtime_identity", "packages", "omnimarket", "commit"),
            "fake-commit",
            "commit or import path",
        ),
        (
            "customer_key",
            ("receipt", "result", "secret_source"),
            "environment",
            "own store",
        ),
        (
            "customer_key",
            ("receipt", "result", "secret_ref"),
            "missing",
            "not a reference",
        ),
        ("names_provider", ("status",), "failed", "receipt status"),
        ("names_provider", ("backend_id",), None, "does not name provider"),
        (
            "names_provider",
            ("endpoint",),
            "https://example.org",
            "not the provider's host",
        ),
        ("names_provider", ("model",), "", "names no model"),
        ("names_provider", ("receipt", "result", "attempts"), [], "no routing attempt"),
        (
            "names_provider",
            ("receipt", "result", "attempts"),
            [{"backend_id": "byok-gemini", "acceptance_decision": "reject"}],
            "left provider",
        ),
        (
            "names_provider",
            ("receipt", "result", "attempts"),
            [
                {
                    "backend_id": "byok-glm",
                    "acceptance_decision": "accept",
                    "model_id": "other",
                }
            ],
            "accepted attempt's model",
        ),
        (
            "names_provider",
            ("receipt", "result", "response"),
            " ",
            "no accepted response",
        ),
        (
            "names_provider",
            ("receipt", "result", "response"),
            "different",
            "not the response",
        ),
    ],
)
def test_receipt_failure_isolated_to_one_clause(
    observations: dict[str, Any],
    clause: str,
    path: tuple[str, ...],
    value: Any,
    reason: str,
) -> None:
    receipt = json.loads(observations["run_files"]["receipt.json"])
    parent = receipt
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    observations["run_files"]["receipt.json"] = json.dumps(receipt)
    _assert_failure(observations, {clause}, reason)


@pytest.mark.parametrize(
    ("step", "field", "value", "clause", "reason"),
    [
        ("secret_set", "returncode", 1, "customer_key", "onex secret set exited"),
        ("secret_set", "stdin", None, "customer_key", "did not reach"),
        ("keyed", "returncode", 1, "names_provider", "keyed delegation exited"),
        ("init", "strace", "omninode", "no_omninode", "omninode external"),
        ("init", "strace", "other_named", "no_omninode", "other_named external"),
        ("init", "strace", "unattributed", "no_omninode", "unattributed external"),
        ("keyed", "strace", "quiet", "no_omninode", "positive control failed"),
        (
            "omninode_control",
            "strace",
            "quiet",
            "no_omninode",
            "positive control failed",
        ),
        ("keyless", "returncode", 0, "no_omninode", "no key registered succeeded"),
        ("keyless", "stdout", "not JSON: fake crash", "no_omninode", "no typed ONEX"),
        ("keyless", "strace", "provider", "no_omninode", "keyless delegation reached"),
    ],
)
def test_step_failure_isolated_to_one_clause(
    observations: dict[str, Any],
    step: str,
    field: str,
    value: Any,
    clause: str,
    reason: str,
) -> None:
    observations["steps"][step][field] = value
    _assert_failure(observations, {clause}, reason)


def _run_args(tmp_path: Path) -> list[str]:
    return [
        "run",
        "--provider",
        "glm",
        "--model",
        "google/gemini-2.5-flash-lite",
        "--key-env",
        "C29_FAKE_KEY",
        "--customer-home",
        str(tmp_path / "home"),
        "--customer-bin",
        str(tmp_path / "bin"),
        "--workdir",
        str(tmp_path / "work"),
        "--trace-dir",
        str(tmp_path / "traces"),
        "--observations-out",
        str(tmp_path / "observations.json"),
        "--record",
        str(tmp_path / "record.json"),
        "--summary",
        str(tmp_path / "summary.md"),
    ]


@pytest.mark.parametrize("failed_clause", [None, *sorted(CLAUSES)])
def test_run_writes_the_verdict_and_matching_exit_code(
    observations: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_clause: str | None,
) -> None:
    if failed_clause == "clean_machine":
        observations["source_trees"] = None
    elif failed_clause == "customer_key":
        observations["key_leaks"] = None
    elif failed_clause == "names_provider":
        observations["run_files"]["result.txt"] = "different"
    elif failed_clause == "no_omninode":
        observations["steps"]["keyless"]["stdout"] = "fake crash"
    monkeypatch.setattr(probe, "observe_live", lambda args: observations)
    code = probe.main(_run_args(tmp_path))
    verdict = "FAIL" if failed_clause else "PASS"
    assert code == (probe.EXIT_FINDINGS if failed_clause else probe.EXIT_OK)
    payload = json.loads((tmp_path / "record.json").read_text())
    assert payload["verdict"] == verdict
    assert {name for name, c in payload["clauses"].items() if not c["passed"]} == (
        {failed_clause} if failed_clause else set()
    )
    assert json.loads((tmp_path / "observations.json").read_text()) == observations
    assert f"— {verdict}" in (tmp_path / "summary.md").read_text()


def test_absent_key_precondition_writes_could_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("C29_FAKE_KEY", raising=False)
    assert probe.main(_run_args(tmp_path)) == probe.EXIT_INPUT
    payload = json.loads((tmp_path / "record.json").read_text())
    assert payload["verdict"] == "COULD_NOT_RUN"
    assert "no provider key in C29_FAKE_KEY" in payload["reason"]
    assert not (tmp_path / "observations.json").exists()
    assert not (tmp_path / "summary.md").exists()


def test_typed_probe_error_writes_could_not_run_instead_of_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def crash(args: argparse.Namespace) -> dict[str, Any]:
        raise probe.ProbeInputError("fake observation failure")

    monkeypatch.setattr(probe, "observe_live", crash)
    assert probe.main(_run_args(tmp_path)) == probe.EXIT_INPUT
    assert json.loads((tmp_path / "record.json").read_text()) == {
        "record_version": probe.RECORD_VERSION,
        "criterion": "C29",
        "verdict": "COULD_NOT_RUN",
        "reason": "fake observation failure",
    }
    captured = capsys.readouterr()
    assert "fake observation failure" in captured.err
    assert "PASS" not in captured.out
    assert not (tmp_path / "observations.json").exists()
    assert not (tmp_path / "summary.md").exists()


@pytest.mark.parametrize("contents", [None, "{invalid json"])
def test_unreadable_replay_writes_could_not_run(
    tmp_path: Path, contents: str | None
) -> None:
    path = tmp_path / "input.json"
    if contents is not None:
        path.write_text(contents)
    record = tmp_path / "record.json"
    assert (
        probe.main(["grade", "--observations", str(path), "--record", str(record)])
        == probe.EXIT_INPUT
    )
    payload = json.loads(record.read_text())
    assert payload["verdict"] == "COULD_NOT_RUN"
    assert payload["reason"].startswith("unreadable observations file:")
