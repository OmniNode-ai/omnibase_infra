# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Grade failed C29 receipts at their terminal payload layer (OMN-17427)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from scripts.ci import c29_customer_byo_key_probe as probe

pytestmark = pytest.mark.unit


@pytest.fixture
def observations(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake traces and a successful receipt, leaving the actual graders intact."""
    traces = {
        "quiet": probe.Trace(),
        "provider": probe.Trace(
            connects=[probe.Connect("AF_INET", "203.0.113.10", 443)],
            queried=["openrouter.ai"],
            answered={"203.0.113.10": "openrouter.ai"},
            answer_names={"openrouter.ai"},
        ),
        "control": probe.Trace(queried=[probe.OMNINODE_CONTROL_HOST]),
    }
    monkeypatch.setattr(probe, "parse_trace", traces.__getitem__)
    receipt = {
        "status": "success",
        "backend_id": "byok-openrouter",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "synthetic-model",
        "receipt": {
            "runtime_identity": {
                "packages": {
                    dist: {"version": "test", "source": "registry"}
                    for dist in probe.ONEX_DISTRIBUTIONS
                }
            },
            "result": {
                "secret_source": "store",
                "secret_ref": "llm.openrouter.api_key",
                "response": "fake response",
                "attempts": [
                    {
                        "tier": "cheap_frontier",
                        "backend_id": "byok-openrouter",
                        "model_id": "synthetic-model",
                        "acceptance_decision": "accept",
                    }
                ],
            },
        },
    }
    return {
        "provider": "openrouter",
        "model": "synthetic-model",
        "customer_env_keys": sorted(probe.ALLOWED_CUSTOMER_ENV_KEYS),
        "registered_refs": ["llm.openrouter.api_key"],
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


def _failed_receipt(
    observations: dict[str, Any],
    failures: list[str],
    *,
    source: str = "store",
    top_level_only: bool = False,
) -> None:
    receipt = json.loads(observations["run_files"]["receipt.json"])
    payload = receipt["receipt"]["result"]
    payload["secret_source"] = source
    payload.pop("response")
    attempts = [
        {
            "tier": "cheap_frontier",
            "backend_id": "byok-openrouter",
            "model_id": "synthetic-model",
            "failure_class": failure,
        }
        for failure in failures
    ]
    attempts[-1]["acceptance_decision"] = None
    payload["attempts"] = attempts
    if top_level_only:
        payload.pop("attempts")
    receipt["attempts"] = attempts
    receipt["receipt"]["result"] = {"terminal_payload": payload}
    receipt["status"] = "failed"
    observations["steps"]["keyed"]["returncode"] = 1
    observations["run_files"] = {"receipt.json": json.dumps(receipt)}


@pytest.mark.parametrize("top_level_only", [False, True])
def test_provider_throttle_keeps_failed_verdict_and_grades_stored_key(
    observations: dict[str, Any], top_level_only: bool
) -> None:
    _failed_receipt(observations, ["rate_limited"] * 3, top_level_only=top_level_only)

    record = probe.grade(observations)
    clauses = {c.name: c for c in record.clauses}
    assert record.verdict == "FAIL"
    assert clauses["customer_key"].passed
    names = clauses["names_provider"]
    assert not names.passed
    assert names.reasons[0].startswith(
        "provider throttled: all 3 attempts on byok-openrouter"
    )
    assert names.evidence["provider_throttled"] is True
    assert all(a["failure_class"] == "rate_limited" for a in names.evidence["attempts"])
    assert any("receipt status" in r for r in names.reasons)
    assert any("keyed delegation exited" in r for r in names.reasons)
    assert "0 accepted attempts, expected exactly one" in names.reasons
    assert "receipt carries no accepted response" in names.reasons
    assert not any(
        "came from None" in r or "records no routing attempt" in r
        for c in record.clauses
        for r in c.reasons
    )


def test_mixed_failures_are_not_a_provider_throttle(
    observations: dict[str, Any],
) -> None:
    _failed_receipt(observations, ["rate_limited", "timeout"])

    record = probe.grade(observations)
    names = next(c for c in record.clauses if c.name == "names_provider")
    assert record.verdict == "FAIL"
    assert not names.passed
    assert not any(r.startswith("provider throttled") for r in names.reasons)
    assert names.evidence["provider_throttled"] is False


def test_failed_payload_environment_key_still_fails(
    observations: dict[str, Any],
) -> None:
    _failed_receipt(observations, ["rate_limited"] * 3, source="env")

    record = probe.grade(observations)
    customer_key = next(c for c in record.clauses if c.name == "customer_key")
    assert record.verdict == "FAIL"
    assert not customer_key.passed
    assert any("came from 'env'" in r for r in customer_key.reasons)


def test_success_payload_still_passes(observations: dict[str, Any]) -> None:
    record = probe.grade(observations)
    clauses = {c.name: c for c in record.clauses}
    assert record.verdict == "PASS"
    assert clauses["customer_key"].passed
    assert clauses["names_provider"].passed
    assert clauses["names_provider"].evidence["provider_throttled"] is False


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/c29_throttled_receipt_omn17427.json"
)
def test_recorded_throttled_run_names_the_throttle_not_the_key(
    recorded_response: dict[str, Any],
) -> None:
    """The 2026-10-23 13:58Z C29 run: three 429s on the customer's own key.

    The recording is that run's own observations and receipt. Graded from the
    receipt's terminal payload, the key is the stored one this session
    registered, and the clause that fails says the provider throttled.
    """
    obs = recorded_response["response"]
    spec = probe.PROVIDERS["openrouter"]
    receipt = obs["run_files"]["receipt.json"]
    assert isinstance(receipt, dict)
    obs = {
        **recorded_response["response"],
        "run_files": {
            **recorded_response["response"]["run_files"],
            "receipt.json": json.dumps(receipt),
        },
    }

    customer_key = probe.grade_customer_key(obs, spec, receipt)
    names = probe.grade_names_provider(obs, spec, receipt)

    assert customer_key.passed, customer_key.reasons
    assert not names.passed
    assert names.reasons[0].startswith(
        "provider throttled: all 3 attempts on byok-openrouter"
    )
    assert names.evidence["provider_throttled"] is True
    assert "receipt records no routing attempt" not in names.reasons
