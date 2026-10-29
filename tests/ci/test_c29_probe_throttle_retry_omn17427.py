# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A provider 429 on every attempt is repeated, never graded as the criterion (OMN-17427).

OpenRouter answers 429 for a shared-pool model on a few percent of hourly C29 runs
and each answer clears by the next run. The probe repeats a keyed delegation whose
every routing attempt was a provider 429, a bounded number of times after a pause.
It grades the last run, and the earlier throttled runs stay in the record with
their network traces graded by no_omninode.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c29_customer_byo_key_probe as probe

pytestmark = pytest.mark.unit


def _receipt(status: str, failures: Sequence[str | None]) -> dict[str, Any]:
    attempts = [
        {"backend_id": "byok-openrouter", "failure_class": failure}
        for failure in failures
    ]
    return {
        "status": status,
        "receipt": {
            "result": {
                "secret_source": "store",
                "secret_ref": "llm.openrouter.api_key",
                "attempts": attempts,
            }
        },
    }


@pytest.mark.parametrize(
    ("receipt", "expected"),
    [
        (_receipt("failed", ["rate_limited"] * 3), True),
        (_receipt("failed", ["rate_limited"]), True),
        (_receipt("success", ["rate_limited"]), False),
        (_receipt("failed", ["rate_limited", "auth"]), False),
        (_receipt("failed", ["auth"]), False),
        (_receipt("failed", []), False),
        ({}, False),
        ("not a receipt", False),
    ],
)
def test_receipt_provider_throttled(receipt: Any, expected: bool) -> None:
    assert probe.receipt_provider_throttled(receipt) is expected


def test_receipt_provider_throttled_reads_the_terminal_payload() -> None:
    payload = _receipt("failed", ["rate_limited"] * 2)["receipt"]["result"]
    receipt = {"status": "failed", "receipt": {"result": {"terminal_payload": payload}}}
    assert probe.receipt_provider_throttled(receipt) is True


class _Customer:
    """A customer machine whose keyed runs answer from a script of receipts."""

    def __init__(self, root: Path, keyed_receipts: list[dict[str, Any]]) -> None:
        self.home = root / "home"
        self.bin = root / "bin"
        self.workdir = root / "work"
        self.traces = root / "traces"
        for directory in (self.home, self.bin):
            directory.mkdir()
        onex = self.bin / "onex"
        onex.write_text(
            '#!/bin/sh\nif [ "$1" = secret ]; then echo "  llm.openrouter.api_key"; fi\n'
        )
        onex.chmod(0o755)
        self.keyed_receipts = keyed_receipts
        self.keyed_calls = 0
        self.runs_root = probe.delegate_runs_root({"HOME": str(self.home)})

    def run_step(self, name: str, argv: Sequence[str], **_: Any) -> dict[str, Any]:
        stdout = ""
        if name == "keyed":
            receipt = self.keyed_receipts[self.keyed_calls]
            run_id = f"run-{self.keyed_calls}"
            self.keyed_calls += 1
            run_dir = self.runs_root / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "receipt.json").write_text(json.dumps(receipt))
            stdout = json.dumps({"run_id": run_id})
        elif name == "keyless":
            self.runs_root.mkdir(parents=True, exist_ok=True)
        return {
            "argv": list(argv),
            "stdin": None,
            "returncode": 0 if name != "keyed" else 1,
            "stdout": stdout,
            "stderr": "",
            "strace": f"strace of {name} #{self.keyed_calls}",
        }

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            key_env="C29_TEST_KEY",
            provider="openrouter",
            model="synthetic-model",
            prompt="hello",
            customer_home=str(self.home),
            customer_bin=str(self.bin),
            workdir=str(self.workdir),
            trace_dir=str(self.traces),
            step_timeout="60",
        )


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(probe.time, "sleep", recorded.append)
    monkeypatch.setattr(probe.shutil, "which", lambda _: "/usr/bin/strace")
    monkeypatch.setenv("C29_TEST_KEY", "sk-test-not-a-real-key")
    return recorded


def _observe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipts: list[dict[str, Any]]
) -> tuple[dict[str, Any], _Customer]:
    customer = _Customer(tmp_path, receipts)
    monkeypatch.setattr(probe, "_run_step", customer.run_step)
    return probe.observe_live(customer.args()), customer


def test_a_throttled_keyed_run_is_repeated_and_the_success_is_graded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    throttled = _receipt("failed", ["rate_limited"] * 3)
    ok = _receipt("success", [None])
    obs, customer = _observe(tmp_path, monkeypatch, [throttled, ok])

    assert customer.keyed_calls == 2
    assert sleeps == [probe.KEYED_THROTTLE_BACKOFF_S[0]]
    assert obs["keyed_throttled_runs"] == 1
    assert list(obs["steps"]) == [
        "init",
        "keyless",
        "secret_set",
        f"{probe.KEYED_THROTTLED_PREFIX}1",
        "keyed",
        "omninode_control",
    ]
    assert obs["run_dir"].endswith("run-1")
    graded = json.loads(obs["run_files"]["receipt.json"])
    assert graded["status"] == "success"


def test_a_run_that_is_not_throttled_is_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    broken = _receipt("failed", ["auth"])
    obs, customer = _observe(tmp_path, monkeypatch, [broken])

    assert customer.keyed_calls == 1
    assert sleeps == []
    assert obs["keyed_throttled_runs"] == 0


def test_repeats_are_bounded_and_the_last_throttle_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    throttled = _receipt("failed", ["rate_limited"] * 3)
    receipts = [throttled] * (probe.KEYED_THROTTLE_RETRIES + 1)
    obs, customer = _observe(tmp_path, monkeypatch, receipts)

    assert customer.keyed_calls == probe.KEYED_THROTTLE_RETRIES + 1
    assert sleeps == list(
        probe.KEYED_THROTTLE_BACKOFF_S[: probe.KEYED_THROTTLE_RETRIES]
    )
    assert obs["keyed_throttled_runs"] == probe.KEYED_THROTTLE_RETRIES
    receipt = json.loads(obs["run_files"]["receipt.json"])
    names = probe.grade_names_provider(obs, probe.PROVIDERS["openrouter"], receipt)
    assert not names.passed
    assert names.evidence["keyed_throttled_runs"] == probe.KEYED_THROTTLE_RETRIES


def test_the_key_is_never_in_the_record_after_a_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    obs, _ = _observe(
        tmp_path,
        monkeypatch,
        [_receipt("failed", ["rate_limited"]), _receipt("success", [None])],
    )
    assert "sk-test-not-a-real-key" not in json.dumps(obs)
    assert obs["key_leaks"] == []


def test_no_omninode_grades_the_throttled_runs_traces_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    omninode_trace = probe.Trace(queried=["api.omninode.ai"])
    quiet = probe.Trace()
    traces = {"quiet": quiet, "omninode": omninode_trace}
    monkeypatch.setattr(probe, "parse_trace", traces.__getitem__)
    obs = {
        "nameservers": [],
        "steps": {
            "init": {"strace": "quiet"},
            "keyless": {"strace": "quiet", "returncode": 1, "stdout": "{}"},
            "secret_set": {"strace": "quiet"},
            f"{probe.KEYED_THROTTLED_PREFIX}1": {"strace": "omninode"},
            "keyed": {"strace": "quiet"},
            "omninode_control": {"strace": "omninode"},
        },
    }
    clause = probe.grade_no_omninode(obs, probe.PROVIDERS["openrouter"])
    assert any(
        r.startswith(f"{probe.KEYED_THROTTLED_PREFIX}1: OmniNode name(s) looked up")
        for r in clause.reasons
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/c29_throttled_receipt_omn17427.json"
)
def test_recorded_throttled_run_is_the_one_the_probe_repeats(
    recorded_response: dict[str, Any],
) -> None:
    """The 2026-10-23 13:58Z C29 run (three 429s) is a receipt the probe repeats."""
    receipt = recorded_response["response"]["run_files"]["receipt.json"]
    assert isinstance(receipt, dict)
    assert probe.receipt_provider_throttled(receipt) is True


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/c29_throttled_receipt_omn17427.json"
)
def test_recorded_throttled_receipt_reads_through_every_layer(
    recorded_response: dict[str, Any],
) -> None:
    """The recorded failed receipt's nested layers stay readable to the grader."""
    receipt = recorded_response["response"]["run_files"]["receipt.json"]
    assert probe._receipt_result(receipt)["secret_source"] == "store"
    clean = probe.grade_clean_machine(recorded_response["response"], receipt)
    assert set(clean.evidence["packages"]) == set(probe.ONEX_DISTRIBUTIONS)
    refusal = probe._find_refusal(receipt)
    assert refusal is not None and "429 Too Many Requests" in refusal
