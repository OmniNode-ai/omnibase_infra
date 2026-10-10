# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20844: the C29 customer chooses the OpenRouter model, and the receipt names it.

The customer's own flow is to name the model when they register the key
(``onex secret set llm.openrouter.api_key --model <id>``); omnimarket refuses an
OpenRouter registration with no model (BYOK_MODEL_NOT_CHOSEN) and never picks a
``:free`` model for the customer. So the probe takes ``--model``, passes it on
the registration, records it, and ``names_provider`` fails unless the receipt
and its accepted attempt name that model. Fakes only, no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import c29_customer_byo_key_probe as probe

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn19200"
CHOSEN = "google/gemini-2.5-flash-lite"

pytestmark = pytest.mark.unit


def _working() -> dict[str, Any]:
    payload = json.loads(
        (FIXTURES / "observations_working_path.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _reasons(record: probe.Record, clause: str) -> list[str]:
    return next(c for c in record.clauses if c.name == clause).reasons


def _with_chosen(obs: dict[str, Any], served: str) -> dict[str, Any]:
    receipt = json.loads(obs["run_files"]["receipt.json"])
    receipt["model"] = served
    result = receipt.get("receipt", {}).get("result", {})
    for attempt in result.get("attempts") or []:
        attempt["model_id"] = served
    obs["run_files"]["receipt.json"] = json.dumps(receipt)
    obs["model"] = CHOSEN
    return obs


def test_the_run_command_requires_the_chosen_model(tmp_path: Path) -> None:
    args = ["run", "--provider", "openrouter", "--key-env", "K"]
    for flag in (
        "--customer-home",
        "--customer-bin",
        "--workdir",
        "--trace-dir",
        "--observations-out",
        "--record",
    ):
        args += [flag, str(tmp_path / flag.strip("-"))]
    with pytest.raises(SystemExit):
        probe.main(args)
    assert not (tmp_path / "record").exists()


def test_a_receipt_on_the_chosen_model_names_it() -> None:
    obs = _with_chosen(_working(), CHOSEN)
    record = probe.grade(obs)
    names = next(c for c in record.clauses if c.name == "names_provider")
    assert not any("chosen model" in r for r in names.reasons)
    assert names.evidence["chosen_model"] == CHOSEN


def test_a_receipt_on_another_model_fails_names_provider() -> None:
    obs = _with_chosen(_working(), "google/gemma-4-31b-it:free")
    reasons = _reasons(probe.grade(obs), "names_provider")
    assert any(
        "google/gemma-4-31b-it:free" in r and CHOSEN in r and "chosen model" in r
        for r in reasons
    )


def test_an_observation_with_no_chosen_model_fails_names_provider() -> None:
    obs = _working()
    obs["provider"] = "openrouter"
    obs.pop("model", None)
    reasons = _reasons(probe.grade(obs), "names_provider")
    assert any("no chosen model" in r for r in reasons)


def test_the_registration_carries_the_chosen_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.ci._fake_delegate_cli_omn20505 import install_fakes

    home, bin_dir, work, strace = install_fakes(tmp_path)
    monkeypatch.setattr(probe.shutil, "which", lambda _name: str(strace))
    monkeypatch.setattr(probe, "find_source_trees", lambda _roots: [])
    monkeypatch.setenv("C29_TEST_KEY", "sk-or-test-not-a-real-key")
    args = argparse.Namespace(
        key_env="C29_TEST_KEY",
        provider="openrouter",
        model=CHOSEN,
        customer_home=str(home),
        customer_bin=str(bin_dir),
        workdir=str(work),
        trace_dir=str(tmp_path / "traces"),
        step_timeout="60",
        prompt="p",
    )

    obs = probe.observe_live(args)

    argv = obs["steps"]["secret_set"]["argv"]
    assert argv[-2:] == ["--model", CHOSEN]
    assert obs["model"] == CHOSEN


def _recorded_obs(recording: dict[str, Any], chosen: str) -> dict[str, Any]:
    receipt = recording["receipt"]
    return {
        "provider": "openrouter",
        "model": chosen,
        "steps": {"keyed": {"returncode": 0}},
        "run_files": {
            "receipt.json": json.dumps(receipt),
            "result.txt": receipt["receipt"]["result"]["response"],
        },
    }


@pytest.mark.live_contact("tests/ci/fixtures/c29_chosen_model_receipt_omn20844.json")
def test_a_recorded_openrouter_receipt_on_the_chosen_model_names_provider(
    recorded_response: dict[str, Any],
) -> None:
    obs = _recorded_obs(recorded_response, CHOSEN)
    clause = probe.grade_names_provider(
        obs, probe.PROVIDERS["openrouter"], recorded_response["receipt"]
    )
    assert clause.reasons == []
    assert clause.evidence["chosen_model"] == CHOSEN
    assert clause.evidence["receipt_model"] == CHOSEN


@pytest.mark.live_contact("tests/ci/fixtures/c29_chosen_model_receipt_omn20844.json")
def test_the_same_recorded_receipt_fails_for_a_model_the_customer_did_not_choose(
    recorded_response: dict[str, Any],
) -> None:
    obs = _recorded_obs(recorded_response, "openai/gpt-4.1-nano")
    clause = probe.grade_names_provider(
        obs, probe.PROVIDERS["openrouter"], recorded_response["receipt"]
    )
    assert any(
        "is not the chosen model 'openai/gpt-4.1-nano'" in r for r in clause.reasons
    )
