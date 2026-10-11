# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bind the reviewed lab-reconcile model to its TLC evidence (OMN-19421).

Use the same committed-evidence gate as the other formal models. Both safety
and liveness need a passing control; every unsafe configuration must fail the
property it removes. A changed model, configuration or log invalidates the run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "formal" / "lab_reconcile"
MODEL_SHA256 = "5ef83ea14b2810b6b74202b3cb1ab037fbdb0abbf17ac4022dd44b2292ef958d"
PASS_MESSAGE = "Model checking completed. No error has been found."
EXPECTED_RESULTS = {
    "MC_design": PASS_MESSAGE,
    "MC_design_live": PASS_MESSAGE,
    "MC_nolock": "Error: Invariant NoOverlap is violated.",
    "MC_guard_in_callers": "Error: Invariant NoWindowCollision is violated.",
    "MC_guard_in_callers_agent_only": "Error: Invariant NoWindowCollision is violated.",
    "MC_no_reaper_live": "Error: Temporal property EveryStartedTerminates was violated.",
    "MC_reaper_blind": "Error: Invariant AtMostOneReceipt is violated.",
    "MC_no_noop": "Error: Invariant RolledBackNeverGreen is violated.",
}


def _verify_digest(path: Path, expected: str) -> None:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path.name


@pytest.mark.unit
def test_model_is_the_reviewed_attachment() -> None:
    _verify_digest(MODEL_DIR / "LabReconcile.tla", MODEL_SHA256)


@pytest.mark.unit
def test_every_configuration_and_result_is_bound() -> None:
    evidence = json.loads((MODEL_DIR / "evidence.json").read_text())
    cfgs = {p.stem for p in MODEL_DIR.glob("*.cfg")}
    logs = {p.stem for p in (MODEL_DIR / "results").glob("*.out")}
    assert cfgs == logs == set(EXPECTED_RESULTS) == set(evidence["runs"])
    for name, run in evidence["runs"].items():
        _verify_digest(MODEL_DIR / f"{name}.cfg", run["config_sha256"])
        _verify_digest(MODEL_DIR / "results" / f"{name}.out", run["log_sha256"])
    assert evidence["model_sha256"] == MODEL_SHA256


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(EXPECTED_RESULTS.items()))
def test_tlc_result(name: str, message: str) -> None:
    out = (MODEL_DIR / "results" / f"{name}.out").read_text()
    assert message in out
    assert "Semantic processing of module LabReconcile" in out
    if name in {"MC_design", "MC_design_live"}:
        assert "Error:" not in out
        assert "421992 distinct states found, 0 states left on queue." in out
    else:
        assert PASS_MESSAGE not in out
        assert "State 1:" in out


@pytest.mark.unit
def test_control_checks_all_ticket_properties() -> None:
    safety = (MODEL_DIR / "MC_design.cfg").read_text().split("INVARIANTS\n")[1]
    assert set(safety.split()) == {
        "TypeOK",
        "NoOverlap",
        "NoWindowCollision",
        "AtMostOneReceipt",
        "RolledBackNeverGreen",
    }
    live = (MODEL_DIR / "MC_design_live.cfg").read_text().split("PROPERTIES\n")[1]
    assert live.split() == ["EveryStartedTerminates"]
    control = (MODEL_DIR / "MC_design.cfg").read_text()
    unlocked = (MODEL_DIR / "MC_nolock.cfg").read_text()
    assert "UseLock = TRUE" in control
    assert "UseLock = FALSE" in unlocked

    # The negative control removes only the lock, keeping all other bounds,
    # guards, receipt behaviour and checked invariants identical.
    def strip_comments(text: str) -> str:
        return "\n".join(
            line.strip() for line in text.splitlines() if not line.startswith("\\*")
        )

    assert strip_comments(control) == strip_comments(
        unlocked.replace("UseLock = FALSE", "UseLock = TRUE")
    )


@pytest.mark.unit
@pytest.mark.live_contact("tests/ci/fixtures/lab_reconcile_ticket_omn19421.json")
def test_recorded_review_and_source(recorded_response: dict[str, object]) -> None:
    evidence = json.loads((MODEL_DIR / "evidence.json").read_text())
    assert evidence["ticket"] == "OMN-19421"
    assert evidence["archive_sha256"] == (
        "396c22ac77a2b60ff392d5dc947430e1e80641e692b81ef41ba5d53afbaf16b3"
    )
    # Replay the real attachment/review metadata captured through the shared
    # Linear adapter, then bind it to the imported model's source record.
    response = recorded_response["response"]
    assert isinstance(response, dict)
    issue = response["data"]["issue"]
    assert issue["identifier"] == evidence["ticket"]
    assert any(
        attachment["url"] == evidence["attachment_url"]
        for attachment in issue["attachments"]["nodes"]
    )
    assert any(
        comment["id"] == evidence["review"]["comment_id"]
        and comment["createdAt"] == evidence["review"]["recorded_at"]
        for comment in issue["comments"]["nodes"]
    )
    assert evidence["review"] == {
        "verdict": "CHANGE",
        "comment_id": "e63757a3-90d9-42f2-85cf-e19ec8685831",
        "recorded_at": "2026-09-24T17:23:57.998Z",
        "url": "https://linear.app/omninode/issue/OMN-19421",
    }


@pytest.mark.unit
def test_digest_check_refuses_changed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "changed.log"
    path.write_text("original evidence\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _verify_digest(path, digest)
    path.write_text("changed evidence\n")
    with pytest.raises(AssertionError, match=r"changed\.log"):
        _verify_digest(path, digest)
