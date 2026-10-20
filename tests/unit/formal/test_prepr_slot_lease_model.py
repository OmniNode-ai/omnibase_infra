# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Committed TLC evidence for the pre-PR slot lease model (OMN-20085).

The model (no two holders on live leases, every acquirer and the expired
holder finish) passes, and every mutation fails its named property, against
the content digest of the spec and cfg files at the time TLC ran.
mut_unguarded_release is the driver before this change: an expired holder's
release removed the lease a breaker had just written.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[3] / "formal" / "prepr_slot_lease"
RESULTS = MODEL_DIR / "results"

EXPECTED_VIOLATIONS = {
    "mut_two_step_break": "Invariant OneHolder is violated.",
    "mut_no_reread": "Invariant OneHolder is violated.",
    "mut_unguarded_release": "Invariant OneHolder is violated.",
    "mut_break_live": "Invariant OneHolder is violated.",
    "mut_leak_guard": "Temporal property Terminates was violated.",
}


def _digest() -> str:
    h = hashlib.sha256()
    h.update((MODEL_DIR / "PreprSlotLease.tla").read_bytes())
    for cfg in sorted(MODEL_DIR.glob("*.cfg")):
        h.update(cfg.read_bytes())
    return h.hexdigest()


@pytest.mark.unit
def test_results_bind_to_current_model_digest() -> None:
    assert (RESULTS / "model.sha256").read_text().strip() == _digest()


@pytest.mark.unit
def test_model_holds_every_property() -> None:
    out = (RESULTS / "Model.out").read_text()
    assert "Model checking completed. No error has been found." in out


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(EXPECTED_VIOLATIONS.items()))
def test_each_mutation_fails_its_property(name: str, message: str) -> None:
    out = (RESULTS / f"{name}.out").read_text()
    assert f"Error: {message}" in out
    assert "No error has been found" not in out


@pytest.mark.unit
def test_every_mutation_cfg_has_a_result() -> None:
    cfgs = {p.stem for p in MODEL_DIR.glob("mut_*.cfg")}
    assert cfgs == set(EXPECTED_VIOLATIONS)


@pytest.mark.unit
def test_results_carry_no_host_paths() -> None:
    for out in RESULTS.glob("*.out"):
        text = out.read_text()
        assert "/Users/" not in text and "/var/folders/" not in text, out.name
