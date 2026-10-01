# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Committed TLC evidence for the delegated code edit loop model (OMN-20290).

The model (one receipt per correlation id, writes confined to the writable
globs, only declared checks run, the turn cap holds, every runner ends) passes,
and every mutation fails its named property, against the
content digest of the spec and cfg files at the time TLC ran on the lab.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[3] / "formal" / "delegated_code_edit_loop"
RESULTS = MODEL_DIR / "results"

EXPECTED_VIOLATIONS = {
    "mut_claim": "Invariant OneReceipt is violated.",
    "mut_glob": "Invariant NoForbiddenWrite is violated.",
    "mut_check": "Invariant NoUndeclaredCheck is violated.",
    "mut_cap": "Invariant TurnBound is violated.",
}


def _digest() -> str:
    h = hashlib.sha256()
    h.update((MODEL_DIR / "DelegatedCodeEditLoop.tla").read_bytes())
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
