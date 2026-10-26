# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Committed TLC evidence for the runtime primary election model (OMN-21418, AC1, AC2, AC5).

The model passes AtMostOneActingPrimary and EpochMonotonic, each mutation
fails its named invariant, the witness is violated, and a design-review verdict
is recorded, against the content digest of the spec and cfg files.

To regenerate results/, for each cfg run
``java -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 4 RuntimePrimaryElection.tla``
in formal/runtime_primary_election into results/<cfg stem>.out, then write the
sha256 of the spec then the cfg files in sorted order to results/model.sha256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[3] / "formal" / "runtime_primary_election"
RESULTS = MODEL_DIR / "results"
SPEC = "RuntimePrimaryElection.tla"

PASSING = ("Model",)
EXPECTED_VIOLATIONS = {
    "mut_no_self_fence": "Invariant AtMostOneActingPrimary is violated.",
    "mut_unguarded_write": "Invariant EpochMonotonic is violated.",
    "wit_two_epochs_applied": "Invariant NotTwoEpochsApplied is violated.",
}


def _digest() -> str:
    h = hashlib.sha256()
    h.update((MODEL_DIR / SPEC).read_bytes())
    for cfg in sorted(MODEL_DIR.glob("*.cfg")):
        h.update(cfg.read_bytes())
    return h.hexdigest()


@pytest.mark.unit
def test_results_bind_to_current_model_digest() -> None:
    assert (RESULTS / "model.sha256").read_text().strip() == _digest()


@pytest.mark.unit
@pytest.mark.parametrize("name", PASSING)
def test_passing_configs_hold_every_invariant(name: str) -> None:
    out = (RESULTS / f"{name}.out").read_text()
    assert "Model checking completed. No error has been found." in out
    assert "0 states left on queue." in out


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(EXPECTED_VIOLATIONS.items()))
def test_each_mutation_fails_its_invariant(name: str, message: str) -> None:
    out = (RESULTS / f"{name}.out").read_text()
    assert f"Error: {message}" in out
    assert "No error has been found" not in out


@pytest.mark.unit
def test_every_cfg_has_a_recorded_result() -> None:
    cfgs = {p.stem for p in MODEL_DIR.glob("*.cfg")}
    assert cfgs == set(PASSING) | set(EXPECTED_VIOLATIONS)
    for name in cfgs:
        assert (RESULTS / f"{name}.out").is_file()


@pytest.mark.unit
def test_design_review_verdict_recorded() -> None:
    verdict = RESULTS / "design_review_verdict.txt"
    assert verdict.is_file()
    text = verdict.read_text()
    assert text.splitlines()[0].startswith("# Design-review verdict")
    assert "OMN-21418" in text
