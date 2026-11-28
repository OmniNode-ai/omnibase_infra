# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The committed TLC results of the lane work loop model hold (OMN-26700)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[2] / "formal" / "lane_work_loop"
RESULTS = MODEL_DIR / "results"

PROPERTIES = (
    "OneTerminal",
    "OneLiveWorkflowPerChange",
    "HandoffAtProvenHead",
    "NoLostOutput",
    "NoDuplicateEffect",
    "RetryBound",
    "AnswerInContract",
    "DeadlineFires",
    "Terminates",
)

MUTATIONS = {
    "mut_terminal_twice": "Invariant OneTerminal is violated",
    "mut_random_workflow_id": "Invariant OneLiveWorkflowPerChange is violated",
    "mut_skip_confirmation": "Invariant HandoffAtProvenHead is violated",
    "mut_commit_before_outputs": "Invariant NoLostOutput is violated",
    "mut_resend_without_readback": "Invariant NoDuplicateEffect is violated",
    "mut_count_refused_input": "Invariant RetryBound is violated",
    "mut_answer_any_state": "Invariant AnswerInContract is violated",
    "mut_no_tick": "Temporal property DeadlineFires was violated",
}


@pytest.mark.unit
def test_model_passes_all_properties() -> None:
    assert "No error has been found" in (RESULTS / "Model.out").read_text()


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(MUTATIONS.items()))
def test_mutation_breaks_its_property(name: str, message: str) -> None:
    assert message in (RESULTS / f"{name}.out").read_text()


@pytest.mark.unit
def test_no_tick_mutation_names_only_its_property() -> None:
    cfg = (MODEL_DIR / "mut_no_tick.cfg").read_text()
    assert re.search(r"^PROPERTIES DeadlineFires$", cfg, re.MULTILINE)
    out = (RESULTS / "mut_no_tick.out").read_text()
    assert "Temporal property Terminates was violated" not in out


@pytest.mark.unit
def test_spec_defines_nine_properties() -> None:
    spec = (MODEL_DIR / "LaneWorkLoop.tla").read_text()
    for prop in PROPERTIES:
        assert re.search(rf"^{prop} ==", spec, re.MULTILINE), prop


@pytest.mark.unit
def test_results_digest_matches_model_files() -> None:
    for line in (RESULTS / "model.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = RESULTS / name.strip().lstrip("*")
        if not path.exists():
            path = MODEL_DIR / name.strip().lstrip("*")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
