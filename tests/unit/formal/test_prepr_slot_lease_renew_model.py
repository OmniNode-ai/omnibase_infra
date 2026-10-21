# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Committed TLC evidence for the pre-PR slot lease model (OMN-20085, AC3).

The model (one operator per slot, a phase only under a live lease of its own,
a signal never abandons a lease, a dead runner's lease frees again and its
residue is cleaned) passes, and every mutation fails its named property,
against the content digest of the spec and cfg files at the time TLC ran on
the lab. The real-shell race tests in
tests/unit/scripts/test_prepr_slot_lease_omn18893.py check the driver's own
remote commands for the critical sections the model treats as one step.

Property (meaning)                             Kind       Mutation that must break it
MutualExclusion (one operator per slot)        invariant  racy_break_mutex, overrun_mutex, no_renew_check_mutex
OperatingOwnsLiveLease (a phase holds a lease) invariant  racy_break, overrun, no_renew_check
NoSoftKillAbandon (signal tears down)          invariant  no_signal_teardown
AbandonedReusable (dead lease frees again)     temporal   no_expiry
DeadSlotCleaned (dead residue is cleaned)      temporal   no_reaper

Bounds: 3 runners, 2 slots, TTL 3, phase length 3 (5 for overrun), 2 phases,
time 0..8, one kill; Model.cfg is about 1.9 million distinct states. To
regenerate results/ on a lab host, for each cfg in Model.cfg and mut_*.cfg run
``java -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 4 PreprSlotLease.tla``
in formal/prepr_slot_lease_renew (or the same java line in the
eclipse-temurin:17-jre-alpine image) into results/<cfg stem>.out, then write
``LC_ALL=C cat PreprSlotLease.tla *.cfg | shasum -a 256`` (the digest alone)
to results/model.sha256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[3] / "formal" / "prepr_slot_lease_renew"
RESULTS = MODEL_DIR / "results"

TEMPORAL = "Temporal properties were violated."
EXPECTED_VIOLATIONS = {
    "mut_racy_break": "Invariant OperatingOwnsLiveLease is violated.",
    "mut_racy_break_mutex": "Invariant MutualExclusion is violated.",
    "mut_overrun": "Invariant OperatingOwnsLiveLease is violated.",
    "mut_overrun_mutex": "Invariant MutualExclusion is violated.",
    "mut_no_renew_check": "Invariant OperatingOwnsLiveLease is violated.",
    "mut_no_renew_check_mutex": "Invariant MutualExclusion is violated.",
    "mut_no_signal_teardown": "Invariant NoSoftKillAbandon is violated.",
    "mut_no_expiry": TEMPORAL,
    "mut_no_reaper": TEMPORAL,
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
    assert "0 states left on queue." in out


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
