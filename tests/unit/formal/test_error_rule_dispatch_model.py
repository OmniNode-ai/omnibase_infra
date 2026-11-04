# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Committed TLC evidence for the error-rule dispatch model (OMN-19998).

The orchestrated error-rule workflow (dispatch key, unique-key claim, effect id,
lease, fence, read-back and escalation) passes every safety invariant, the action
properties StaleIgnored and ClearOnlyWhenGone, and the temporal properties
EveryKeyEnds and RecurrenceEscalates in the base, question-row and two-subject
models. Each mutation fails its named property and each reachability witness is
violated (the state it negates exists), against the content digest of the spec
and cfg files at the time TLC ran.

To regenerate results/, for each cfg run
``java -cp tla2tools.jar tlc2.TLC -config <cfg> -workers 4 ErrorRuleDispatch.tla``
in formal/error_rule_dispatch into results/<cfg stem>.out, then write the sha256
of the spec then the cfg files in sorted order to results/model.sha256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parents[3] / "formal" / "error_rule_dispatch"
RESULTS = MODEL_DIR / "results"
SPEC = "ErrorRuleDispatch.tla"

PASSING = ("Model", "Question", "TwoSubjects")

EXPECTED_VIOLATIONS = {
    "mut_nonatomic": "Invariant AtMostOneClaimPerGen is violated.",
    "mut_no_fence": "Action property StaleIgnored is violated.",
    "mut_lane_clears": "Action property ClearOnlyWhenGone is violated.",
    "mut_no_budget": "Invariant GenWithinBudget is violated.",
    "mut_mode_ungated": "Invariant NoShadowDispatch is violated.",
    "mut_repeat_question": "Invariant AskOnce is violated.",
    "mut_effect_twice": "Invariant EffectOnce is violated.",
    "mut_effect_twice_question": "Invariant AskOnce is violated.",
    "mut_no_expiry": "Temporal property EveryKeyEnds was violated.",
}

EXPECTED_WITNESSES = {
    "wit_expired_before_effect": "Invariant WitExpiredBeforeEffect is violated.",
    "wit_expired_after_lost_ack": "Invariant WitExpiredAfterLostAck is violated.",
    "wit_single_gen_clear_through_expiry": (
        "Invariant WitSingleGenClearThroughExpiry is violated."
    ),
    "wit_stale_dropped": "Invariant WitStaleDropped is violated.",
    "wit_second_gen": "Invariant WitSecondGen is violated.",
    "wit_budget_escalated": "Invariant WitBudgetEscalated is violated.",
    "wit_recurrence_escalated": "Invariant WitRecurrenceEscalated is violated.",
    "wit_demoted_cancel": "Invariant WitDemotedCancel is violated.",
    "wit_every_subject_dispatched": "Invariant WitEverySubjectDispatched is violated.",
}

PROPERTIES = (
    "StaleIgnored",
    "ClearOnlyWhenGone",
    "EveryKeyEnds",
    "RecurrenceEscalates",
)


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
def test_passing_configs_hold_every_property(name: str) -> None:
    cfg = (MODEL_DIR / f"{name}.cfg").read_text()
    for prop in PROPERTIES:
        assert prop in cfg
    out = (RESULTS / f"{name}.out").read_text()
    assert "Model checking completed. No error has been found." in out
    assert "0 states left on queue." in out


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(EXPECTED_VIOLATIONS.items()))
def test_each_mutation_fails_its_property(name: str, message: str) -> None:
    out = (RESULTS / f"{name}.out").read_text()
    assert f"Error: {message}" in out
    assert "No error has been found" not in out


@pytest.mark.unit
@pytest.mark.parametrize(("name", "message"), sorted(EXPECTED_WITNESSES.items()))
def test_each_reachability_witness_is_reached(name: str, message: str) -> None:
    out = (RESULTS / f"{name}.out").read_text()
    assert f"Error: {message}" in out


@pytest.mark.unit
def test_every_cfg_has_a_recorded_result() -> None:
    cfgs = {p.stem for p in MODEL_DIR.glob("*.cfg")}
    assert cfgs == set(PASSING) | set(EXPECTED_VIOLATIONS) | set(EXPECTED_WITNESSES)
    for name in cfgs:
        assert (RESULTS / f"{name}.out").is_file()


@pytest.mark.unit
def test_two_subject_model_keys_each_subject_separately() -> None:
    cfg = (MODEL_DIR / "TwoSubjects.cfg").read_text()
    assert 'Subjects = {"s1", "s2"}' in cfg
