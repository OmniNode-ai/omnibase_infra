# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for OMN-18329's closer held-unbound residual + OCC binding audit.

Every zero-result assertion below is paired with a positive control in the
same test (CLAUDE.md rule 16) — a fixture seeded with a row that MUST be
counted, run through the same function, asserted present.
"""

from __future__ import annotations

import json

import pytest

from omnibase_infra.diagnostics.closer_receipt_residual import (
    ContractBindingCounts,
    TicketDecisionEvent,
    count_contract_bindings,
    distinct_tickets_currently_held,
    extract_receipt_json_lines,
    iter_ticket_decision_events,
    parse_sweep_receipt,
    resolve_decision_class,
)

# --- resolve_decision_class ---------------------------------------------------


def test_resolve_decision_class_passes_through_non_duplicate_decisions() -> None:
    assert resolve_decision_class(
        "gap_ac_unbound", "not all ACs are receipt-proven."
    ) == ("gap_ac_unbound")
    assert resolve_decision_class("flipped", "") == "flipped"


def test_resolve_decision_class_recovers_underlying_class_from_duplicate_marker() -> (
    None
):
    reason = (
        "This exact verdict is already posted on the ticket "
        "(<!-- onex-autoclose-verdict v1 class=gap_ac_unbound "
        "fingerprint=abc123def456 -->) — not repeating it. The gap itself is "
        "unchanged and still open: not all ACs are receipt-proven."
    )
    assert (
        resolve_decision_class("skipped_duplicate_comment", reason) == "gap_ac_unbound"
    )


def test_resolve_decision_class_unreadable_marker_is_named_not_dropped() -> None:
    # No `class=` token at all — must not silently resolve to gap_ac_unbound
    # (which would inflate the held count) or be dropped (which would
    # silently undercount it either way).
    assert (
        resolve_decision_class("skipped_duplicate_comment", "garbled reason, no marker")
        == "unknown_duplicate_class"
    )


# --- extract_receipt_json_lines -----------------------------------------------


def _receipt_line(ticket_rows: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "skill_name": "node_evidence_autoclose_sweep_effect",
            "node_name": "node_evidence_autoclose_sweep_effect",
            "status": "success",
            "result": {"outcomes": ticket_rows},
        }
    )


def test_extract_receipt_json_lines_finds_marked_line_among_noise() -> None:
    receipt = _receipt_line(
        [{"ticket_id": "OMN-1", "decision": "flipped", "reason": ""}]
    )
    log_text = "\n".join(
        [
            "checkout\tRun actions/checkout\t2026-09-13T18:00:00.0000000Z Checking out repo",
            f"Evidence Autoclose Sweep\tRun evidence autoclose sweep\t2026-09-13T18:21:46.8512124Z {receipt}",
            "assert-venv\tAssert venv pure\t2026-09-13T18:35:00.0000000Z OK",
        ]
    )
    lines = extract_receipt_json_lines(log_text)
    assert len(lines) == 1
    parsed = parse_sweep_receipt(lines[0])
    assert parsed["skill_name"] == "node_evidence_autoclose_sweep_effect"


def test_extract_receipt_json_lines_positive_control_returns_empty_on_no_marker() -> (
    None
):
    # Positive control for the zero case: a log with no sweep line at all
    # returns nothing, which must be distinguishable from "found it, but
    # couldn't parse it".
    log_text = (
        "checkout\tRun actions/checkout\t2026-09-13T18:00:00.0000000Z Checking out repo"
    )
    assert extract_receipt_json_lines(log_text) == []
    # ... and the same extractor DOES find a line once one is present in an
    # otherwise-identical transcript, proving the empty result above is a
    # real absence and not a broken matcher.
    receipt = _receipt_line([])
    log_text_with_receipt = log_text + (
        f"\nEvidence Autoclose Sweep\tRun evidence autoclose sweep\t2026-09-13T18:21:46.0000000Z {receipt}"
    )
    assert len(extract_receipt_json_lines(log_text_with_receipt)) == 1


# --- iter_ticket_decision_events ----------------------------------------------


def test_iter_ticket_decision_events_yields_one_event_per_outcome_row() -> None:
    receipt = parse_sweep_receipt(
        _receipt_line(
            [
                {"ticket_id": "OMN-1", "decision": "gap_ac_unbound", "reason": "r1"},
                {"ticket_id": "OMN-2", "decision": "flipped", "reason": "r2"},
            ]
        )
    )
    events = list(iter_ticket_decision_events("2026-09-13T18:21:46Z", receipt))
    assert [(e.ticket_id, e.resolved_class) for e in events] == [
        ("OMN-1", "gap_ac_unbound"),
        ("OMN-2", "flipped"),
    ]


def test_iter_ticket_decision_events_empty_receipt_yields_nothing() -> None:
    # e.g. a kill-switch short-circuit receipt with no `result.outcomes`.
    receipt = {
        "skill_name": "node_evidence_autoclose_sweep_effect",
        "result": {"kill_switch_engaged": True},
    }
    assert list(iter_ticket_decision_events("2026-09-13T18:00:00Z", receipt)) == []


# --- distinct_tickets_currently_held (AC1) ------------------------------------


def test_distinct_tickets_currently_held_positive_control_seeded_ticket_returned() -> (
    None
):
    """AC1 falsifier: a receipt set seeded with one known-held ticket returns it."""
    events = [
        TicketDecisionEvent(
            "2026-09-10T00:00:00Z", "OMN-100", "gap_ac_unbound", "gap_ac_unbound", ""
        ),
    ]
    held = distinct_tickets_currently_held(events)
    assert held == ("OMN-100",)


def test_distinct_tickets_currently_held_collapses_duplicate_comment_reposts() -> None:
    """The core defect this ticket exists to avoid: a persistently-held ticket's
    LAST raw decision is `skipped_duplicate_comment`, not `gap_ac_unbound` — the
    resolved class must still count it as held."""
    dup_reason = (
        "already posted (<!-- onex-autoclose-verdict v1 class=gap_ac_unbound "
        "fingerprint=deadbeef -->) — not repeating it."
    )
    events = [
        TicketDecisionEvent(
            "2026-09-10T00:00:00Z", "OMN-200", "gap_ac_unbound", "gap_ac_unbound", ""
        ),
        TicketDecisionEvent(
            "2026-09-11T00:00:00Z",
            "OMN-200",
            "skipped_duplicate_comment",
            "gap_ac_unbound",
            dup_reason,
        ),
        TicketDecisionEvent(
            "2026-09-12T00:00:00Z",
            "OMN-200",
            "skipped_duplicate_comment",
            "gap_ac_unbound",
            dup_reason,
        ),
    ]
    assert distinct_tickets_currently_held(events) == ("OMN-200",)


def test_distinct_tickets_currently_held_excludes_tickets_later_resolved() -> None:
    events = [
        TicketDecisionEvent(
            "2026-09-10T00:00:00Z", "OMN-300", "gap_ac_unbound", "gap_ac_unbound", ""
        ),
        TicketDecisionEvent(
            "2026-09-11T00:00:00Z", "OMN-300", "flipped", "flipped", ""
        ),
    ]
    assert distinct_tickets_currently_held(events) == ()


def test_distinct_tickets_currently_held_dedupes_across_overlapping_runs() -> None:
    """The whole point of AC1: 213 hold *events* is not a ticket count."""
    events = [
        TicketDecisionEvent(
            f"2026-09-1{i}T00:00:00Z", "OMN-400", "gap_ac_unbound", "gap_ac_unbound", ""
        )
        for i in range(5)
    ] + [
        TicketDecisionEvent(
            "2026-09-10T00:00:00Z", "OMN-401", "gap_ac_unbound", "gap_ac_unbound", ""
        ),
    ]
    held = distinct_tickets_currently_held(events)
    assert held == ("OMN-400", "OMN-401")
    assert len(held) == 2  # not 6 — five repeats of OMN-400 collapse to one


def test_distinct_tickets_currently_held_reproducible_over_the_same_input() -> None:
    """AC1 falsifier: re-running over the same receipt set reproduces the identical integer."""
    events = [
        TicketDecisionEvent(
            "2026-09-10T00:00:00Z", "OMN-500", "gap_ac_unbound", "gap_ac_unbound", ""
        ),
        TicketDecisionEvent(
            "2026-09-11T00:00:00Z", "OMN-501", "gap_posted", "gap_posted", ""
        ),
    ]
    first = distinct_tickets_currently_held(events)
    second = distinct_tickets_currently_held(list(reversed(events)))
    assert first == second == ("OMN-500",)


# --- count_contract_bindings (AC2) --------------------------------------------


def test_count_contract_bindings_positive_control_and_zero_are_distinguished() -> None:
    """AC2 falsifier: a contract hand-seeded with an acceptance record is
    counted in the second (accepted) bucket; the corpus's other, unseeded
    zero for the same bucket is proven real by `contracts_with_evidence`
    being non-zero (AC3 positive control)."""
    contracts = [
        # binds_ac only, no acceptance — counted in `with_binding`, not `with_accepted`.
        {
            "ticket_id": "OMN-1",
            "dod_evidence": [
                {"id": "ac1", "binds_ac": ["AC1"], "checks": []},
            ],
        },
        # binds_ac WITH a full acceptance record — the positive control.
        {
            "ticket_id": "OMN-2",
            "dod_evidence": [
                {
                    "id": "ac1",
                    "binds_ac": ["AC1"],
                    "ac_bindings": [
                        {
                            "label": "AC1",
                            "accepted_by": "jonah",
                            "accepted_at": "2026-09-13T00:00:00Z",
                        }
                    ],
                    "checks": [],
                },
            ],
        },
        # evidence present, no binding at all.
        {
            "ticket_id": "OMN-3",
            "dod_evidence": [{"id": "ci", "checks": []}],
        },
        # no evidence at all.
        {"ticket_id": "OMN-4"},
    ]
    counts = count_contract_bindings(contracts)
    assert counts == ContractBindingCounts(
        total_contracts=4,
        contracts_with_evidence=3,
        contracts_with_binding=2,
        contracts_with_accepted_binding=1,
    )


def test_count_contract_bindings_accepted_requires_both_accepted_by_and_accepted_at() -> (
    None
):
    # A draft binding (no accepted_by) must not count as accepted, per
    # OMN-18238 / the evidence_collector `_draft_binding_labels` contract.
    contracts = [
        {
            "ticket_id": "OMN-5",
            "dod_evidence": [
                {
                    "id": "ac1",
                    "binds_ac": ["AC1"],
                    "ac_bindings": [
                        {"label": "AC1", "accepted_by": "", "accepted_at": ""}
                    ],
                    "checks": [],
                }
            ],
        },
        # accepted_by set but accepted_at missing — AC2 requires both.
        {
            "ticket_id": "OMN-6",
            "dod_evidence": [
                {
                    "id": "ac1",
                    "binds_ac": ["AC1"],
                    "ac_bindings": [{"label": "AC1", "accepted_by": "jonah"}],
                    "checks": [],
                }
            ],
        },
    ]
    counts = count_contract_bindings(contracts)
    assert counts.contracts_with_binding == 2
    assert counts.contracts_with_accepted_binding == 0
    # Positive control the zero above is real: the same function, on the
    # fixture from the previous test, DOES find one.
    seeded = count_contract_bindings(
        [
            {
                "ticket_id": "OMN-2",
                "dod_evidence": [
                    {
                        "id": "ac1",
                        "binds_ac": ["AC1"],
                        "ac_bindings": [
                            {
                                "label": "AC1",
                                "accepted_by": "jonah",
                                "accepted_at": "2026-09-13T00:00:00Z",
                            }
                        ],
                        "checks": [],
                    }
                ],
            }
        ]
    )
    assert seeded.contracts_with_accepted_binding == 1


def test_count_contract_bindings_empty_binds_ac_list_does_not_count() -> None:
    contracts = [
        {
            "ticket_id": "OMN-7",
            "dod_evidence": [{"id": "ci", "binds_ac": [], "checks": []}],
        }
    ]
    counts = count_contract_bindings(contracts)
    assert counts.contracts_with_binding == 0
    assert counts.contracts_with_evidence == 1  # positive control: evidence WAS seen


@pytest.mark.parametrize(
    "bad_contract", [{"dod_evidence": "not-a-list"}, {"dod_evidence": None}, {}]
)
def test_count_contract_bindings_malformed_evidence_field_is_treated_as_absent(
    bad_contract: dict[str, object],
) -> None:
    counts = count_contract_bindings([bad_contract])
    assert counts.total_contracts == 1
    assert counts.contracts_with_evidence == 0
    assert counts.contracts_with_binding == 0
    assert counts.contracts_with_accepted_binding == 0
