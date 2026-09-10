# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the link-5 replay + tier-2 verifier (OMN-16964).

These cover the four cases OMN-16964 acceptance item 4 names: complete chain
replays green, a missing hop, a verifier SKIP, and a verifier that could not
run. No network, no database, no bus — the whole point of putting the replay
and the verify in a pure module is that their honesty is provable here.

The defect this module exists to prevent is a verifier that reports a pass
because it never ran the check. Every test below that asserts a non-passing
outcome is asserting the absence of that defect, so none of them may be
weakened into "returns something".
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.chain_replay import (
    assemble_replay_and_verify,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.enum_tier_two_verdict import (
    EnumTierTwoVerdict,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_hop import (
    ModelObservedHop,
)

CORRELATION = UUID("11111111-2222-3333-4444-555555555555")

# Envelope ids, spelled as constants so a test reads as a CHAIN rather than
# as a wall of hex. The linkage between them is the thing under test.
E0 = UUID("aaaaaaaa-0000-0000-0000-000000000000")
E1 = UUID("aaaaaaaa-0000-0000-0000-000000000001")
E2 = UUID("aaaaaaaa-0000-0000-0000-000000000002")
E3 = UUID("aaaaaaaa-0000-0000-0000-000000000003")
UNRELATED = UUID("bbbbbbbb-0000-0000-0000-00000000ffff")
OTHER_CORRELATION = UUID("99999999-9999-9999-9999-999999999999")

DECLARED = (
    "onex.cmd.omnimarket.delegate-skill.v1",
    "onex.cmd.omnibase-infra.delegation-routing-request.v1",
    "onex.evt.omnibase-infra.routing-decision.v1",
    "onex.evt.omnimarket.delegate-skill-completed.v1",
)


def _hop(
    topic: str, envelope_id: str, parent: str, correlation: str = CORRELATION
) -> ModelObservedHop:
    return ModelObservedHop(
        topic=topic,
        envelope_id=envelope_id,
        parent_envelope_id=parent,
        correlation_id=correlation,
    )


def _complete_chain() -> tuple[ModelObservedHop, ...]:
    return (
        _hop(DECLARED[0], E0, None),
        _hop(DECLARED[1], E1, E0),
        _hop(DECLARED[2], E2, E1),
        _hop(DECLARED[3], E3, E2),
    )


@pytest.mark.unit
def test_complete_chain_replays_green_and_verifies_pass() -> None:
    """A complete, causally intact chain that matches the declared topology."""
    rows = assemble_replay_and_verify(CORRELATION, _complete_chain(), DECLARED)

    assert len(rows) == 4
    assert [row.hop for row in rows] == list(DECLARED)
    assert [row.hop_index for row in rows] == [0, 1, 2, 3]
    assert all(row.replay_green for row in rows), [
        (row.hop, row.replay_detail) for row in rows if not row.replay_green
    ]
    assert all(row.verifier_verdict is EnumTierTwoVerdict.PASS for row in rows)
    # The canary reads the LAST row's verdict as the chain verdict.
    assert rows[-1].verifier_verdict is EnumTierTwoVerdict.PASS


@pytest.mark.unit
def test_missing_hop_is_absent_from_the_chain_and_never_silently_filled() -> None:
    """A hop that never happened must not appear as a row at all.

    OMN-16964 scope: "no gaps tolerated silently". The canary checks
    completeness by hop NAME against its own declared expected set, so a
    fabricated placeholder row would defeat that check outright.
    """
    observed = (
        _hop(DECLARED[0], E0, None),
        # DECLARED[1] never happened.
        _hop(DECLARED[2], E2, E0),
        _hop(DECLARED[3], E3, E2),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED)

    assert [row.hop for row in rows] == [DECLARED[0], DECLARED[2], DECLARED[3]]
    assert DECLARED[1] not in [row.hop for row in rows]
    # The out-of-order arrival is a topology violation the tier-2 verifier sees.
    assert rows[1].verifier_verdict is EnumTierTwoVerdict.FAIL
    assert rows[-1].verifier_verdict is EnumTierTwoVerdict.FAIL


@pytest.mark.unit
def test_broken_causal_link_fails_the_replay() -> None:
    """Replay is a re-derivation, not a copy of a stored flag.

    Hop 2's recorded parent disagrees with hop 1's envelope id, so the chain
    did not reproduce and replay_green is false for that hop.
    """
    observed = (
        _hop(DECLARED[0], E0, None),
        _hop(DECLARED[1], E1, E0),
        _hop(DECLARED[2], E2, UNRELATED),
        _hop(DECLARED[3], E3, E2),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED)

    assert rows[0].replay_green is True
    assert rows[1].replay_green is True
    assert rows[2].replay_green is False
    assert rows[2].replay_detail != ""
    # A broken link is a replay failure, NOT a verifier skip.
    assert rows[2].verifier_verdict is EnumTierTwoVerdict.PASS


@pytest.mark.unit
def test_correlation_drift_mid_chain_fails_the_replay() -> None:
    """The OMN-16931 defect class: the correlation id changes mid-chain."""
    observed = (
        _hop(DECLARED[0], E0, None),
        _hop(DECLARED[1], E1, E0, correlation=OTHER_CORRELATION),
        _hop(DECLARED[2], E2, E1),
        _hop(DECLARED[3], E3, E2),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED)

    assert rows[1].replay_green is False
    assert "correlation" in rows[1].replay_detail.lower()


@pytest.mark.unit
def test_verifier_skips_when_no_topology_is_declared_and_skip_is_not_pass() -> None:
    """No declaration to verify against yields SKIP on every row.

    This is the member the word "honest" in the OMN-16025 gate text exists
    for. A verifier with nothing to check against has NOT passed.
    """
    rows = assemble_replay_and_verify(CORRELATION, _complete_chain(), ())

    assert len(rows) == 4
    assert all(row.verifier_verdict is EnumTierTwoVerdict.SKIP for row in rows)
    assert rows[-1].verifier_verdict is not EnumTierTwoVerdict.PASS
    assert all(row.verifier_detail != "" for row in rows)
    # The replay itself still ran — it needs no declaration.
    assert all(row.replay_green for row in rows)


@pytest.mark.unit
def test_verifier_skips_the_hops_the_declaration_does_not_reach() -> None:
    """A declaration shorter than the observed chain skips the overflow.

    Reporting PASS for a hop the declaration says nothing about would be a
    verdict invented rather than derived.
    """
    rows = assemble_replay_and_verify(CORRELATION, _complete_chain(), DECLARED[:2])

    assert rows[0].verifier_verdict is EnumTierTwoVerdict.PASS
    assert rows[1].verifier_verdict is EnumTierTwoVerdict.PASS
    assert rows[2].verifier_verdict is EnumTierTwoVerdict.SKIP
    assert rows[3].verifier_verdict is EnumTierTwoVerdict.SKIP


@pytest.mark.unit
def test_empty_observation_yields_no_rows_rather_than_a_green_chain() -> None:
    """Nothing observed is not a chain that replayed green.

    The canary reads `all(...)` over the rows, which is vacuously true on an
    empty sequence — so an empty chain MUST return zero rows and let the
    canary's own row-count check refuse it, never one synthetic green row.
    """
    rows = assemble_replay_and_verify(CORRELATION, (), DECLARED)

    assert rows == ()


@pytest.mark.unit
def test_verdict_never_upgrades_a_skip_to_a_pass_under_any_ordering() -> None:
    """Property: SKIP is never rendered as PASS, whatever the input shape."""
    for declared in ((), DECLARED[:1], DECLARED[:3], DECLARED):
        rows = assemble_replay_and_verify(CORRELATION, _complete_chain(), declared)
        for index, row in enumerate(rows):
            if index >= len(declared):
                assert row.verifier_verdict is EnumTierTwoVerdict.SKIP
