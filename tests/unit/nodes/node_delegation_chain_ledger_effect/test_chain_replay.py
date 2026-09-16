# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the link-5 replay + tier-2 verifier (OMN-16964, OMN-18419).

These cover the four cases OMN-16964 acceptance item 4 names: complete chain
replays green, a missing hop, a verifier SKIP, and a verifier that could not
run. No network, no database, no bus — the whole point of putting the replay
and the verify in a pure module is that their honesty is provable here.

The defect this module exists to prevent is a verifier that reports a pass
because it never ran the check. Every test below that asserts a non-passing
outcome is asserting the absence of that defect, so none of them may be
weakened into "returns something".

OMN-18419 adds the second half: the declared topology is a TREE, and the
replay must grade each hop against the parent the DECLARATION names rather
than against whatever hop happened to precede it in time. The tree tests below
carry the real shape read off the .201 compose dev lane — a terminal that
branches off the head — plus the negative control that a genuinely wrong
parent still fails, because a grading rule that accepts a correct branch by
accepting anything is not a grading rule.
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
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
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

TOPICS = (
    "onex.cmd.omnimarket.delegate-skill.v1",
    "onex.cmd.omnibase-infra.delegation-routing-request.v1",
    "onex.evt.omnibase-infra.routing-decision.v1",
    "onex.evt.omnimarket.delegate-skill-completed.v1",
)

# A LINE: every hop is caused by the one before it. Kept as its own fixture so
# the tree tests below are visibly a different topology and not a tweak.
DECLARED = (
    ModelDeclaredChainHop(topic=TOPICS[0], parent=None),
    ModelDeclaredChainHop(topic=TOPICS[1], parent=TOPICS[0]),
    ModelDeclaredChainHop(topic=TOPICS[2], parent=TOPICS[1]),
    ModelDeclaredChainHop(topic=TOPICS[3], parent=TOPICS[2]),
)

# The TREE the .201 dev lane actually produces: the terminal is a consequence
# of consuming the delegate-skill COMMAND, so it branches off the head rather
# than hanging off the routing decision that precedes it in time.
DECLARED_TREE = (
    ModelDeclaredChainHop(topic=TOPICS[0], parent=None),
    ModelDeclaredChainHop(topic=TOPICS[1], parent=TOPICS[0]),
    ModelDeclaredChainHop(topic=TOPICS[2], parent=TOPICS[1]),
    ModelDeclaredChainHop(topic=TOPICS[3], parent=TOPICS[0]),
)


def _hop(
    topic: str,
    envelope_id: UUID,
    parent: UUID | None,
    correlation: UUID = CORRELATION,
) -> ModelObservedHop:
    return ModelObservedHop(
        topic=topic,
        envelope_id=envelope_id,
        parent_envelope_id=parent,
        correlation_id=correlation,
    )


def _complete_chain() -> tuple[ModelObservedHop, ...]:
    return (
        _hop(TOPICS[0], E0, None),
        _hop(TOPICS[1], E1, E0),
        _hop(TOPICS[2], E2, E1),
        _hop(TOPICS[3], E3, E2),
    )


def _observed_tree() -> tuple[ModelObservedHop, ...]:
    """The observed chain whose last hop branches off the head."""
    return (
        _hop(TOPICS[0], E0, None),
        _hop(TOPICS[1], E1, E0),
        _hop(TOPICS[2], E2, E1),
        _hop(TOPICS[3], E3, E0),
    )


@pytest.mark.unit
def test_complete_chain_replays_green_and_verifies_pass() -> None:
    """A complete, causally intact chain that matches the declared topology."""
    rows = assemble_replay_and_verify(CORRELATION, _complete_chain(), DECLARED)

    assert len(rows) == 4
    assert [row.hop for row in rows] == list(TOPICS)
    assert [row.hop_index for row in rows] == [0, 1, 2, 3]
    assert all(row.replay_green for row in rows), [
        (row.hop, row.replay_detail) for row in rows if not row.replay_green
    ]
    assert all(row.verifier_verdict is EnumTierTwoVerdict.PASS for row in rows)
    # The canary reads the LAST row's verdict as the chain verdict.
    assert rows[-1].verifier_verdict is EnumTierTwoVerdict.PASS


@pytest.mark.unit
def test_a_declared_tree_replays_green_when_a_hop_branches_off_the_head() -> None:
    """OMN-18419 AC2. The exact chain the dev lane produces must grade green.

    This is the case the positional re-derivation got wrong: the terminal's
    recorded parent IS an envelope observed on this correlation — the head —
    and it was marked red solely because that envelope is not at ``index - 1``.
    """
    rows = assemble_replay_and_verify(CORRELATION, _observed_tree(), DECLARED_TREE)

    assert all(row.replay_green for row in rows), [
        (row.hop, row.replay_detail) for row in rows if not row.replay_green
    ]
    assert rows[3].parent_envelope_id == E0
    assert rows[3].replay_detail == ""
    assert all(row.verifier_verdict is EnumTierTwoVerdict.PASS for row in rows)


@pytest.mark.unit
def test_the_same_tree_graded_against_a_line_declaration_fails_the_branch() -> None:
    """The falsifier for the test above: the grading is not vacuous.

    Same observed evidence, a declaration that says the terminal hangs off the
    routing decision. The edge then genuinely does not close and the row is
    red — which is what makes the green above a statement about the
    DECLARATION and not about the replay having stopped checking.
    """
    rows = assemble_replay_and_verify(CORRELATION, _observed_tree(), DECLARED)

    assert [row.replay_green for row in rows] == [True, True, True, False]
    assert str(E2) in rows[3].replay_detail
    assert str(E0) in rows[3].replay_detail


@pytest.mark.unit
def test_a_wrong_parent_still_fails_under_the_tree_declaration() -> None:
    """The negative control. A parent that is not the declared hop's envelope.

    ``UNRELATED`` is not observed anywhere on this correlation, so no reading
    of the topology can make this edge close.
    """
    observed = (
        _hop(TOPICS[0], E0, None),
        _hop(TOPICS[1], E1, E0),
        _hop(TOPICS[2], E2, E1),
        _hop(TOPICS[3], E3, UNRELATED),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED_TREE)

    assert rows[3].replay_green is False
    assert str(UNRELATED) in rows[3].replay_detail
    # Not a verifier problem: the topology matched, the causal edge did not.
    assert rows[3].verifier_verdict is EnumTierTwoVerdict.PASS


@pytest.mark.unit
def test_a_hop_that_records_no_parent_fails_when_one_is_declared() -> None:
    """OMN-18419 AC1, at the grading tier.

    An absent ``parent_message_id`` is the checkable statement "this hop is a
    chain HEAD". A hop the declaration says is caused by another must not be
    able to assert that and pass — which is exactly what the live
    ``delegation-routing-request`` hop was doing.
    """
    observed = (
        _hop(TOPICS[0], E0, None),
        _hop(TOPICS[1], E1, None),
        _hop(TOPICS[2], E2, E1),
        _hop(TOPICS[3], E3, E0),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED_TREE)

    assert rows[1].replay_green is False
    assert "records no parent" in rows[1].replay_detail
    assert TOPICS[0] in rows[1].replay_detail


@pytest.mark.unit
def test_a_declared_head_that_records_a_parent_fails_the_replay() -> None:
    """The head's claim is checked, not assumed.

    Before OMN-18419 the head was green by construction — there was no
    predecessor to compare against — so a head carrying a parent went
    unreported.
    """
    observed = (
        _hop(TOPICS[0], E0, UNRELATED),
        _hop(TOPICS[1], E1, E0),
        _hop(TOPICS[2], E2, E1),
        _hop(TOPICS[3], E3, E0),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED_TREE)

    assert rows[0].replay_green is False
    assert "head" in rows[0].replay_detail


@pytest.mark.unit
def test_a_declared_parent_that_was_never_observed_fails_the_replay() -> None:
    """An edge cannot be re-derived from evidence that is not there."""
    observed = (
        _hop(TOPICS[0], E0, None),
        # TOPICS[1] never happened, so TOPICS[2]'s declared parent is absent.
        _hop(TOPICS[2], E2, E1),
        _hop(TOPICS[3], E3, E0),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED_TREE)

    assert rows[1].hop == TOPICS[2]
    assert rows[1].replay_green is False
    assert "not observed" in rows[1].replay_detail


@pytest.mark.unit
def test_missing_hop_is_absent_from_the_chain_and_never_silently_filled() -> None:
    """A hop that never happened must not appear as a row at all.

    OMN-16964 scope: "no gaps tolerated silently". The canary checks
    completeness by hop NAME against its own declared expected set, so a
    fabricated placeholder row would defeat that check outright.
    """
    observed = (
        _hop(TOPICS[0], E0, None),
        # TOPICS[1] never happened.
        _hop(TOPICS[2], E2, E0),
        _hop(TOPICS[3], E3, E2),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED)

    assert [row.hop for row in rows] == [TOPICS[0], TOPICS[2], TOPICS[3]]
    assert TOPICS[1] not in [row.hop for row in rows]
    # The out-of-order arrival is a topology violation the tier-2 verifier sees.
    assert rows[1].verifier_verdict is EnumTierTwoVerdict.FAIL
    assert rows[-1].verifier_verdict is EnumTierTwoVerdict.FAIL


@pytest.mark.unit
def test_broken_causal_link_fails_the_replay() -> None:
    """Replay is a re-derivation, not a copy of a stored flag.

    Hop 2's recorded parent disagrees with the envelope observed on its
    declared parent topic, so the chain did not reproduce and replay_green is
    false for that hop.
    """
    observed = (
        _hop(TOPICS[0], E0, None),
        _hop(TOPICS[1], E1, E0),
        _hop(TOPICS[2], E2, UNRELATED),
        _hop(TOPICS[3], E3, E2),
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
        _hop(TOPICS[0], E0, None),
        _hop(TOPICS[1], E1, E0, correlation=OTHER_CORRELATION),
        _hop(TOPICS[2], E2, E1),
        _hop(TOPICS[3], E3, E2),
    )

    rows = assemble_replay_and_verify(CORRELATION, observed, DECLARED)

    assert rows[1].replay_green is False
    assert "correlation" in rows[1].replay_detail.lower()


@pytest.mark.unit
def test_no_declaration_skips_tier_two_and_reds_tier_one_with_a_reason() -> None:
    """No declaration to check against yields SKIP, and no green anywhere.

    This is the member the word "honest" in the OMN-16025 gate text exists
    for. A verifier with nothing to check against has NOT passed — and after
    OMN-18419 neither has the replay, which now derives its expected parent
    from the declaration rather than from the preceding position. A check that
    could not run reports that it could not run.
    """
    rows = assemble_replay_and_verify(CORRELATION, _complete_chain(), ())

    assert len(rows) == 4
    assert all(row.verifier_verdict is EnumTierTwoVerdict.SKIP for row in rows)
    assert rows[-1].verifier_verdict is not EnumTierTwoVerdict.PASS
    assert all(row.verifier_detail != "" for row in rows)
    assert not any(row.replay_green for row in rows)
    assert all("no declared chain topology" in row.replay_detail for row in rows)


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
    # Tier 1 reads the same declaration: the two unreached hops have no parent
    # relation to re-derive, and say so.
    assert [row.replay_green for row in rows] == [True, True, False, False]


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


@pytest.mark.unit
def test_a_declared_hop_may_not_name_itself_as_its_own_cause() -> None:
    """The declaration cannot ask for an edge the transport would reject."""
    with pytest.raises(ValueError, match="its own parent"):
        ModelDeclaredChainHop(topic=TOPICS[0], parent=TOPICS[0])
