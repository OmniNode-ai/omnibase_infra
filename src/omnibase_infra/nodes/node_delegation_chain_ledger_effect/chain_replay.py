# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Assemble, replay and tier-2 verify one delegation chain (OMN-16964).

This module is the half of chain-canary link 5 that OMN-16025 actually asks
for. The canary itself is a READER: ``_replay_ledger_chain_via_asyncpg``
selects ``hop, replay_green, verifier_verdict`` out of ``ledger_chain`` and
classifies what it finds. It computes nothing. Whoever writes those two derived
columns is where "complete ledger chain + replay green through an HONEST
tier-2 verifier" either happens or is faked, and this module is that writer's
brain.

WHAT "REPLAY" MEANS HERE, STATED PLAINLY
----------------------------------------
It is a re-derivation of the chain's CAUSAL LINKAGE from the recorded
envelopes, compared against what was recorded. For each hop, the expected
parent is resolved from the DECLARED topology -- the hop whose consumption the
contract says causes this one -- and the envelope id observed on that declared
parent topic is compared to the parent the envelope actually carries; the
correlation id on each hop is compared to the chain's own. A hop replays green
only when both re-derivations reproduce.

THE DECLARED TOPOLOGY IS A TREE, AND THAT IS NOT A DETAIL (OMN-18419)
--------------------------------------------------------------------
Until OMN-18419 the expected parent was recomputed from ``observed[index - 1]``
-- the hop that happened to precede this one in time. That is correct for a
LINE and wrong for anything else, and the chain this module grades is a TREE.
Measured read-only on the .201 compose dev lane, correlation
``41235987-425c-481b-b2e3-8970083ce512`` (chain-canary run 35037024216):
``delegate-skill-completed`` records the ``delegate-skill`` COMMAND envelope as
its parent, because it is a consequence of consuming that command and not of
the routing decision that preceded it in time. The recorded edge was right; the
positional re-derivation was wrong; a correct chain graded red.

So the parent relation is now DECLARED, one entry per hop
(``ModelDeclaredChainHop``), and the replay grades each recorded edge against
the hop the declaration names. Nothing about this particular canary is
special-cased: a line is a tree whose every parent is the preceding hop, and it
grades identically under both readings.

A consequence worth stating because it changes a verdict: the replay now needs
the declaration. With no declaration reaching a hop there is nothing to
re-derive FROM, so that hop replays RED with a stated reason -- it does not
fall back to the positional guess and it does not report green. This is the
same rule tier 2 already held: a check that could not run has not passed.

It is deliberately NOT a re-execution of the delegation's inference. One hop of
this chain calls a language model, which is not a deterministic function, so a
module claiming to have re-executed and reproduced it would be lying. Saying
what the replay covers — and, by omission, what it does not — is the honest
form. The linkage is worth checking on its own terms: a correlation id that
changes mid-chain is exactly the OMN-16931 defect, which reported a green
terminal for a delegation whose identity had been replaced.

WHY THE TIER-2 VERIFIER IS A SECOND TIER
----------------------------------------
Tier 1 (the replay) asks whether the chain that happened is internally
coherent. Tier 2 asks a different question with an independent authority:
does the chain that happened match the topology the node contracts DECLARE
must happen? A chain can be perfectly self-consistent and still be the wrong
chain — a hop skipped, two hops transposed — and tier 1 cannot see that
because it only ever compares the evidence to itself.

Tier 2 needs a declaration to check against. When it has none reaching a hop,
it returns SKIP, and SKIP is not PASS. That is the entire point of the link:
OMN-16773 recorded the same shape as "an unconfigured check reports itself —
SKIPPED_NOT_CONFIGURED is not CLEAN", and OMN-16931 found a verdict derived
from a claim rather than from evidence. A verifier that returned PASS for a
hop it had no declaration for would re-create both defects at once.

NO GAP IS FILLED
----------------
A hop that was never observed produces NO ROW. It is tempting to emit a
placeholder row carrying ``replay_green = false`` so the gap is visible, and
that is wrong: the canary checks completeness by comparing hop NAMES against
its own declared ``expected_ledger_hops``, so a placeholder bearing the missing
hop's name would satisfy the completeness check and convert a
CHAIN_INCOMPLETE into a REPLAY_FAILED. Absence is the honest encoding of
absence.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from uuid import UUID

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.enum_tier_two_verdict import (
    EnumTierTwoVerdict,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_ledger_chain_row import (
    ModelLedgerChainRow,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_hop import (
    ModelObservedHop,
)

_NO_DECLARATION_DETAIL = (
    "no declared chain topology reached this hop, so no tier-2 check ran; "
    "SKIP is not PASS"
)

_NO_DECLARED_PARENT_DETAIL = (
    "no declared chain topology names this hop, so there is no parent "
    "relation to re-derive the causal edge from; a check that could not run "
    "has not passed"
)


def _declared_hop_for(
    topic: str, declared_chain: Sequence[ModelDeclaredChainHop]
) -> ModelDeclaredChainHop | None:
    """Return the declaration for ``topic``, or None when none names it.

    Resolved by TOPIC, not by position. Position is tier 2's question ("is the
    observed chain the declared chain?"), and answering it twice from two
    places would let the two tiers disagree about what a hop IS. Tier 1 asks
    only whether the recorded causal edge closes, which is a statement about
    this hop's identity and not about where it sits.

    OMN-18937: matched against every name the hop may be observed on, not
    only its canonical one. Without this the delegation FAILURE terminal has
    no declaration to grade against and replays red as an undeclared hop.
    """
    for candidate in declared_chain:
        if topic in candidate.topics:
            return candidate
    return None


def _replay_one_hop(
    index: int,
    hop: ModelObservedHop,
    observed: Sequence[ModelObservedHop],
    declared_chain: Sequence[ModelDeclaredChainHop],
    correlation_id: UUID,
) -> tuple[bool, str]:
    """Re-derive this hop's linkage and compare it to what was recorded.

    Returns ``(replay_green, detail)``. ``detail`` is EMPTY exactly when the
    replay was green -- a green carrying an explanation would suggest the check
    was hedged.
    """
    if hop.correlation_id is not None and hop.correlation_id != correlation_id:
        return (
            False,
            (
                f"correlation id on this hop is {hop.correlation_id!r}, but the "
                f"chain being replayed is {correlation_id!r} — the delegation's "
                "identity changed mid-chain"
            ),
        )

    declared = _declared_hop_for(hop.topic, declared_chain)
    if declared is None:
        return False, _NO_DECLARED_PARENT_DETAIL

    if declared.parent is None:
        # The declaration says this hop is the chain HEAD. That is a claim with
        # a falsifier, not a free pass: a head that records a parent is either a
        # mis-declared topology or a hop that is not actually the head.
        if hop.parent_envelope_id is not None:
            return (
                False,
                (
                    f"the declared topology makes {hop.topic!r} the chain head, "
                    f"but this hop records parent {hop.parent_envelope_id!r} — "
                    "a head has no cause to point at"
                ),
            )
        return True, ""

    # A redelivery is projected as a second row carrying the SAME deterministic
    # envelope id, so the declared parent topic can legitimately appear more
    # than once. Matching against ANY of them is not a loosening: the edge is
    # checked against a concrete observed envelope id either way.
    # OMN-18937: the parent is CITED by its canonical topic, but it may have
    # been OBSERVED on one of its alternatives. Resolving the citation to the
    # declared hop first, then matching observations against every name that
    # hop answers to, keeps `alternatives` usable on a hop that is somebody's
    # parent. Citing a hop that is not declared at all is already refused at
    # parse time, so the fallback below is unreachable in a parsed topology
    # and exists so this function is total on its own.
    declared_parent = _declared_hop_for(declared.parent, declared_chain)
    parent_names = (
        declared_parent.topics if declared_parent is not None else (declared.parent,)
    )
    candidates = tuple(
        candidate.envelope_id
        for candidate in observed
        if candidate.topic in parent_names
    )
    if not candidates:
        return (
            False,
            (
                f"the declared parent of {hop.topic!r} is {declared.parent!r}, "
                "which was not observed on this correlation — the causal edge "
                "cannot be re-derived from evidence that is not there"
            ),
        )

    if hop.parent_envelope_id is None:
        return (
            False,
            (
                f"this hop records no parent, which is the checkable statement "
                f"that it is a chain HEAD, but the declared topology says "
                f"{hop.topic!r} is caused by {declared.parent!r} "
                f"(observed as {candidates[0]!r}) — the causal edge does not close"
            ),
        )

    if hop.parent_envelope_id not in candidates:
        return (
            False,
            (
                f"the declared parent {declared.parent!r} was observed as "
                f"{list(candidates)!r}, but this hop records "
                f"{hop.parent_envelope_id!r} — the causal link does not close"
            ),
        )

    return True, ""


def _declared_index_for(
    topic: str, declared_chain: Sequence[ModelDeclaredChainHop]
) -> int | None:
    """The position of the declared hop this topic belongs to, or None.

    Resolved over every name a hop answers to (OMN-18937 alternatives), so a
    terminal observed on its failure topic resolves to the same declared hop
    as one observed on its success topic.
    """
    for position, candidate in enumerate(declared_chain):
        if topic in candidate.topics:
            return position
    return None


def _verify_one_hop(
    hop: ModelObservedHop,
    declared_chain: Sequence[ModelDeclaredChainHop],
    first_seen_declared_indices: Set[int],
) -> tuple[EnumTierTwoVerdict, str]:
    """Tier 2: is this observation a declared hop, appearing in declared order?

    Returns ``(verdict, detail)``. SKIP always carries a detail: a SKIP with no
    stated reason is indistinguishable from a check that was never wired, and
    telling those apart is the whole job.
    """
    if not declared_chain:
        return EnumTierTwoVerdict.SKIP, _NO_DECLARATION_DETAIL

    # OMN-18916: matched by declared-hop IDENTITY, not by position.
    #
    # Position was never the right key. A declared hop legitimately occurs
    # MORE THAN ONCE -- the attempts ladder climbs a rung and issues a second
    # routing request with its own envelope and its own correct parent -- so
    # the observed sequence is routinely longer than the declaration. Graded
    # positionally, every hop after the first repeat is compared against the
    # wrong declaration and a causally correct chain grades red. Measured on
    # the .201 dev lane: a SUCCESSFUL delegation produced 11 observed hops
    # against a 5-hop declaration, and 31 of 165 chains exceeded five hops.
    #
    # OMN-18937's ALTERNATIVES still apply: one hop may answer to more than
    # one topic, which is how the terminal is either the success or the
    # failure event.
    declared_index = _declared_index_for(hop.topic, declared_chain)
    if declared_index is None:
        known = sorted({topic for entry in declared_chain for topic in entry.topics})
        return (
            EnumTierTwoVerdict.FAIL,
            (
                f"{hop.topic!r} is not a topic the declared chain names "
                f"(declared: {known!r}) -- allowing a declared hop to repeat "
                "is not allowing an undeclared hop to appear"
            ),
        )

    # ORDER still matters, and this is where it is checked. Repeats are free,
    # but the FIRST time each declared hop appears it must appear in declared
    # order. Without this, tier 2 degrades into "is this topic known?" and
    # stops being the check that catches a transposed chain -- which is the
    # one thing tier 1 cannot catch on its own, because a transposition can
    # still carry individually resolvable parent edges.
    if declared_index in first_seen_declared_indices:
        return EnumTierTwoVerdict.PASS, ""

    out_of_order = [i for i in first_seen_declared_indices if i > declared_index]
    if out_of_order:
        later = sorted({declared_chain[i].topic for i in out_of_order})
        return (
            EnumTierTwoVerdict.FAIL,
            (
                f"{hop.topic!r} is declared at position {declared_index}, but "
                f"{later!r} were already observed first -- the declared hops "
                "appeared out of order, which a repeat allowance does not excuse"
            ),
        )

    return EnumTierTwoVerdict.PASS, ""


def assemble_replay_and_verify(
    correlation_id: UUID,
    observed: Sequence[ModelObservedHop],
    declared_chain: Sequence[ModelDeclaredChainHop],
) -> tuple[ModelLedgerChainRow, ...]:
    """Turn observed envelopes into replayed, tier-2-verified ledger rows.

    ``observed`` must already be in the order the envelopes were seen; this
    function does not reorder it, because the observed order IS the evidence
    and sorting it would erase a transposition that tier 2 exists to catch.
    It does REMOVE exact redeliveries (OMN-18916) -- the same envelope id
    seen more than once -- which drops no evidence, because a second copy of
    one envelope says nothing the first did not.

    ``declared_chain`` is the authority for BOTH tiers: the ordered hops the
    node contracts say a delegation traverses, each naming the declared topic
    that causes it. Tier 2 reads the ORDER; tier 1 reads the PARENT RELATION.
    An EMPTY declaration is legitimate and yields SKIP on every tier-2 row and
    a stated-reason RED on every tier-1 row — the canary then reports
    VERIFIER_SKIPPED, which is red. It is never treated as "nothing to check,
    therefore fine".

    An empty ``observed`` returns zero rows rather than any synthetic row. The
    canary computes ``all(row.replay_green for row in rows)``, which is
    vacuously TRUE over an empty sequence, so it guards on the row count
    separately; a single fabricated green row here would defeat that guard.
    """
    # OMN-18916: collapse a REDELIVERY before grading anything.
    #
    # A redelivery is the identical envelope arriving twice -- measured on the
    # lane as one envelope id at two adjacent Kafka offsets. It is one hop
    # that was projected twice, so recording it as two hops invents a hop
    # that never happened and, graded positionally, shifted every later one.
    #
    # Keyed on the ENVELOPE, never on the topic. A retry carries a NEW
    # envelope id for the same declared hop and is a real second attempt:
    # collapsing by topic would erase it, which is the opposite error and
    # equally wrong. First occurrence wins, so the surviving row keeps the
    # earliest offset's evidence.
    deduplicated: list[ModelObservedHop] = []
    seen_envelope_ids: set[UUID] = set()
    for candidate in observed:
        if candidate.envelope_id in seen_envelope_ids:
            continue
        seen_envelope_ids.add(candidate.envelope_id)
        deduplicated.append(candidate)
    observed = tuple(deduplicated)

    rows: list[ModelLedgerChainRow] = []
    # Which declared hops have already had their FIRST occurrence. Tier 2
    # reads this to allow repeats while still refusing a transposition.
    first_seen_declared_indices: set[int] = set()

    for index, hop in enumerate(observed):
        replay_green, replay_detail = _replay_one_hop(
            index, hop, observed, declared_chain, correlation_id
        )
        verdict, verifier_detail = _verify_one_hop(
            hop, declared_chain, first_seen_declared_indices
        )
        declared_index = _declared_index_for(hop.topic, declared_chain)
        if declared_index is not None:
            first_seen_declared_indices.add(declared_index)

        rows.append(
            ModelLedgerChainRow(
                correlation_id=correlation_id,
                hop_index=index,
                hop=hop.topic,
                replay_green=replay_green,
                verifier_verdict=verdict,
                observed_topic=hop.topic,
                envelope_id=hop.envelope_id,
                parent_envelope_id=hop.parent_envelope_id,
                replay_detail=replay_detail,
                verifier_detail=verifier_detail,
            )
        )

    return tuple(rows)


__all__ = ["assemble_replay_and_verify"]
