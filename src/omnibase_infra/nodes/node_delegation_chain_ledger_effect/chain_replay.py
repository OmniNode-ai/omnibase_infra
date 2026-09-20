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

from collections.abc import Sequence
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


def _verify_one_hop(
    index: int,
    hop: ModelObservedHop,
    declared_chain: Sequence[ModelDeclaredChainHop],
) -> tuple[EnumTierTwoVerdict, str]:
    """Tier 2: does this observation match the DECLARED topology at this position?

    Returns ``(verdict, detail)``. SKIP always carries a detail: a SKIP with no
    stated reason is indistinguishable from a check that was never wired, and
    telling those apart is the whole job.
    """
    if index >= len(declared_chain):
        return EnumTierTwoVerdict.SKIP, _NO_DECLARATION_DETAIL

    # OMN-18937: a hop may declare ALTERNATIVES -- the delegation terminal is
    # one hop observed on either the success or the failure topic. The
    # position is still one position; what widens is the set of topics that
    # position accepts. The FAIL detail names the whole accepted set, because
    # a message naming one topic when two were acceptable is a red verdict
    # that misreports what was expected.
    accepted = declared_chain[index].topics
    if hop.topic in accepted:
        return EnumTierTwoVerdict.PASS, ""

    expected = accepted[0] if len(accepted) == 1 else f"one of {list(accepted)!r}"
    return (
        EnumTierTwoVerdict.FAIL,
        (
            f"the declared chain has {expected if len(accepted) > 1 else repr(expected)} "
            f"at position {index}, but {hop.topic!r} was observed there"
        ),
    )


def assemble_replay_and_verify(
    correlation_id: UUID,
    observed: Sequence[ModelObservedHop],
    declared_chain: Sequence[ModelDeclaredChainHop],
) -> tuple[ModelLedgerChainRow, ...]:
    """Turn observed envelopes into replayed, tier-2-verified ledger rows.

    ``observed`` must already be in the order the envelopes were seen; this
    function does not reorder it, because the observed order IS the evidence
    and sorting it would erase a transposition that tier 2 exists to catch.

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
    rows: list[ModelLedgerChainRow] = []

    for index, hop in enumerate(observed):
        replay_green, replay_detail = _replay_one_hop(
            index, hop, observed, declared_chain, correlation_id
        )
        verdict, verifier_detail = _verify_one_hop(index, hop, declared_chain)

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
