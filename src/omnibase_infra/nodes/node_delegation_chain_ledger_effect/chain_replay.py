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
envelopes, compared against what was recorded. For each hop after the head,
the expected parent envelope id is recomputed from the preceding observed hop
and compared to the parent the envelope actually carries; the correlation id
on each hop is compared to the chain's own. A hop replays green only when both
re-derivations reproduce.

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


def _replay_one_hop(
    index: int,
    hop: ModelObservedHop,
    previous: ModelObservedHop | None,
    correlation_id: UUID,
) -> tuple[bool, str]:
    """Re-derive this hop's linkage and compare it to what was recorded.

    Returns ``(replay_green, detail)``. ``detail`` is EMPTY exactly when the
    replay was green — a green carrying an explanation would suggest the check
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

    if previous is None:
        # The head of the chain has no predecessor. There is nothing to
        # re-derive, and nothing is claimed beyond the two checks above.
        return True, ""

    expected_parent = previous.envelope_id
    if hop.parent_envelope_id != expected_parent:
        return (
            False,
            (
                f"parent re-derives to {expected_parent!r} from the preceding "
                f"hop {previous.topic!r}, but this hop records "
                f"{hop.parent_envelope_id!r} — the causal link does not close"
            ),
        )

    return True, ""


def _verify_one_hop(
    index: int,
    hop: ModelObservedHop,
    declared_chain: Sequence[str],
) -> tuple[EnumTierTwoVerdict, str]:
    """Tier 2: does this observation match the DECLARED topology at this position?

    Returns ``(verdict, detail)``. SKIP always carries a detail: a SKIP with no
    stated reason is indistinguishable from a check that was never wired, and
    telling those apart is the whole job.
    """
    if index >= len(declared_chain):
        return EnumTierTwoVerdict.SKIP, _NO_DECLARATION_DETAIL

    expected_topic = declared_chain[index]
    if hop.topic == expected_topic:
        return EnumTierTwoVerdict.PASS, ""

    return (
        EnumTierTwoVerdict.FAIL,
        (
            f"the declared chain has {expected_topic!r} at position {index}, "
            f"but {hop.topic!r} was observed there"
        ),
    )


def assemble_replay_and_verify(
    correlation_id: UUID,
    observed: Sequence[ModelObservedHop],
    declared_chain: Sequence[str],
) -> tuple[ModelLedgerChainRow, ...]:
    """Turn observed envelopes into replayed, tier-2-verified ledger rows.

    ``observed`` must already be in the order the envelopes were seen; this
    function does not reorder it, because the observed order IS the evidence
    and sorting it would erase a transposition that tier 2 exists to catch.

    ``declared_chain`` is the tier-2 authority: the ordered topics the node
    contracts say a delegation traverses. An EMPTY declaration is legitimate
    and yields SKIP on every row — the canary then reports VERIFIER_SKIPPED,
    which is red. It is never treated as "nothing to check, therefore fine".

    An empty ``observed`` returns zero rows rather than any synthetic row. The
    canary computes ``all(row.replay_green for row in rows)``, which is
    vacuously TRUE over an empty sequence, so it guards on the row count
    separately; a single fabricated green row here would defeat that guard.
    """
    rows: list[ModelLedgerChainRow] = []
    previous: ModelObservedHop | None = None

    for index, hop in enumerate(observed):
        replay_green, replay_detail = _replay_one_hop(
            index, hop, previous, correlation_id
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
        previous = hop

    return tuple(rows)


__all__ = ["assemble_replay_and_verify"]
