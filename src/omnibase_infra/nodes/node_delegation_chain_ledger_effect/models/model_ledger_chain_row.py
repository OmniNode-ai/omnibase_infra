# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One row of ``ledger_chain`` — a hop plus what was DERIVED about it.

The column set here is not a design choice made in this module: it is the
interface ``node_chain_canary_effect`` already reads, verbatim, in
``_replay_ledger_chain_via_asyncpg``::

    SELECT hop, replay_green, verifier_verdict FROM ledger_chain
    WHERE correlation_id = $1 ORDER BY hop_index

Those five columns are therefore a contract with a merged consumer (OMN-16964
PRs #3072 / #3079) and may not be renamed here. The remaining columns are
evidence the consumer does not read but a human diagnosing a red canary does.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.enum_tier_two_verdict import (
    EnumTierTwoVerdict,
)


class ModelLedgerChainRow(BaseModel):
    """A replayed, tier-2-verified hop, ready to persist."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        description=(
            "The chain this hop belongs to. Typed here; persisted as TEXT — "
            "see the note on the column type in migration 104."
        )
    )
    hop_index: int = Field(
        ge=0,
        description=(
            "Position in the OBSERVED order. The consumer orders by this "
            "column, so it must be dense and ascending across a chain's rows."
        ),
    )
    hop: str = Field(
        description=(
            "The hop's name, which is the topic it was observed on. The canary "
            "checks completeness by comparing these names against its own "
            "declared expected set, so a hop that did not happen must be "
            "ABSENT here rather than present with a false verdict."
        )
    )
    replay_green: bool = Field(
        description=(
            "Did this hop's causal linkage RE-DERIVE from the recorded "
            "evidence? Not a flag copied from the envelope: the expected "
            "parent is recomputed from the preceding observed hop and compared."
        )
    )
    verifier_verdict: EnumTierTwoVerdict = Field(
        description="The tier-2 verifier's own word. SKIP is never a pass."
    )
    observed_topic: str = Field(
        description="The raw topic, kept separate from `hop` so a future hop-naming change stays diagnosable."
    )
    envelope_id: UUID | None = Field(
        default=None, description="This hop's recorded envelope id."
    )
    parent_envelope_id: UUID | None = Field(
        default=None, description="This hop's recorded parent envelope id."
    )
    replay_detail: str = Field(
        default="",
        description=(
            "Why the replay was not green. EMPTY when it was. Never carries a "
            "payload, a credential, or a connection string."
        ),
    )
    verifier_detail: str = Field(
        default="",
        description=(
            "Why the tier-2 verifier said what it said. Populated on SKIP and "
            "FAIL — a SKIP with no stated reason is indistinguishable from a "
            "check that was never wired, which is the defect, not the report."
        ),
    )


__all__ = ["ModelLedgerChainRow"]
