# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of one correlation-scoped ledger-chain write."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.enum_tier_two_verdict import (
    EnumTierTwoVerdict,
)


class ModelLedgerChainWriteResult(BaseModel):
    """Summary returned after all observed rows are persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    rows_written: int = Field(ge=0)
    chain_complete: bool
    replay_green: bool
    verifier_verdict: EnumTierTwoVerdict


__all__ = ["ModelLedgerChainWriteResult"]
