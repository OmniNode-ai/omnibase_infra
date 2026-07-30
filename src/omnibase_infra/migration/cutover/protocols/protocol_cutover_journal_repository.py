# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Persistence port for durable cutover receipts and journal state."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from omnibase_infra.migration.cutover.models import (
    ModelCutoverFamilyContract,
    ModelCutoverFamilyState,
    ModelCutoverJournalEvent,
    ModelCutoverJournalRequest,
    ModelReverseDeltaProof,
    ModelRollbackDecision,
    ModelTransformationReceipt,
)


class ProtocolCutoverJournalRepository(Protocol):
    """Storage operations required by the cutover coordinator."""

    async def initialize(self) -> None:
        """Create the explicit proof/journal schema idempotently."""
        ...

    async def register_family(self, contract: ModelCutoverFamilyContract) -> None:
        """Register an immutable family contract or reject drift."""
        ...

    async def record_receipt(self, receipt: ModelTransformationReceipt) -> None:
        """Persist a receipt and block only its family on mismatch."""
        ...

    async def append_event(
        self,
        family_id: UUID,
        request: ModelCutoverJournalRequest,
    ) -> ModelCutoverJournalEvent:
        """Append one validated, hash-chained family event."""
        ...

    async def record_reverse_delta_proof(
        self,
        proof: ModelReverseDeltaProof,
    ) -> None:
        """Persist complete reverse-delta coverage for later journal attestation."""
        ...

    async def get_state(self, family_id: UUID) -> ModelCutoverFamilyState:
        """Read the durable family-local projection."""
        ...

    async def evaluate_direct_rollback(
        self,
        family_id: UUID,
    ) -> ModelRollbackDecision:
        """Return the mechanical direct-DSN rollback decision."""
        ...


__all__ = ["ProtocolCutoverJournalRepository"]
