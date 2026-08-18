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

    async def record_receipt(
        self, receipt: ModelTransformationReceipt
    ) -> ModelTransformationReceipt:
        """Persist a receipt and block only its family on mismatch.

        Idempotent on ``(family_id, idempotency_key)``: returns the original
        persisted receipt on a retried call with identical content.
        """
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
        """Persist complete reverse-delta coverage for later journal attestation.

        Requires every entry's inverse artifact and the proof's behavioral
        readback ref to dereference to an artifact durably registered via
        ``register_reverse_delta_artifact`` and hash-bound to its declared
        before-image.
        """
        ...

    async def register_reverse_delta_artifact(
        self,
        family_id: UUID,
        artifact_ref: str,
        content: dict[str, object],
    ) -> str:
        """Durably register one dereferenceable reverse-delta artifact."""
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
