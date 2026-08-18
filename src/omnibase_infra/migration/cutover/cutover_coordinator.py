# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Application service joining pure receipts to the durable family journal."""

from __future__ import annotations

from uuid import UUID

from omnibase_infra.migration.cutover.models import (
    ModelCutoverContinuityEvidence,
    ModelCutoverFamilyContract,
    ModelCutoverJournalEvent,
    ModelCutoverJournalRequest,
    ModelReconciliationInput,
    ModelReverseDeltaProof,
    ModelRollbackDecision,
    ModelTransformationEvidence,
    ModelTransformationReceipt,
)
from omnibase_infra.migration.cutover.protocols import (
    ProtocolCutoverJournalRepository,
)
from omnibase_infra.migration.cutover.transformation_receipt_builder import (
    TransformationReceiptBuilder,
)


class CutoverCoordinator:
    """Coordinate receipt persistence and guarded journal transitions."""

    def __init__(
        self,
        repository: ProtocolCutoverJournalRepository,
        receipt_service: TransformationReceiptBuilder | None = None,
    ) -> None:
        self._repository = repository
        self._receipt_service = receipt_service or TransformationReceiptBuilder()

    async def register_family(self, contract: ModelCutoverFamilyContract) -> None:
        """Persist a family's immutable cutover and rollback contract."""
        await self._repository.register_family(contract)

    async def reconcile(
        self,
        contract: ModelCutoverFamilyContract,
        source: ModelTransformationEvidence,
        target: ModelTransformationEvidence,
        continuity: ModelCutoverContinuityEvidence,
        idempotency_key: str,
    ) -> ModelTransformationReceipt:
        """Build and durably persist a complete PASS-or-FAIL receipt.

        Idempotent on ``idempotency_key``: retrying with the same key and
        identical inputs returns the original persisted receipt.
        """
        receipt = self._receipt_service.build(
            ModelReconciliationInput(
                contract=contract,
                source=source,
                target=target,
                continuity=continuity,
                idempotency_key=idempotency_key,
            )
        )
        return await self._repository.record_receipt(receipt)

    async def append(
        self,
        family_id: UUID,
        request: ModelCutoverJournalRequest,
    ) -> ModelCutoverJournalEvent:
        """Append a family event through the repository's state machine."""
        return await self._repository.append_event(family_id, request)

    async def record_reverse_delta(
        self,
        proof: ModelReverseDeltaProof,
    ) -> None:
        """Persist complete reverse-delta coverage before attesting it."""
        await self._repository.record_reverse_delta_proof(proof)

    async def register_reverse_delta_artifact(
        self,
        family_id: UUID,
        artifact_ref: str,
        content: dict[str, object],
    ) -> str:
        """Durably register one dereferenceable reverse-delta artifact."""
        return await self._repository.register_reverse_delta_artifact(
            family_id, artifact_ref, content
        )

    async def evaluate_direct_rollback(
        self,
        family_id: UUID,
    ) -> ModelRollbackDecision:
        """Return the fail-closed rollback decision for one family."""
        return await self._repository.evaluate_direct_rollback(family_id)


__all__ = ["CutoverCoordinator"]
