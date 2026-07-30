# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Immutable, transformation-aware family receipt."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.migration.cutover.enums import (
    EnumReceiptDimension,
    EnumReceiptStatus,
)
from omnibase_infra.migration.cutover.models.model_cutover_continuity_evidence import (
    ModelCutoverContinuityEvidence,
)
from omnibase_infra.migration.cutover.models.model_receipt_check import (
    ModelReceiptCheck,
)
from omnibase_infra.migration.cutover.models.model_transformation_evidence import (
    ModelTransformationEvidence,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def calculate_transformation_receipt_hash(
    *,
    receipt_id: UUID,
    family_id: UUID,
    family_contract_hash: str,
    generated_at: datetime,
    source: ModelTransformationEvidence,
    target: ModelTransformationEvidence,
    continuity: ModelCutoverContinuityEvidence,
    checks: tuple[ModelReceiptCheck, ...],
    status: EnumReceiptStatus,
) -> str:
    """Return the canonical binding for every immutable receipt field."""
    body = {
        "receipt_id": str(receipt_id),
        "family_id": str(family_id),
        "family_contract_hash": family_contract_hash,
        "generated_at": generated_at.isoformat(),
        "source": source.model_dump(mode="json"),
        "target": target.model_dump(mode="json"),
        "continuity": continuity.model_dump(mode="json"),
        "checks": [check.model_dump(mode="json") for check in checks],
        "status": status.value,
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ModelTransformationReceipt(BaseModel):
    """Durable receipt binding every comparison and continuity proof."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    receipt_id: UUID
    family_id: UUID
    family_contract_hash: str = Field(..., pattern=_SHA256_PATTERN)
    generated_at: datetime
    source: ModelTransformationEvidence
    target: ModelTransformationEvidence
    continuity: ModelCutoverContinuityEvidence
    checks: tuple[ModelReceiptCheck, ...]
    status: EnumReceiptStatus
    receipt_hash: str = Field(..., pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _checks_are_complete_and_truthful(self) -> ModelTransformationReceipt:
        dimensions = [check.dimension for check in self.checks]
        if dimensions != list(EnumReceiptDimension):
            raise ValueError(
                "receipt checks must contain every dimension exactly once in "
                "canonical enum order"
            )
        passed = all(check.passed for check in self.checks)
        if passed != (self.status is EnumReceiptStatus.PASS):
            raise ValueError("receipt status must match the complete check set")
        if self.generated_at.tzinfo is None:
            raise ValueError("receipt timestamp must be timezone-aware")
        expected_hash = calculate_transformation_receipt_hash(
            receipt_id=self.receipt_id,
            family_id=self.family_id,
            family_contract_hash=self.family_contract_hash,
            generated_at=self.generated_at,
            source=self.source,
            target=self.target,
            continuity=self.continuity,
            checks=self.checks,
            status=self.status,
        )
        if self.receipt_hash != expected_hash:
            raise ValueError("receipt hash does not bind the complete receipt")
        return self


__all__ = ["ModelTransformationReceipt", "calculate_transformation_receipt_hash"]
