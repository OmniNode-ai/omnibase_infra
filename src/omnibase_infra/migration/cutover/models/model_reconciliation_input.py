# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed bundle of everything one reconciliation call needs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.cutover.models.model_cutover_continuity_evidence import (
    ModelCutoverContinuityEvidence,
)
from omnibase_infra.migration.cutover.models.model_cutover_family_contract import (
    ModelCutoverFamilyContract,
)
from omnibase_infra.migration.cutover.models.model_transformation_evidence import (
    ModelTransformationEvidence,
)

_IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"


class ModelReconciliationInput(BaseModel):
    """Bundle a receipt build call's inputs into a single typed request."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract: ModelCutoverFamilyContract
    source: ModelTransformationEvidence
    target: ModelTransformationEvidence
    continuity: ModelCutoverContinuityEvidence
    idempotency_key: str = Field(
        ..., min_length=1, max_length=200, pattern=_IDEMPOTENCY_KEY_PATTERN
    )


__all__ = ["ModelReconciliationInput"]
