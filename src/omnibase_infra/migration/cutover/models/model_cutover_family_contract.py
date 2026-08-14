# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract for one independently stoppable database relation family."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.migration.cutover.enums import (
    EnumCutoverFamilyKind,
    EnumPostCheckpointMode,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelCutoverFamilyContract(BaseModel):
    """Declare continuity and rollback semantics before cutover work begins.

    Binding fields are secret-free topology references, never DSNs.  A family
    must choose reverse-delta or forward-fix-only up front.  A dual-write window
    is disabled by default and, when explicitly enabled, is strictly bounded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    family_id: UUID
    family_key: str = Field(
        ...,
        min_length=3,
        pattern=r"^[a-z0-9][a-z0-9._-]+$",
        description="Stable semantic key for the coherent relation family",
    )
    family_kind: EnumCutoverFamilyKind
    source_binding_ref: str = Field(..., min_length=1, max_length=200)
    target_binding_ref: str = Field(..., min_length=1, max_length=200)
    source_evidence_contract_hash: str = Field(..., pattern=_SHA256_PATTERN)
    target_evidence_contract_hash: str = Field(..., pattern=_SHA256_PATTERN)
    post_checkpoint_mode: EnumPostCheckpointMode
    reverse_delta_contract_ref: str = Field(default="", max_length=300)
    forward_fix_runbook_ref: str = Field(default="", max_length=300)
    dual_write_max_seconds: int = Field(
        default=0,
        ge=0,
        le=3600,
        description="Zero disables dual-write; non-zero is a hard bounded window",
    )
    observation_window_seconds: int = Field(..., ge=1, le=604800)

    @model_validator(mode="after")
    def _validate_semantics(self) -> ModelCutoverFamilyContract:
        if self.source_binding_ref == self.target_binding_ref:
            raise ValueError("source and target binding refs must be distinct")

        if "://" in self.source_binding_ref or "://" in self.target_binding_ref:
            raise ValueError("binding refs must be secret-free names, not DSNs")

        if self.post_checkpoint_mode is EnumPostCheckpointMode.REVERSE_DELTA:
            if not self.reverse_delta_contract_ref:
                raise ValueError(
                    "reverse_delta mode requires reverse_delta_contract_ref"
                )
            if self.forward_fix_runbook_ref:
                raise ValueError(
                    "reverse_delta mode cannot declare a forward-fix runbook"
                )
        else:
            if not self.forward_fix_runbook_ref:
                raise ValueError(
                    "forward_fix_only mode requires forward_fix_runbook_ref"
                )
            if self.reverse_delta_contract_ref:
                raise ValueError(
                    "forward_fix_only mode cannot advertise reverse-delta proof"
                )
        return self


__all__ = ["ModelCutoverFamilyContract"]
