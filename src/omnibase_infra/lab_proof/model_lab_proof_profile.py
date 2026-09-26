# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""One registry row: the proof profile of one repository.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.lab_proof.enum_lab_proof_exempt_class import (
    EnumLabProofExemptClass,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_variant import (
    ModelLabProofProfileVariant,
)


class ModelLabProofProfile(BaseModel):
    """A repository's proof profile (plan section 2).

    ``enforce`` is refused while true: the bar record that would justify it
    (five PASS receipts on distinct real PRs plus one negative-control FAIL for
    this ``profile_version``) is read from the receipt projection that T2
    (OMN-19566) and the gate (OMN-19584) build. Until then no row can claim it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_key: str = Field(pattern=r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_-]*$")
    profile_version: int = Field(ge=1)
    repo: str = Field(pattern=r"^OmniNode-ai/[A-Za-z0-9_.-]+$")
    exempt_classes: tuple[EnumLabProofExemptClass, ...] = ()
    docs_globs: tuple[str, ...] = ()
    variants: tuple[ModelLabProofProfileVariant, ...] = Field(min_length=1)
    enforce: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> ModelLabProofProfile:
        ids = [variant.variant_key for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"profile {self.profile_key}: duplicate variant_key in {ids}"
            )
        if len(self.exempt_classes) != len(set(self.exempt_classes)):
            raise ValueError(f"profile {self.profile_key}: duplicate exempt class")
        if (
            EnumLabProofExemptClass.DOCS_ONLY in self.exempt_classes
            and not self.docs_globs
        ):
            raise ValueError(
                f"profile {self.profile_key}: exempt class docs_only needs docs_globs"
            )
        if self.enforce:
            raise ValueError(
                f"profile {self.profile_key}: enforce is true but no bar record exists "
                "for profile_version "
                f"{self.profile_version}; the bar (five PASS receipts on distinct real "
                "PRs plus one negative-control FAIL) is read from the receipt "
                "projection (OMN-19566, OMN-19584), not declared here"
            )
        return self


__all__ = ["ModelLabProofProfile"]
