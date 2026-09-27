# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Redacted observation from a mandatory current source reread."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_sim_archive_selected_record import (
    CanonicalSha256,
)


class ModelSimArchiveFreshnessObservation(BaseModel):
    """Caller-constructible freshness claim, not a trusted source receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner_binding_sha256: CanonicalSha256
    source_scan_sha256: CanonicalSha256
    head_envelope_sha256: CanonicalSha256
    exact_owner_rows: int = Field(ge=0, strict=True)
    head_count: int = Field(ge=0, strict=True)
    foreign_explicit_tenant_count: int = Field(ge=0, strict=True)
    conflicting_envelope_count: int = Field(ge=0, strict=True)


__all__ = ["ModelSimArchiveFreshnessObservation"]
