# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Behavioral tenant-isolation evidence for a tenant-facing SQL surface."""

from collections.abc import Mapping
from types import MappingProxyType
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class ModelApplicationDatabaseTenantIsolationEvidence(BaseModel):
    """Expected and observed results for two tenants plus denied bad contexts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_rows_by_tenant: Mapping[UUID, int] = Field(..., min_length=2)
    observed_rows_by_tenant: Mapping[UUID, int] = Field(..., min_length=2)
    unset_context_rows: int = Field(..., ge=0)
    malformed_context_denied: bool

    @field_validator("expected_rows_by_tenant", "observed_rows_by_tenant")
    @classmethod
    def validate_nonnegative_counts(
        cls,
        values: Mapping[UUID, int],
    ) -> Mapping[UUID, int]:
        """Reject invalid counts and freeze evidence against in-place mutation."""
        if any(value < 0 for value in values.values()):
            raise ValueError("tenant isolation row counts must be nonnegative")
        return MappingProxyType(dict(values))

    @field_serializer("expected_rows_by_tenant", "observed_rows_by_tenant")
    def serialize_row_counts(
        self,
        values: Mapping[UUID, int],
    ) -> dict[UUID, int]:
        """Restore the JSON/YAML mapping shape for immutable row counts."""
        return dict(values)


__all__ = ["ModelApplicationDatabaseTenantIsolationEvidence"]
