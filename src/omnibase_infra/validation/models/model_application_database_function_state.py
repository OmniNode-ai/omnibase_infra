# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Catalog and audit state for one application-owned PostgreSQL function."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.validation.models.model_application_database_tenant_isolation_evidence import (
    ModelApplicationDatabaseTenantIsolationEvidence,
)


class ModelApplicationDatabaseFunctionState(BaseModel):
    """Security-relevant function state plus its independent audit proof."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    security_definer: bool
    search_path: tuple[str, ...] = ()
    public_execute: bool
    audit_id: str | None = Field(default=None, min_length=1)
    definition_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    audited_definition_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    tenant_isolation_evidence: (
        ModelApplicationDatabaseTenantIsolationEvidence | None
    ) = None

    @field_validator("search_path")
    @classmethod
    def validate_unique_search_path(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate path entries without blessing unsafe entries."""
        if len(set(values)) != len(values):
            raise ValueError("function search_path entries must be unique")
        return values


__all__ = ["ModelApplicationDatabaseFunctionState"]
