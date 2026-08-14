# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live control-plane evidence for an exceptional tenant identity root."""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.validation.enums.enum_application_database_identity_root_operation import (
    EnumApplicationDatabaseIdentityRootOperation,
)


class ModelApplicationDatabaseIdentityRootControlState(BaseModel):
    """Bind declared root operations to one audited, non-runtime role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    role_can_login: bool
    role_superuser: bool
    role_bypass_rls: bool
    runtime_membership_principals: tuple[str, ...]
    runtime_set_role_denied_principals: tuple[str, ...]
    declared_operations: tuple[EnumApplicationDatabaseIdentityRootOperation, ...]
    observed_operations: tuple[EnumApplicationDatabaseIdentityRootOperation, ...]
    behavioral_proof_ids: tuple[str, ...]

    @field_validator(
        "runtime_membership_principals",
        "runtime_set_role_denied_principals",
    )
    @classmethod
    def validate_runtime_principal_census(
        cls,
        principals: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Keep role evidence deterministic and constrained to SQL identifiers."""
        invalid = [
            principal
            for principal in principals
            if re.fullmatch(r"[a-z_][a-z0-9_]*", principal) is None
        ]
        if invalid:
            raise ValueError(
                f"identity-root runtime principal names are invalid: {invalid!r}"
            )
        if tuple(sorted(principals)) != principals:
            raise ValueError(
                "identity-root runtime principal evidence must be canonically sorted"
            )
        return principals

    @model_validator(mode="after")
    def validate_exact_proof_shape(
        self,
    ) -> "ModelApplicationDatabaseIdentityRootControlState":
        """Every distinct observed operation carries one durable proof identity."""
        for label, values in {
            "runtime membership": self.runtime_membership_principals,
            "runtime SET ROLE denial": self.runtime_set_role_denied_principals,
            "declared": self.declared_operations,
            "observed": self.observed_operations,
            "behavioral proof": self.behavioral_proof_ids,
        }.items():
            if len(set(values)) != len(values):
                raise ValueError(f"identity-root {label} entries must be unique")
        if len(self.behavioral_proof_ids) != len(self.observed_operations):
            raise ValueError(
                "identity-root observed operations require one behavioral proof each"
            )
        if any(not proof.strip() for proof in self.behavioral_proof_ids):
            raise ValueError("identity-root behavioral proof IDs cannot be blank")
        return self


__all__ = ["ModelApplicationDatabaseIdentityRootControlState"]
