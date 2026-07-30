# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit CONNECT policy for one PostgreSQL database."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class ModelApplicationDatabaseConnectionPolicy(BaseModel):
    """Principals allowed to connect to one protected physical database."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    database_ref: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    physical_database: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    observed_database_owner_role: str = Field(
        ...,
        pattern=r"^[a-z_][a-z0-9_]*$",
    )
    allowed_principals: tuple[str, ...] = Field(..., min_length=1)

    @field_validator("allowed_principals")
    @classmethod
    def validate_allowed_principals(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require a unique, safely quotable allowlist."""
        if len(set(values)) != len(values):
            raise ValueError("allowed_principals must be unique")
        invalid = sorted(
            value for value in values if _SQL_IDENTIFIER.fullmatch(value) is None
        )
        if invalid:
            raise ValueError(
                f"allowed_principals contain unsafe identifiers: {invalid!r}"
            )
        return values


__all__ = ["ModelApplicationDatabaseConnectionPolicy"]
