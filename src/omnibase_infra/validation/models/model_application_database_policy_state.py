# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Catalog-observed PostgreSQL policy state for domain enforcement."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelApplicationDatabasePolicyState(BaseModel):
    """One policy's exact composition and predicates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    permissive: bool
    command: Literal["ALL", "SELECT", "INSERT", "UPDATE", "DELETE"]
    roles: tuple[str, ...] = Field(..., min_length=1)
    using_expression: str | None = None
    with_check_expression: str | None = None

    @field_validator("roles")
    @classmethod
    def canonicalize_roles(cls, roles: tuple[str, ...]) -> tuple[str, ...]:
        """Keep exact PostgreSQL role names as unique immutable set evidence."""
        if any(not role for role in roles):
            raise ValueError("policy roles cannot contain an empty role name")
        if len(set(roles)) != len(roles):
            raise ValueError("policy roles must be unique")
        return tuple(sorted(roles))


__all__ = ["ModelApplicationDatabasePolicyState"]
