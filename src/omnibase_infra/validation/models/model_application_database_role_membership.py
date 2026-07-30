# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit permitted role membership in an application database ACL matrix."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelApplicationDatabaseRoleMembership(BaseModel):
    """One exact PostgreSQL role membership and its PG16 option state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    database_ref: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    role: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    member: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    admin_option: bool = False
    inherit_option: bool = False
    set_option: bool = True

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.database_ref, self.role, self.member)


__all__ = ["ModelApplicationDatabaseRoleMembership"]
