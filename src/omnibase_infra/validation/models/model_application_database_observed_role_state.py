# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Catalog-observed attributes for an application-database principal role."""

from pydantic import BaseModel, ConfigDict, Field


class ModelApplicationDatabaseObservedRoleState(BaseModel):
    """Catalog-observed attributes for a present principal role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    login: bool
    superuser: bool
    bypass_rls: bool
    create_database: bool
    create_role: bool
    replication: bool
    inherit: bool


__all__ = ["ModelApplicationDatabaseObservedRoleState"]
