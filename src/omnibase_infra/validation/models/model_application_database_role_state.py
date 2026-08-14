# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit desired state for an ACL-governed PostgreSQL role."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelApplicationDatabaseRoleState(BaseModel):
    """Role attributes that an ACL policy explicitly authorizes changing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    role_kind: Literal["owner", "workload", "migration", "external_connect"]
    manage_attributes: bool = True
    manage_memberships: bool = True
    login: bool
    superuser: Literal[False] = False
    bypass_rls: Literal[False] = False
    create_database: Literal[False] = False
    create_role: Literal[False] = False
    replication: Literal[False] = False
    inherit: Literal[False] = False


__all__ = ["ModelApplicationDatabaseRoleState"]
