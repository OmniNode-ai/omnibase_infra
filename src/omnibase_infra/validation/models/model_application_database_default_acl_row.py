# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Future-object privilege cell in an application-database ACL matrix."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege


class ModelApplicationDatabaseDefaultAclRow(BaseModel):
    """One future-object privilege cell for an actual schema owner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner: str = Field(..., min_length=1)
    database_ref: str = Field(..., min_length=1)
    physical_database: str = Field(..., min_length=1)
    schema_ref: str = Field(..., min_length=1)
    object_type: Literal[
        EnumDatabaseGrantObjectType.TABLE,
        EnumDatabaseGrantObjectType.SEQUENCE,
        EnumDatabaseGrantObjectType.FUNCTION,
        EnumDatabaseGrantObjectType.TYPE,
    ]
    grantee: str = Field(..., min_length=1)
    privileges: tuple[EnumDatabasePrivilege, ...] = ()


__all__ = ["ModelApplicationDatabaseDefaultAclRow"]
