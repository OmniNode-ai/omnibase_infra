# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Privileges for a pruned relation."""

from pydantic import BaseModel, ConfigDict

from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege


class ModelTableGrantPrunePrivileges(BaseModel):
    """Privileges for a pruned relation; empty when the relation is not pruned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    privileges: tuple[EnumDatabasePrivilege, ...]
