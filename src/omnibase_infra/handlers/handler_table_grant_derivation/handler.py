# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler for pruned-relation table grant derivation."""

from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_table_grant_prune_privileges import (
    ModelTableGrantPrunePrivileges,
)
from omnibase_infra.models.model_table_grant_prune_query import (
    ModelTableGrantPruneQuery,
)

# A retention prune's DELETE reads the rows it removes, so DELETE rides with SELECT and nothing else, never TRUNCATE; a principal that prunes declares it on the one relation it prunes.
PRUNE_PRIVILEGES: frozenset[EnumDatabasePrivilege] = frozenset(
    {EnumDatabasePrivilege.SELECT, EnumDatabasePrivilege.DELETE}
)


# The relations whose writer also deletes rows. The node contract's
# config.pruned_relations declares the same list with the reason for each; a test
# asserts the two agree, because a compute handler reads no file.
PRUNED_RELATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("omninode_internal", "gate_store_reservations"),
        ("omninode_internal", "floor_integration_rows"),
        ("public", "savings_priced_models"),
    }
)


class HandlerTableGrantDerivation:
    """Handler naming the extra privileges a pruned relation needs."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the handler type."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the handler category."""
        return EnumHandlerTypeCategory.COMPUTE

    def needs_delete(self, schema: str, table: str) -> bool:
        """Return whether the relation is pruned."""
        return (schema, table) in PRUNED_RELATIONS

    def handle(
        self, request: ModelTableGrantPruneQuery
    ) -> ModelTableGrantPrunePrivileges:
        """Return prune privileges for the requested relation."""
        if self.needs_delete(request.schema_name, request.table_name):
            return ModelTableGrantPrunePrivileges(
                privileges=tuple(
                    sorted(PRUNE_PRIVILEGES, key=lambda privilege: privilege.value)
                )
            )
        return ModelTableGrantPrunePrivileges(privileges=())


__all__ = ["PRUNE_PRIVILEGES", "PRUNED_RELATIONS", "HandlerTableGrantDerivation"]
