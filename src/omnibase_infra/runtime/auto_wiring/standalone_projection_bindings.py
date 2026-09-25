# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-binding database identities for a standalone projection writer (OMN-17454).

The in-process dispatch path resolves each projection table to the topology
binding that serves its schema domain -- tenant relations to
``tenant_projection``, ``omninode_internal`` relations to
``omninode_runtime_service`` -- and connects as that binding's principal. A
standalone writer (omnimarket ``BaseProjectionRunner``) never went through that
path: it opened one pool from one legacy DSN, so a writer whose tables span two
domains always had one of them refused, at minimum its
``omninode_internal.projection_watermarks`` write.

``resolve_standalone_projection_bindings`` is the public entry point that gives
a standalone writer the same resolution. It reuses the in-process resolver and
DSN boundary rather than a second copy of either, adds the runner's own
watermark table, and refuses -- naming the binding and its carrier, never a
value -- any binding whose DSN cannot be resolved. It does not open
connections; the caller opens one pool per binding and checks each pool's
``current_user`` against the principal returned here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import SecretStr

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_binding_dsn,
    _resolve_projection_database_target,
)

if TYPE_CHECKING:
    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )
    from omnibase_infra.runtime.secret_resolver import SecretResolver

PROJECTION_WATERMARK_TABLE = ModelDbTableDeclaration(
    name="projection_watermarks",
    database_ref="application",
    schema="omninode_internal",
    migration=(
        "docker/migrations/forward/nodes/node_projection_registration/"
        "0005_create_projection_watermarks.sql"
    ),
    access="read_write",
    role="projection_watermark",
)
"""The watermark every ``BaseProjectionRunner`` upserts after each message.

The runner writes it itself, so no node contract declares it; it is added here
so the watermark resolves to the binding the topology declares for
``omninode_internal`` instead of riding whichever pool the writer opened.
"""


@dataclass(frozen=True, slots=True)
class ResolvedStandaloneBinding:
    """One topology binding a standalone writer connects as, with its DSN."""

    binding_ref: str
    principal: str
    physical_database: str
    carrier_description: str
    dsn: SecretStr = field(repr=False)


@dataclass(frozen=True, slots=True)
class StandaloneProjectionBindings:
    """Every binding a standalone writer needs, and which table uses which."""

    physical_database: str
    bindings: Mapping[str, ResolvedStandaloneBinding]
    tables: tuple[str, ...]
    _write_binding_by_table: Mapping[str, str]
    _read_binding_by_table: Mapping[str, str]

    def write_binding_for(self, table: str) -> str:
        """Return the binding a write to ``table`` must go through."""
        return self._binding_for(table, "write")

    def read_binding_for(self, table: str) -> str:
        """Return the binding a read of ``table`` must go through."""
        return self._binding_for(table, "read")

    @property
    def watermark_binding(self) -> str:
        """Return the binding the runner's watermark upsert goes through."""
        return self.write_binding_for(PROJECTION_WATERMARK_TABLE.name)

    def _binding_for(self, table: str, operation: Literal["read", "write"]) -> str:
        by_table = (
            self._write_binding_by_table
            if operation == "write"
            else self._read_binding_by_table
        )
        binding = by_table.get(table)
        if binding is None:
            raise KeyError(
                f"table {table!r} has no declared {operation} binding; "
                f"declared tables: {sorted(self.tables)!r}"
            )
        return binding


def resolve_standalone_projection_bindings(
    db_tables: Sequence[ModelDbTableDeclaration],
    topology: ModelDeploymentTopology,
    *,
    secret_resolver: SecretResolver | None = None,
    catalog_read_binding: str | None = None,
    catalog_write_binding: str | None = None,
) -> StandaloneProjectionBindings:
    """Resolve a standalone writer's tables, plus its watermark, to bindings.

    Raises ``ValueError`` when a table maps to no granted binding (the
    in-process resolver's own refusal, unchanged) or when a binding's DSN
    carrier resolves to nothing. The refusal names the binding and its carrier
    -- the environment variable or ``secret_ref`` -- and never a DSN value.
    """
    tables = tuple(db_tables)
    if all(table.name != PROJECTION_WATERMARK_TABLE.name for table in tables):
        tables = (*tables, PROJECTION_WATERMARK_TABLE)

    target = _resolve_projection_database_target(
        tables,
        topology,
        catalog_read_binding=catalog_read_binding,
        catalog_write_binding=catalog_write_binding,
    )

    resolved: dict[str, ResolvedStandaloneBinding] = {}
    for binding in target.bindings:
        dsn = _resolve_binding_dsn(binding, secret_resolver)
        if not dsn:
            raise ValueError(
                f"Standalone projection binding {binding.binding_ref!r} "
                f"(principal {binding.principal!r}) has no DSN: its carrier "
                f"{binding.carrier_description} resolved to nothing"
            )
        resolved[binding.binding_ref] = ResolvedStandaloneBinding(
            binding_ref=binding.binding_ref,
            principal=binding.principal,
            physical_database=binding.physical_database,
            carrier_description=binding.carrier_description,
            dsn=SecretStr(dsn),
        )

    write_by_table = {
        table_target.table.name: table_target.write_binding.binding_ref
        for table_target in target.table_targets
        if table_target.write_binding is not None
    }
    read_by_table = {
        table_target.table.name: table_target.read_binding.binding_ref
        for table_target in target.table_targets
        if table_target.read_binding is not None
    }
    return StandaloneProjectionBindings(
        physical_database=target.physical_database,
        bindings=resolved,
        tables=tuple(table.name for table in tables),
        _write_binding_by_table=write_by_table,
        _read_binding_by_table=read_by_table,
    )


__all__ = [
    "PROJECTION_WATERMARK_TABLE",
    "ResolvedStandaloneBinding",
    "StandaloneProjectionBindings",
    "resolve_standalone_projection_bindings",
]
