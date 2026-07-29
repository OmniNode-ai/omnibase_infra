# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed physical-table preflight for discovered ``db_io`` declarations.

Global exactly-one ownership is enforced by
``validation.application_relation_ownership``. This optional runtime preflight
has the narrower job of warning when an already-typed table is absent from the
current PostgreSQL connection. It never parses raw contract dictionaries and it
never invents a default database or schema.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract
from omnibase_infra.runtime.auto_wiring.models.model_db_table_validation_warning import (
    ModelDbTableValidationWarning,
)
from omnibase_infra.runtime.auto_wiring.protocol_db_table_catalog_connection import (
    ProtocolDbTableCatalogConnection,
)

logger = logging.getLogger(__name__)


async def validate_db_tables(
    contracts: Sequence[ModelDiscoveredContract],
    db_conn: ProtocolDbTableCatalogConnection,
) -> tuple[ModelDbTableValidationWarning, ...]:
    """Warn for typed declared tables absent from the current connection."""
    warnings: list[ModelDbTableValidationWarning] = []
    for contract in contracts:
        if contract.db_io is None:
            continue
        for table in contract.db_io.db_tables:
            exists = await _table_exists(db_conn, table.schema, table.name)
            if exists:
                continue
            warning = ModelDbTableValidationWarning(
                table=table.name,
                database_ref=table.database_ref,
                schema=table.schema,
                node=contract.name,
            )
            warnings.append(warning)
            logger.warning(
                "Node %s declares missing table %s.%s in database_ref=%s. "
                "Run the declared migration before starting this node.",
                contract.name,
                table.schema,
                table.name,
                table.database_ref,
            )
    return tuple(warnings)


async def _table_exists(
    db_conn: ProtocolDbTableCatalogConnection,
    schema: str,
    table_name: str,
) -> bool:
    """Check the exact typed schema/name pair on the current connection."""
    row = await db_conn.fetchval(
        "SELECT tablename FROM pg_tables WHERE schemaname = $1 AND tablename = $2",
        schema,
        table_name,
    )
    return row is not None


__all__ = [
    "ModelDbTableValidationWarning",
    "ProtocolDbTableCatalogConnection",
    "validate_db_tables",
]
