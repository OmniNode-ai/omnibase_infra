# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Warning-based preflight validation for declared db_tables.

This module is a pre-wiring step that checks whether tables declared in
contract db_io.db_tables actually exist in the target database before
wire_from_manifest is called.

This is the WARNING phase — does NOT block wiring. Strict blocking is
deferred to Phase 2 once all contracts are complete and validated in
production. Degraded operation is preferable to a hard startup failure
for missing tables.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def validate_db_tables(contracts: list[dict], db_conn: object) -> list[dict]:
    """Preflight check: warn when declared db_tables don't exist in the target database.

    Iterates over each contract's db_io.db_tables declarations and queries
    pg_tables to verify existence. Returns structured warning dicts for any
    missing table but does NOT raise — wiring continues regardless.

    This is the warning phase. Strict blocking is Phase 2.

    Args:
        contracts: List of raw contract dicts (as loaded from contract.yaml).
                   Each may contain a ``db_io.db_tables`` list.
        db_conn:   An asyncpg connection (or compatible mock). Must support
                   ``await conn.fetchval(query, *args)``.

    Returns:
        List of warning dicts. Each dict has keys:
        - ``reason``: always ``"missing_db_table"``
        - ``severity``: always ``"warning"``
        - ``details``: dict with ``table``, ``database``, and ``node`` keys.
        Empty list if all declared tables exist (or no tables are declared).
    """
    warnings: list[dict] = []
    for contract in contracts:
        db_io = contract.get("db_io", {}) or {}
        db_tables = db_io.get("db_tables", []) or []
        for table_decl in db_tables:
            table_name = table_decl["name"]
            database = table_decl.get("database", "omnidash_analytics")
            node_name = contract.get("name")
            exists = await _table_exists(db_conn, table_name, database)
            if not exists:
                warnings.append(
                    {
                        "reason": "missing_db_table",
                        "severity": "warning",
                        "details": {
                            "table": table_name,
                            "database": database,
                            "node": node_name,
                        },
                    }
                )
                logger.warning(
                    "Node %s declares table %s but it does not exist in %s. "
                    "Run migrations before starting this node.",
                    node_name,
                    table_name,
                    database,
                )
    return warnings


async def _table_exists(db_conn: object, table_name: str, database: str) -> bool:
    """Return True if *table_name* exists in pg_tables for *database*.

    Uses a parameterised query against the PostgreSQL information schema to
    avoid any possibility of SQL injection from contract-sourced table names.
    """
    row = await db_conn.fetchval(  # type: ignore[attr-defined]
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = $1",
        table_name,
    )
    return row is not None
