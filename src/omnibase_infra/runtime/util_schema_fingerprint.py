# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Schema fingerprint computation and validation utility.

Computes a deterministic SHA-256 fingerprint of the live database schema
(columns + constraints) for tables declared in a ``ModelSchemaManifest``.
At startup, the kernel compares the live fingerprint against the expected
value stored in ``public.db_metadata`` to detect schema drift.

Algorithm:
    1. Query ``pg_catalog`` for columns and constraints of manifest tables.
    2. Build a canonical JSON record per table (columns + constraints).
    3. SHA-256 each per-table record.
    4. SHA-256 the sorted list of per-table hashes to produce the overall
       fingerprint.

Related:
    - OMN-2087: Handshake hardening -- Schema fingerprint manifest + startup assertion
    - OMN-2085: Handshake hardening -- DB ownership marker + startup assertion
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from uuid import UUID, uuid4

import asyncpg
import asyncpg.exceptions

from omnibase_infra.errors.error_schema_fingerprint import (
    SchemaFingerprintMismatchError,
    SchemaFingerprintMissingError,
)
from omnibase_infra.runtime.model_schema_fingerprint_result import (
    ModelSchemaFingerprintResult,
)
from omnibase_infra.runtime.model_schema_manifest import ModelSchemaManifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL queries -- use pg_catalog for stable, format-aware column metadata.
# ---------------------------------------------------------------------------

_COLUMNS_QUERY = """\
SELECT
    c.relname AS table_name,
    a.attname AS column_name,
    format_type(a.atttypid, a.atttypmod) AS canonical_type,
    a.attnotnull AS not_null,
    pg_get_expr(d.adbin, d.adrelid) AS column_default,
    a.attnum AS ordinal_position
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
WHERE n.nspname = $1
  AND c.relname = ANY($2)
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY c.relname, a.attnum
"""

_CONSTRAINTS_QUERY = """\
SELECT
    c.relname AS table_name,
    con.contype AS constraint_type,
    (
        SELECT array_agg(att.attname ORDER BY u.ord)
        FROM unnest(con.conkey) WITH ORDINALITY AS u(attnum, ord)
        JOIN pg_catalog.pg_attribute att
            ON att.attrelid = con.conrelid AND att.attnum = u.attnum
    ) AS local_columns,
    confrel.relname AS ref_table,
    (
        SELECT array_agg(att.attname ORDER BY u.ord)
        FROM unnest(con.confkey) WITH ORDINALITY AS u(attnum, ord)
        JOIN pg_catalog.pg_attribute att
            ON att.attrelid = con.confrelid AND att.attnum = u.attnum
    ) AS ref_columns,
    con.confupdtype AS on_update,
    con.confdeltype AS on_delete,
    pg_get_constraintdef(con.oid, true) AS constraint_def
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_class confrel ON confrel.oid = con.confrelid
WHERE n.nspname = $1
  AND c.relname = ANY($2)
  AND con.contype IN ('p', 'u', 'f', 'c')
ORDER BY c.relname, con.contype, con.conkey
"""

_EXPECTED_FINGERPRINT_QUERY = (
    "SELECT expected_schema_fingerprint FROM public.db_metadata WHERE id = TRUE LIMIT 1"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Canonical record type for constraint and column dicts that get JSON-serialized
# for hashing. Values are heterogeneous (str, int, bool, list[str], None)
# depending on the column/constraint type, so we use ``object`` as the value type.
_CanonicalRecord = dict[str, object]


def _build_constraint_record(row: asyncpg.Record) -> _CanonicalRecord:
    """Build a canonical constraint dict from a pg_catalog row."""
    con_type: str = row["constraint_type"]
    local_cols: list[str] | None = row["local_columns"]

    if con_type in ("p", "u"):
        # PRIMARY KEY or UNIQUE -- only columns matter
        return {
            "type": con_type,
            "columns": list(local_cols) if local_cols else [],
        }

    if con_type == "f":
        # FOREIGN KEY -- include referencing info and actions
        ref_cols: list[str] | None = row["ref_columns"]
        return {
            "type": con_type,
            "columns": list(local_cols) if local_cols else [],
            "ref_table": row["ref_table"],
            "ref_columns": list(ref_cols) if ref_cols else [],
            "on_update": row["on_update"],
            "on_delete": row["on_delete"],
        }

    # CHECK -- whitespace-normalize the expression
    constraint_def: str = row["constraint_def"] or ""
    normalized_expr = " ".join(constraint_def.split())
    return {
        "type": con_type,
        "expression": normalized_expr,
    }


def _sort_key_for_constraint(c: _CanonicalRecord) -> tuple[str, ...]:
    """Produce a deterministic sort key for a constraint record."""
    con_type = str(c.get("type", ""))
    if con_type in ("p", "u", "f"):
        cols = c.get("columns", [])
        col_list: list[str] = list(cols) if isinstance(cols, list) else []
        return (con_type, ",".join(col_list))
    return (con_type, str(c.get("expression", "")))


def _sha256_json(obj: object) -> str:
    """SHA-256 hex digest of a JSON-serialised object."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _compute_schema_diff(
    expected_per_table: dict[str, str],
    actual_per_table: dict[str, str],
) -> str:
    """Bounded diff summary (max 10 lines).

    Reports tables that were added, removed, or changed between expected
    and actual per-table hashes.

    Args:
        expected_per_table: Mapping of table_name -> hash from expected.
        actual_per_table: Mapping of table_name -> hash from live schema.

    Returns:
        Human-readable diff summary, truncated at 10 lines.
    """
    lines: list[str] = []
    expected_tables = set(expected_per_table)
    actual_tables = set(actual_per_table)

    for table in sorted(actual_tables - expected_tables):
        lines.append(f"  + added: {table}")

    for table in sorted(expected_tables - actual_tables):
        lines.append(f"  - removed: {table}")

    for table in sorted(expected_tables & actual_tables):
        if expected_per_table[table] != actual_per_table[table]:
            lines.append(f"  ~ changed: {table}")

    max_lines = 10
    if len(lines) > max_lines:
        overflow = len(lines) - max_lines
        lines = lines[:max_lines]
        lines.append(f"  ... and {overflow} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def compute_schema_fingerprint(
    pool: asyncpg.Pool,
    manifest: ModelSchemaManifest,
    correlation_id: UUID | None = None,
) -> ModelSchemaFingerprintResult:
    """Compute live schema fingerprint from database.

    Queries ``pg_catalog`` for column and constraint metadata for each table
    in the manifest, then produces a deterministic SHA-256 fingerprint.

    Args:
        pool: asyncpg connection pool (must already be created).
        manifest: Schema manifest declaring tables to fingerprint.
        correlation_id: Optional correlation ID for tracing.

    Returns:
        ``ModelSchemaFingerprintResult`` with fingerprint, counts, and
        per-table hashes.
    """
    if correlation_id is None:
        correlation_id = uuid4()

    table_list = list(manifest.tables)

    if not table_list:
        empty_fingerprint = _sha256_json([])
        return ModelSchemaFingerprintResult(
            fingerprint=empty_fingerprint,
            table_count=0,
            column_count=0,
            constraint_count=0,
            per_table_hashes=(),
        )

    async with pool.acquire() as conn:
        column_rows = await conn.fetch(_COLUMNS_QUERY, manifest.schema_name, table_list)
        constraint_rows = await conn.fetch(
            _CONSTRAINTS_QUERY, manifest.schema_name, table_list
        )

    # Group columns by table
    columns_by_table: dict[str, list[_CanonicalRecord]] = defaultdict(list)
    for row in column_rows:
        col_record = {
            "name": row["column_name"],
            "canonical_type": row["canonical_type"],
            "not_null": row["not_null"],
            "column_default": row["column_default"],
            "ordinal_position": row["ordinal_position"],
        }
        columns_by_table[row["table_name"]].append(col_record)

    # Group constraints by table
    constraints_by_table: dict[str, list[_CanonicalRecord]] = defaultdict(list)
    for row in constraint_rows:
        constraint_record = _build_constraint_record(row)
        constraints_by_table[row["table_name"]].append(constraint_record)

    # Build per-table hashes
    total_columns = 0
    total_constraints = 0
    per_table_hashes: list[tuple[str, str]] = []

    for table_name in sorted(table_list):
        cols = columns_by_table.get(table_name, [])
        cons = constraints_by_table.get(table_name, [])

        # Sort constraints deterministically
        sorted_cons = sorted(cons, key=_sort_key_for_constraint)

        total_columns += len(cols)
        total_constraints += len(sorted_cons)

        table_record = {
            "columns": cols,
            "constraints": sorted_cons,
        }
        table_hash = _sha256_json(table_record)
        per_table_hashes.append((table_name, table_hash))

    # Overall fingerprint
    overall_fingerprint = _sha256_json(per_table_hashes)

    logger.debug(
        "Schema fingerprint computed: %s (tables=%d, columns=%d, constraints=%d, "
        "correlation_id=%s)",
        overall_fingerprint[:16],
        len(per_table_hashes),
        total_columns,
        total_constraints,
        correlation_id,
    )

    return ModelSchemaFingerprintResult(
        fingerprint=overall_fingerprint,
        table_count=len(per_table_hashes),
        column_count=total_columns,
        constraint_count=total_constraints,
        per_table_hashes=tuple(per_table_hashes),
    )


async def validate_schema_fingerprint(
    pool: asyncpg.Pool,
    manifest: ModelSchemaManifest,
    correlation_id: UUID | None = None,
) -> None:
    """Hard-gate: compare live schema fingerprint to expected in db_metadata.

    Reads ``expected_schema_fingerprint`` from ``public.db_metadata``, computes
    the live fingerprint, and compares. On mismatch the kernel must terminate.

    Steps:
        1. Read ``expected_schema_fingerprint`` from db_metadata.
        2. If NULL or missing -> ``SchemaFingerprintMissingError``.
        3. Compute live fingerprint.
        4. Compare.
        5. Mismatch -> ``SchemaFingerprintMismatchError`` with diff.

    Args:
        pool: asyncpg connection pool (must already be created).
        manifest: Schema manifest declaring tables to fingerprint.
        correlation_id: Optional correlation ID for tracing. Auto-generated
            if not provided.

    Raises:
        SchemaFingerprintMissingError: Expected fingerprint not in db_metadata
            or db_metadata table does not exist.
        SchemaFingerprintMismatchError: Live fingerprint differs from expected.
    """
    if correlation_id is None:
        correlation_id = uuid4()

    # 1. Read expected fingerprint from db_metadata
    try:
        async with pool.acquire() as conn:
            expected_fp: str | None = await conn.fetchval(  # type: ignore[assignment]  # asyncpg.fetchval returns Any
                _EXPECTED_FINGERPRINT_QUERY
            )
    except Exception as exc:
        if isinstance(
            exc,
            (
                asyncpg.exceptions.UndefinedTableError,
                asyncpg.exceptions.UndefinedColumnError,
            ),
        ):
            raise SchemaFingerprintMissingError(
                "db_metadata table or expected_schema_fingerprint column does "
                "not exist -- run migrations first. "
                f"Expected owner '{manifest.owner_service}'. "
                "Hint: check OMNIBASE_INFRA_DB_URL points to the correct "
                "service database.",
                expected_owner=manifest.owner_service,
                correlation_id=correlation_id,
            ) from exc
        # Transient errors (connection, timeout, auth) propagate as-is
        raise

    # 2. Check for NULL/missing
    if expected_fp is None:
        raise SchemaFingerprintMissingError(
            "expected_schema_fingerprint is NULL in db_metadata. "
            f"Expected owner '{manifest.owner_service}'. "
            "Hint: run the fingerprint update command to populate: "
            "poetry run python -m omnibase_infra.runtime.util_schema_fingerprint update",
            expected_owner=manifest.owner_service,
            correlation_id=correlation_id,
        )

    expected_fingerprint: str = str(expected_fp)

    # 3. Compute live fingerprint
    result = await compute_schema_fingerprint(
        pool=pool,
        manifest=manifest,
        correlation_id=correlation_id,
    )

    # 4. Compare
    if result.fingerprint == expected_fingerprint:
        logger.info(
            "Schema fingerprint validated: %s (tables=%d, correlation_id=%s)",
            result.fingerprint[:16],
            result.table_count,
            correlation_id,
        )
        return

    # 5. Mismatch -- compute diff and raise
    # Build per-table hash dicts for diff computation.
    # We don't have per-table expected hashes stored in db_metadata, so we
    # construct expected_per_table from the manifest table list using the
    # actual hashes for tables that exist (so matching tables don't appear as
    # "changed"). Missing tables keep an empty sentinel so they surface as
    # removed in the diff.
    actual_per_table = dict(result.per_table_hashes)
    expected_per_table = {
        table: actual_per_table.get(table, "") for table in manifest.tables
    }

    table_diff = _compute_schema_diff(expected_per_table, actual_per_table)
    diff_header = (
        f"Expected fingerprint: {expected_fingerprint}\n"
        f"Actual fingerprint:   {result.fingerprint}\n"
        f"Tables fingerprinted: {result.table_count}\n"
        f"Columns: {result.column_count}, Constraints: {result.constraint_count}"
    )
    diff_summary = f"{diff_header}\n{table_diff}" if table_diff else diff_header

    raise SchemaFingerprintMismatchError(
        f"Schema fingerprint mismatch: expected '{expected_fingerprint[:16]}...', "
        f"computed '{result.fingerprint[:16]}...'. "
        "The live database schema does not match what this version of the code "
        "expects. Run migrations or update the expected fingerprint. "
        "Hint: check OMNIBASE_INFRA_DB_URL points to the correct service database.",
        expected_fingerprint=expected_fingerprint,
        actual_fingerprint=result.fingerprint,
        diff_summary=diff_summary,
        correlation_id=correlation_id,
    )


__all__ = [
    "compute_schema_fingerprint",
    "validate_schema_fingerprint",
]
