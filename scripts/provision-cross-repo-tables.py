#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Provision cross-repo tables that omnibase_infra owns but other services need.

omnibase_infra owns the ``idempotency_records`` table definition and migrations.
Other services (e.g. omniintelligence) share the same database but cannot run
omnibase_infra migrations directly.  This script provisions those tables into a
target database so that downstream services can start cleanly.

The DDL is copied verbatim from
``src/omnibase_infra/idempotency/store_postgres.py`` (the authoritative source).

Usage:
    uv run python scripts/provision-cross-repo-tables.py \\
        --target-db postgresql://postgres:pw@localhost:5436/omniintelligence

    # Dry run (prints SQL without executing)
    uv run python scripts/provision-cross-repo-tables.py \\
        --target-db postgresql://postgres:pw@localhost:5436/omniintelligence \\
        --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import asyncpg

# ---------------------------------------------------------------------------
# DDL — copied verbatim from
#   src/omnibase_infra/idempotency/store_postgres.py _ensure_table_exists()
#
# IMPORTANT: if the idempotency_records schema ever changes, update BOTH files.
# The table_name literal "idempotency_records" matches the default value of
# ModelPostgresIdempotencyStoreConfig.table_name.
# ---------------------------------------------------------------------------

_TABLE_NAME = "idempotency_records"

CROSS_REPO_TABLES: dict[str, list[str]] = {
    _TABLE_NAME: [
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
            id UUID PRIMARY KEY,
            domain VARCHAR(255),
            message_id UUID NOT NULL,
            correlation_id UUID,
            processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            UNIQUE (domain, message_id)
        )
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_processed_at
        ON {_TABLE_NAME}(processed_at)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_domain
        ON {_TABLE_NAME}(domain)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_correlation_id
        ON {_TABLE_NAME}(correlation_id)
        WHERE correlation_id IS NOT NULL
        """,
    ],
}


async def provision(target_db_url: str, *, dry_run: bool = False) -> None:
    """Provision cross-repo tables into the target database.

    Args:
        target_db_url: asyncpg-compatible DSN for the target database.
        dry_run: If True, print SQL statements without executing them.

    Raises:
        SystemExit: On database connection or execution failure.
    """
    if dry_run:
        print("[DRY-RUN] Would execute the following SQL:")
        for table_name, statements in CROSS_REPO_TABLES.items():
            print(f"\n  -- table: {table_name}")
            for stmt in statements:
                for line in stmt.strip().splitlines():
                    print(f"  {line}")
        return

    try:
        conn = await asyncpg.connect(target_db_url)
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(
            f"ERROR: Could not connect to target database: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        for table_name, statements in CROSS_REPO_TABLES.items():
            async with conn.transaction():
                for stmt in statements:
                    await conn.execute(stmt)
            print(f"  provisioned: {table_name}")
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(
            f"ERROR: Failed to provision cross-repo tables: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        await conn.close()


def main() -> None:
    """Parse arguments and run provisioning."""
    parser = argparse.ArgumentParser(
        description="Provision cross-repo tables owned by omnibase_infra into a target DB.",
    )
    parser.add_argument(
        "--target-db",
        required=True,
        help="Target database DSN (postgresql://user:pass@host:port/dbname)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print SQL without executing",
    )
    args = parser.parse_args()

    asyncio.run(provision(args.target_db, dry_run=args.dry_run))
    if not args.dry_run:
        print("Cross-repo provisioning complete.")


if __name__ == "__main__":
    main()
