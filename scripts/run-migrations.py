#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Migration runner for omnibase_infra.

Applies pending SQL migrations from both migration sets to a PostgreSQL database
and updates schema_migrations tracking table. Safe to run on fresh and existing DBs.

Usage:
    uv run python scripts/run-migrations.py --dry-run
    uv run python scripts/run-migrations.py
    uv run python scripts/run-migrations.py --target 030
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg

REPO_ROOT = Path(__file__).parent.parent

MIGRATION_SETS = [
    ("docker", REPO_ROOT / "docker" / "migrations" / "forward"),
    ("src", REPO_ROOT / "src" / "omnibase_infra" / "migrations" / "forward"),
]
CREATE_DATABASE_DIRECTIVE = re.compile(
    r"^--\s*onex-create-database:\s*([A-Za-z_][A-Za-z0-9_-]*)\s*$",
    re.IGNORECASE,
)
CREATE_DATABASE_DIRECTIVE_PREFIX = re.compile(
    r"^--\s*onex-create-database\s*:",
    re.IGNORECASE,
)


def extract_sequence_number(filename: str) -> int:
    """Extract leading integer sequence number from migration filename."""
    stem = Path(filename).stem
    prefix = ""
    for ch in stem:
        if ch.isdigit():
            prefix += ch
        else:
            break
    if not prefix:
        raise ValueError(
            f"no leading sequence number in migration filename: {filename!r}"
        )
    return int(prefix)


def validate_no_duplicates(files: list[Path]) -> None:
    """Raise if any two files share a sequence number."""
    seen: dict[int, Path] = {}
    for f in files:
        n = extract_sequence_number(f.name)
        if n in seen:
            raise ValueError(
                f"duplicate sequence number {n}: {seen[n].name!r} and {f.name!r}"
            )
        seen[n] = f


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_connect_directive(sql: str) -> tuple[str | None, str]:
    """Return an optional target database from a leading psql ``\\connect`` line.

    The migration runner executes SQL through asyncpg, not psql, so psql
    meta-commands are not valid SQL. A leading ``\\connect <database>`` is
    treated as migration metadata and removed from the SQL sent to Postgres.
    """

    remaining: list[str] = []
    target_database: str | None = None

    for line in sql.splitlines(keepends=True):
        stripped = line.strip()
        if target_database is None and not stripped:
            remaining.append(line)
            continue
        if target_database is None and stripped.startswith("--"):
            remaining.append(line)
            continue
        if target_database is None and stripped.startswith("\\connect"):
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(f"unsupported psql connect directive: {stripped!r}")
            target_database = parts[1]
            validate_database_identifier(target_database)
            continue
        if stripped.startswith("\\"):
            raise ValueError(
                f"unsupported psql meta-command in migration: {stripped!r}"
            )
        remaining.append(line)

    return target_database, "".join(remaining)


def parse_create_database_directive(sql: str) -> tuple[str | None, str]:
    """Return optional database creation directive and SQL without the directive.

    ``-- onex-create-database: <database>`` is migration metadata consumed by
    both the Python runner and the shell forward-migration runner. The directive
    keeps database creation idempotent without embedding psql-only meta commands
    such as ``\\gexec`` in migrations that CI applies through asyncpg.
    """

    remaining: list[str] = []
    database: str | None = None

    for line in sql.splitlines(keepends=True):
        stripped = line.strip()
        if database is None:
            match = CREATE_DATABASE_DIRECTIVE.match(stripped)
            if match:
                database = match.group(1)
                validate_database_identifier(database)
                continue
            if CREATE_DATABASE_DIRECTIVE_PREFIX.match(stripped):
                raise ValueError(f"invalid database name in directive: {stripped!r}")
        remaining.append(line)

    return database, "".join(remaining)


def validate_database_identifier(database: str) -> None:
    if len(database) > 63 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", database):
        raise ValueError(f"invalid database name in connect directive: {database!r}")


def database_url_for_database(db_url: str, database: str) -> str:
    validate_database_identifier(database)
    parsed = urlsplit(db_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("database URL must include scheme and network location")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{database}", parsed.query, parsed.fragment)
    )


async def ensure_database_exists(conn: asyncpg.Connection, database: str) -> None:
    validate_database_identifier(database)
    if await database_exists(conn, database):
        return
    await conn.execute(f'CREATE DATABASE "{database}"')


async def database_exists(conn: asyncpg.Connection, database: str) -> bool:
    validate_database_identifier(database)
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            database,
        )
    )


def split_sql_statements(sql: str) -> list[str]:
    """Split SQL text into individual statements, respecting dollar-quoting.

    Needed for migrations that contain CREATE INDEX CONCURRENTLY, which cannot
    run inside a multi-statement simple-query block (Postgres wraps those in an
    implicit transaction).

    Uses a token-scan approach:
    - SQL single-line comments (--) are consumed without splitting.
    - Dollar-quoted strings ($$ or $tag$) are consumed without splitting.
    - Single-quoted strings are consumed without splitting.
    - Semicolons outside of the above contexts terminate a statement.
    """

    statements: list[str] = []
    buf: list[str] = []
    pos = 0
    n = len(sql)

    while pos < n:
        ch = sql[pos]

        # Single-line comment: consume to end of line
        if ch == "-" and pos + 1 < n and sql[pos + 1] == "-":
            end = sql.find("\n", pos)
            if end == -1:
                end = n
            else:
                end += 1
            buf.append(sql[pos:end])
            pos = end
            continue

        # Dollar-quoted string: $tag$...$tag$
        if ch == "$":
            m = re.match(r"\$([^$]*)\$", sql[pos:])
            if m:
                tag = m.group(0)  # e.g. "$$" or "$BODY$"
                close_pos = sql.find(tag, pos + len(tag))
                if close_pos == -1:
                    # Unclosed dollar-quote: consume rest of input
                    buf.append(sql[pos:])
                    pos = n
                else:
                    end = close_pos + len(tag)
                    buf.append(sql[pos:end])
                    pos = end
                continue

        # Single-quoted string: consume respecting '' escapes
        if ch == "'":
            end = pos + 1
            while end < n:
                if sql[end] == "'" and end + 1 < n and sql[end + 1] == "'":
                    end += 2  # escaped quote
                elif sql[end] == "'":
                    end += 1
                    break
                else:
                    end += 1
            buf.append(sql[pos:end])
            pos = end
            continue

        # Statement terminator
        if ch == ";":
            buf.append(";")
            stmt = "".join(buf).strip()
            if stmt and stmt != ";":
                statements.append(stmt)
            buf = []
            pos += 1
            continue

        buf.append(ch)
        pos += 1

    # Trailing statement without semicolon
    remainder = "".join(buf).strip()
    if remainder:
        statements.append(remainder)

    # Filter out fragments that contain only comments (no SQL keywords or DDL).
    def _has_sql_content(stmt: str) -> bool:
        for line in stmt.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("--"):
                return True
        return False

    return [s for s in statements if _has_sql_content(s)]


async def ensure_schema_migrations_table(conn: asyncpg.Connection) -> None:
    """Create schema_migrations if it does not exist (idempotent)."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS public.schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            checksum     TEXT NOT NULL,
            source_set   TEXT NOT NULL
        )
    """)


async def get_applied_ids(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch("SELECT migration_id FROM public.schema_migrations")
    return {r["migration_id"] for r in rows}


def migration_id(source_set: str, filename: str) -> str:
    return f"{source_set}/{filename}"


async def apply_migration(
    conn: asyncpg.Connection,
    source_set: str,
    path: Path,
    dry_run: bool,
    *,
    base_conn: asyncpg.Connection | None = None,
) -> None:
    mid = migration_id(source_set, path.name)
    _, sql = parse_connect_directive(path.read_text())
    create_database, sql = parse_create_database_directive(sql)
    checksum = file_checksum(path)

    if dry_run:
        print(f"  [dry-run] would apply: {mid}")
        return

    if create_database is not None:
        await ensure_database_exists(base_conn or conn, create_database)

    # Migrations that use CREATE INDEX CONCURRENTLY cannot run inside an implicit
    # transaction block (which Postgres creates when you send multiple statements
    # via the simple-query protocol). Execute each statement individually so that
    # every statement runs in its own autocommit context.
    #
    # Check for CONCURRENTLY only in non-comment lines to avoid false positives
    # on migration files that mention CONCURRENTLY only in comments.
    non_comment_lines = [
        line for line in sql.splitlines() if not line.strip().startswith("--")
    ]
    has_concurrent_index = "CONCURRENTLY" in "\n".join(non_comment_lines).upper()
    if has_concurrent_index:
        for stmt in split_sql_statements(sql):
            await conn.execute(stmt)
    else:
        await conn.execute(sql)
    await conn.execute(
        """
        INSERT INTO public.schema_migrations (migration_id, checksum, source_set)
        VALUES ($1, $2, $3)
        ON CONFLICT (migration_id) DO NOTHING
        """,
        mid,
        checksum,
        source_set,
    )
    print(f"  applied: {mid}")


async def run(db_url: str, dry_run: bool, target: int | None) -> int:
    """Apply all pending migrations. Returns count of migrations applied."""
    base_conn = await asyncpg.connect(db_url)
    conns: dict[str, asyncpg.Connection] = {"": base_conn}
    applied_by_database: dict[str, set[str]] = {}

    async def _conn_for_database(database: str | None) -> asyncpg.Connection:
        key = database or ""
        if key in conns:
            return conns[key]
        await ensure_database_exists(base_conn, key)
        conns[key] = await asyncpg.connect(database_url_for_database(db_url, key))
        return conns[key]

    async def _applied_ids_for_database(database: str | None) -> set[str]:
        key = database or ""
        if key not in applied_by_database:
            if dry_run and database and not await database_exists(base_conn, database):
                applied_by_database[key] = set()
                return applied_by_database[key]
            conn = await _conn_for_database(database)
            await ensure_schema_migrations_table(conn)
            applied_by_database[key] = await get_applied_ids(conn)
        return applied_by_database[key]

    try:
        pending: list[tuple[int, str, Path, str | None]] = []
        for source_set, migration_dir in MIGRATION_SETS:
            if not migration_dir.exists():
                print(f"  warning: migration directory not found: {migration_dir}")
                continue
            files = sorted(
                migration_dir.glob("*.sql"),
                key=lambda f: f.name,
            )
            validate_no_duplicates(files)
            for f in files:
                mid = migration_id(source_set, f.name)
                target_database, _ = parse_connect_directive(f.read_text())
                applied = await _applied_ids_for_database(target_database)
                if mid in applied:
                    continue
                seq = extract_sequence_number(f.name)
                if target is not None and seq > target:
                    continue
                pending.append((seq, source_set, f, target_database))

        pending.sort(key=lambda t: (t[0], t[1]))

        if not pending:
            print("No pending migrations.")
            return 0

        print(f"Applying {len(pending)} pending migration(s)...")
        for _, source_set, path, target_database in pending:
            conn = base_conn if dry_run else await _conn_for_database(target_database)
            await apply_migration(
                conn,
                source_set,
                path,
                dry_run,
                base_conn=base_conn,
            )

        return len(pending)
    finally:
        for conn in reversed(list(conns.values())):
            await conn.close()


def restamp_fingerprint(db_url: str) -> None:
    """Call util_schema_fingerprint stamp after successful apply."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omnibase_infra.runtime.util_schema_fingerprint",
            "stamp",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMNIBASE_INFRA_DB_URL": db_url},
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        print(
            f"  warning: fingerprint restamp failed (exit {result.returncode}): "
            f"{detail}"
        )
    else:
        print("  fingerprint restamped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply pending omnibase_infra migrations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pending migrations without applying",
    )
    parser.add_argument(
        "--target", type=int, help="Only apply up to this sequence number"
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("OMNIBASE_INFRA_DB_URL"),
        help="PostgreSQL connection URL (default: OMNIBASE_INFRA_DB_URL env var)",
    )
    args = parser.parse_args()

    if not args.db_url:
        print("ERROR: --db-url or OMNIBASE_INFRA_DB_URL required", file=sys.stderr)
        sys.exit(1)

    count = asyncio.run(run(args.db_url, args.dry_run, args.target))

    if count > 0 and not args.dry_run:
        restamp_fingerprint(args.db_url)


if __name__ == "__main__":
    main()
