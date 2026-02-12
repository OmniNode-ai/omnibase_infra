# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database ownership validation utility.

Validates that the connected database is owned by the expected service by
querying the ``public.db_metadata`` singleton table. Used during kernel
startup to prevent cross-service data corruption after the DB-per-repo split.

Related:
    - OMN-2085: Handshake hardening -- DB ownership marker + startup assertion
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

import asyncpg
import asyncpg.exceptions

from omnibase_infra.errors.error_db_ownership import (
    DbOwnershipMismatchError,
    DbOwnershipMissingError,
)

logger = logging.getLogger(__name__)

_OWNERSHIP_QUERY = "SELECT owner_service FROM public.db_metadata LIMIT 1"


async def validate_db_ownership(
    pool: asyncpg.Pool,
    expected_owner: str,
    correlation_id: UUID | None = None,
) -> None:
    """Assert that the connected database is owned by ``expected_owner``.

    Queries ``public.db_metadata`` for the singleton ownership row and compares
    ``owner_service`` against ``expected_owner``. Raises on mismatch or missing
    data -- the kernel catches these typed errors and hard-fails.

    Args:
        pool: asyncpg connection pool (must already be created).
        expected_owner: Service name that should own this database
            (e.g. ``"omnibase_infra"``). Must be a non-empty, non-whitespace
            string.
        correlation_id: Optional correlation ID for tracing. Auto-generated
            if not provided.

    Raises:
        ValueError: If ``expected_owner`` is empty or whitespace-only.
        DbOwnershipMismatchError: Database is owned by a different service.
        DbOwnershipMissingError: ``db_metadata`` table or ownership row
            does not exist (database not migrated).
    """
    if not expected_owner or not expected_owner.strip():
        raise ValueError("expected_owner must be a non-empty, non-whitespace string")

    if correlation_id is None:
        correlation_id = uuid4()

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(_OWNERSHIP_QUERY)
    except Exception as exc:
        # Only treat "table does not exist" as a missing-metadata error.
        # Transient failures (connection errors, timeouts) must propagate
        # with their original type so callers can distinguish recoverable
        # errors from a genuinely un-migrated database.
        if isinstance(exc, asyncpg.exceptions.UndefinedTableError):
            raise DbOwnershipMissingError(
                "db_metadata table does not exist — run migrations first. "
                f"Expected owner '{expected_owner}'. "
                "Hint: check OMNIBASE_INFRA_DB_URL points "
                "to the correct service database.",
                expected_owner=expected_owner,
                correlation_id=correlation_id,
            ) from exc
        # Connection errors, timeouts, auth failures, etc. -- propagate as-is
        # so the caller can apply appropriate retry/backoff logic.
        raise

    if row is None:
        raise DbOwnershipMissingError(
            f"public.db_metadata table exists but contains no rows. "
            f"Expected owner '{expected_owner}'. "
            "Hint: run migrations to seed the ownership row.",
            expected_owner=expected_owner,
            correlation_id=correlation_id,
        )

    actual_owner = row["owner_service"]
    if actual_owner != expected_owner:
        raise DbOwnershipMismatchError(
            f"Database ownership mismatch: expected '{expected_owner}', "
            f"found '{actual_owner}'. "
            "Hint: check OMNIBASE_INFRA_DB_URL points to the correct "
            "service database, not a database owned by another service.",
            expected_owner=expected_owner,
            actual_owner=actual_owner,
            correlation_id=correlation_id,
        )

    logger.info(
        "DB ownership validated: owner_service='%s' (correlation_id=%s)",
        actual_owner,
        correlation_id,
    )


__all__ = ["validate_db_ownership"]
