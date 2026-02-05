# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Ledger event models for runtime traceability.

This module provides Pydantic models for ledger events emitted by runtime
components (PostgresRepositoryRuntime, etc.) for observability and replay.

Models:
    - ModelLedgerEventBase: Base class with common envelope fields
    - ModelDbQueryRequested: Entry event for database operations
    - ModelDbQuerySucceeded: Success event with duration and row count
    - ModelDbQueryFailed: Failure event with error details

Security:
    All models explicitly exclude raw SQL and unredacted parameters.
    Only operation names, fingerprints, and sanitized errors are logged.

Example:
    >>> from omnibase_infra.models.ledger import (
    ...     ModelDbQueryRequested,
    ...     ModelDbQuerySucceeded,
    ...     ModelDbQueryFailed,
    ... )
    >>>
    >>> event = ModelDbQueryRequested(
    ...     event_id=uuid4(),
    ...     correlation_id=correlation_id,
    ...     idempotency_key=f"{correlation_id}:find_by_id:db.query.requested",
    ...     contract_id="users",
    ...     contract_fingerprint="sha256:abc123...",
    ...     operation_name="find_by_id",
    ...     query_fingerprint="sha256:def456...",
    ...     emitted_at=datetime.now(UTC),
    ... )
"""

from omnibase_infra.models.ledger.model_db_query_failed import ModelDbQueryFailed
from omnibase_infra.models.ledger.model_db_query_requested import ModelDbQueryRequested
from omnibase_infra.models.ledger.model_db_query_succeeded import ModelDbQuerySucceeded
from omnibase_infra.models.ledger.model_ledger_event_base import ModelLedgerEventBase

__all__ = [
    "ModelDbQueryFailed",
    "ModelDbQueryRequested",
    "ModelDbQuerySucceeded",
    "ModelLedgerEventBase",
]
