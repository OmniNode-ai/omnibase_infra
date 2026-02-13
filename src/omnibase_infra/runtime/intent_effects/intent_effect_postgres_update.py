# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent effect adapter for PostgreSQL registration UPDATE operations.

This module provides the IntentEffectPostgresUpdate adapter, which bridges
ModelPayloadPostgresUpdateRegistration intent payloads to actual PostgreSQL
UPDATE operations via raw SQL on the ProjectorShell's connection pool.

Architecture:
    RegistrationReducerService
        -> ModelPayloadPostgresUpdateRegistration (intent payload)
        -> IntentExecutor
        -> IntentEffectPostgresUpdate.execute()
        -> asyncpg UPDATE query (with monotonic heartbeat guard)

The adapter performs a conditional UPDATE WHERE with a monotonic guard
on last_heartbeat_at to ensure idempotent heartbeat processing:

    UPDATE registration_projections
    SET col1 = $1, col2 = $2, ...
    WHERE entity_id = $N AND domain = $M
      AND (last_heartbeat_at IS NULL OR last_heartbeat_at < $heartbeat_param)

The heartbeat guard is only applied when the updates dict contains a
`last_heartbeat_at` key. For non-heartbeat updates (e.g., ACK state
transitions), the guard is omitted.

Related:
    - ModelPayloadPostgresUpdateRegistration: Intent payload model
    - RegistrationReducerService: Emits this intent from decide_ack / decide_heartbeat
    - IntentEffectPostgresUpsert: Sibling effect for INSERT...ON CONFLICT

.. versionadded:: 0.8.0
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ContainerWiringError, RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.projectors.util_sql_identifiers import quote_identifier
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column type sets for asyncpg normalization.
#
# These sets MUST stay in sync with schema_registration_projection.sql
# (src/omnibase_infra/schemas/schema_registration_projection.sql).
#
# When the database schema adds, renames, or removes UUID or TIMESTAMPTZ
# columns, update the corresponding frozenset below AND the
# _ALLOWED_COLUMNS frozenset on IntentEffectPostgresUpdate.
#
# Recommended: add a schema-sync test that parses the CREATE TABLE DDL
# and asserts parity with these sets.  See the sibling upsert effect
# (intent_effect_postgres_upsert.py) for the same pattern.
# ---------------------------------------------------------------------------

# TIMESTAMPTZ columns that need str -> datetime conversion for asyncpg.
_TIMESTAMP_COLUMNS: frozenset[str] = frozenset(
    {
        "ack_deadline",
        "liveness_deadline",
        "last_heartbeat_at",
        "ack_timeout_emitted_at",
        "liveness_timeout_emitted_at",
        "registered_at",
        "updated_at",
    }
)

# UUID columns that need str -> UUID conversion for asyncpg.
_UUID_COLUMNS: frozenset[str] = frozenset(
    {"entity_id", "last_applied_event_id", "correlation_id"}
)


class IntentEffectPostgresUpdate:
    """Intent effect adapter for PostgreSQL registration UPDATE operations.

    Bridges ModelPayloadPostgresUpdateRegistration intent payloads to
    plain UPDATE queries on the registration_projections table. Includes
    a monotonic guard on last_heartbeat_at for idempotent heartbeat
    processing.

    Thread Safety:
        This class is designed for single-threaded async use. The underlying
        asyncpg pool handles connection concurrency.

    Attributes:
        _pool: asyncpg connection pool for executing UPDATE queries.
        _ALLOWED_COLUMNS: Frozenset of column names permitted in UPDATE SET
            and WHERE clauses. Any column not in this set is rejected before
            query construction to prevent SQL injection via identifier
            interpolation.

    .. versionadded:: 0.8.0
    """

    # -----------------------------------------------------------------------
    # Column allowlist for SQL identifier injection prevention.
    #
    # Every column used in SET or WHERE clauses MUST appear here. This set
    # is the union of all columns in the registration_projections table as
    # defined in schema_registration_projection.sql
    # (src/omnibase_infra/schemas/schema_registration_projection.sql).
    #
    # When the schema changes, update this frozenset AND the type-specific
    # _TIMESTAMP_COLUMNS / _UUID_COLUMNS sets above.
    # -----------------------------------------------------------------------
    _ALLOWED_COLUMNS: frozenset[str] = frozenset(
        {
            # Identity (composite primary key)
            "entity_id",
            "domain",
            # FSM State
            "current_state",
            # Node Information
            "node_type",
            "node_version",
            "capabilities",
            # Capability fields (OMN-1134)
            "contract_type",
            "intent_types",
            "protocols",
            "capability_tags",
            "contract_version",
            # Timeout Deadlines
            "ack_deadline",
            "liveness_deadline",
            "last_heartbeat_at",
            # Timeout Emission Markers
            "ack_timeout_emitted_at",
            "liveness_timeout_emitted_at",
            # Idempotency and Ordering
            "last_applied_event_id",
            "last_applied_offset",
            "last_applied_sequence",
            "last_applied_partition",
            # Timestamps
            "registered_at",
            "updated_at",
            # Tracing
            "correlation_id",
        }
    )

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize the PostgreSQL UPDATE intent effect.

        Args:
            pool: asyncpg connection pool. Must be fully initialized.

        Raises:
            ContainerWiringError: If pool is None.
        """
        if pool is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=uuid4(),
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_init",
            )
            raise ContainerWiringError(
                "asyncpg pool is required for IntentEffectPostgresUpdate",
                context=context,
            )
        self._pool = pool

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute a PostgreSQL UPDATE from an intent payload.

        Builds and executes:
            UPDATE registration_projections
            SET <updates>
            WHERE entity_id = $e AND domain = $d
              [AND (last_heartbeat_at IS NULL OR last_heartbeat_at < $hb)]

        The heartbeat monotonic guard is applied only when the updates
        dict contains a ``last_heartbeat_at`` key.

        Args:
            payload: The ModelPayloadPostgresUpdateRegistration intent payload.
            correlation_id: Optional correlation ID for tracing.

        Raises:
            RuntimeHostError: If the payload type is wrong, updates are empty,
                or the UPDATE query fails.
        """
        effective_correlation_id = (
            correlation_id or getattr(payload, "correlation_id", None) or uuid4()
        )

        if not isinstance(payload, ModelPayloadPostgresUpdateRegistration):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_update",
            )
            raise RuntimeHostError(
                f"Expected ModelPayloadPostgresUpdateRegistration, "
                f"got {type(payload).__name__}",
                context=context,
            )

        updates = payload.updates
        if not updates:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_update",
            )
            raise RuntimeHostError(
                "Intent payload has empty updates dict -- UPDATE would be a no-op",
                context=context,
            )

        try:
            # Normalize types for asyncpg
            normalized_updates = self._normalize_for_asyncpg(dict(updates))
            entity_id = payload.entity_id
            domain = payload.domain

            # ------------------------------------------------------------------
            # Column allowlist validation.
            # Reject any column name not in the schema to prevent identifier
            # injection. Values are already parameterized ($1, $2, ...) but
            # column names are interpolated via quote_identifier(), so this
            # allowlist is a defence-in-depth measure.
            # ------------------------------------------------------------------
            disallowed_set_cols = set(normalized_updates.keys()) - self._ALLOWED_COLUMNS
            if disallowed_set_cols:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=effective_correlation_id,
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="intent_effect_postgres_update",
                )
                raise RuntimeHostError(
                    f"UPDATE SET contains column(s) not in allowlist: "
                    f"{sorted(disallowed_set_cols)}. "
                    f"If this is a new schema column, add it to "
                    f"IntentEffectPostgresUpdate._ALLOWED_COLUMNS.",
                    context=context,
                )

            # Validate WHERE clause keys (entity_id, domain are hardcoded but
            # the check is included for completeness and future-proofing).
            where_columns = {"entity_id", "domain"}
            disallowed_where_cols = where_columns - self._ALLOWED_COLUMNS
            if disallowed_where_cols:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=effective_correlation_id,
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="intent_effect_postgres_update",
                )
                raise RuntimeHostError(
                    f"UPDATE WHERE contains column(s) not in allowlist: "
                    f"{sorted(disallowed_where_cols)}. "
                    f"If this is a new schema column, add it to "
                    f"IntentEffectPostgresUpdate._ALLOWED_COLUMNS.",
                    context=context,
                )

            # Build SET clause
            set_parts: list[str] = []
            params: list[object] = []
            idx = 1
            for col, val in normalized_updates.items():
                set_parts.append(f"{quote_identifier(col)} = ${idx}")
                params.append(val)
                idx += 1

            set_clause = ", ".join(set_parts)

            # WHERE clause: entity_id + domain
            where_parts = [
                f"{quote_identifier('entity_id')} = ${idx}",
            ]
            params.append(entity_id)
            idx += 1

            where_parts.append(f"{quote_identifier('domain')} = ${idx}")
            params.append(domain)
            idx += 1

            # Monotonic guard for heartbeat idempotency
            heartbeat_val = normalized_updates.get("last_heartbeat_at")
            if heartbeat_val is not None:
                where_parts.append(
                    f"({quote_identifier('last_heartbeat_at')} IS NULL "
                    f"OR {quote_identifier('last_heartbeat_at')} < ${idx})"
                )
                params.append(heartbeat_val)
                idx += 1

            where_clause = " AND ".join(where_parts)

            # S608: Safe -- identifiers quoted, values parameterized
            sql = (
                f"UPDATE {quote_identifier('registration_projections')} "  # noqa: S608
                f"SET {set_clause} "
                f"WHERE {where_clause}"
            )

            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, *params, timeout=30.0)

            # Parse row count (e.g., "UPDATE 1" -> 1)
            rows_affected = 0
            parts = result.split()
            if parts and parts[-1].isdigit():
                rows_affected = int(parts[-1])

            logger.info(
                "PostgreSQL UPDATE executed: entity_id=%s rows=%d correlation_id=%s",
                str(entity_id),
                rows_affected,
                str(effective_correlation_id),
            )

        except RuntimeHostError:
            raise
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_update",
            )
            logger.warning(
                "PostgreSQL UPDATE intent failed: error=%s correlation_id=%s",
                sanitize_error_message(e),
                str(effective_correlation_id),
                extra={"error_type": type(e).__name__},
            )
            raise RuntimeHostError(
                "Failed to execute PostgreSQL UPDATE intent",
                context=context,
            ) from e

    @staticmethod
    def _normalize_for_asyncpg(
        record: dict[str, object],
    ) -> dict[str, object]:
        """Normalize values from JSON-serializable types to asyncpg-native types.

        Converts string UUIDs to UUID objects and ISO datetime strings to
        datetime objects, matching the schema column types.

        Args:
            record: Dict of column name -> value.

        Returns:
            New dict with UUID and datetime columns converted to native types.
        """
        normalized: dict[str, object] = {}
        for key, value in record.items():
            if value is None:
                normalized[key] = value
            elif key in _UUID_COLUMNS:
                normalized[key] = (
                    UUID(str(value)) if not isinstance(value, UUID) else value
                )
            elif key in _TIMESTAMP_COLUMNS:
                if isinstance(value, str):
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    normalized[key] = dt
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        return normalized


__all__: list[str] = ["IntentEffectPostgresUpdate"]
