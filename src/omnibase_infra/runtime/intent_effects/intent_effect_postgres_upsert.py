# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent effect adapter for PostgreSQL registration upserts.

This module provides the IntentEffectPostgresUpsert adapter, which bridges
ModelPayloadPostgresUpsertRegistration intent payloads to actual PostgreSQL
upsert operations via the ProjectorShell.

Architecture:
    HandlerNodeIntrospected
        -> ModelPayloadPostgresUpsertRegistration (intent payload)
        -> IntentExecutor
        -> IntentEffectPostgresUpsert.execute()
        -> ProjectorShell.upsert_partial() (PostgreSQL)

    The adapter extracts the ``record`` field from the intent payload,
    converts it to a dict of column values, and delegates to the
    ProjectorShell for atomic upsert with ordering enforcement.

Related:
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
    - ModelPayloadPostgresUpsertRegistration: Intent payload model
    - ProjectorShell: Declarative projector with upsert_partial()
    - schema_registration_projection.sql: Target table schema

.. versionadded:: 0.7.0
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
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.runtime.projector_shell import ProjectorShell

logger = logging.getLogger(__name__)


class IntentEffectPostgresUpsert:
    """Intent effect adapter for PostgreSQL registration upserts.

    Bridges ModelPayloadPostgresUpsertRegistration intent payloads to
    ProjectorShell upsert operations. The adapter extracts the serialized
    projection record from the payload and persists it via upsert_partial().

    Thread Safety:
        This class is designed for single-threaded async use. The underlying
        ProjectorShell handles connection pool concurrency.

    Attributes:
        _projector: ProjectorShell for registration projection upserts.

    Example:
        ```python
        effect = IntentEffectPostgresUpsert(projector=projector_shell)
        await effect.execute(payload, correlation_id=correlation_id)
        ```

    .. versionadded:: 0.7.0
    """

    def __init__(self, projector: ProjectorShell) -> None:
        """Initialize the PostgreSQL upsert intent effect.

        Args:
            projector: ProjectorShell for registration projection upserts.
                Must be fully initialized with a valid connection pool.

        Raises:
            ContainerWiringError: If projector is None.
        """
        if projector is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=uuid4(),
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_init",
            )
            raise ContainerWiringError(
                "ProjectorShell is required for IntentEffectPostgresUpsert",
                context=context,
            )
        self._projector = projector

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Execute a PostgreSQL upsert from an intent payload.

        Extracts the ``record`` field from the payload, serializes it to a
        dict, and delegates to ProjectorShell.upsert_partial() for atomic
        PostgreSQL upsert.

        The record is expected to contain all columns for the
        registration_projections table, including entity_id for conflict
        resolution.

        Args:
            payload: The ModelPayloadPostgresUpsertRegistration intent payload.
                Validated via isinstance at entry.
            correlation_id: Optional correlation ID for tracing.
                Falls back to payload.correlation_id if not provided.

        Raises:
            RuntimeHostError: If the upsert operation fails.
        """
        # Compute effective correlation_id before type checks so error contexts
        # always carry a non-None ID, preserving any ID from the payload when
        # available and falling back to uuid4() only as a last resort.
        effective_correlation_id = (
            correlation_id or getattr(payload, "correlation_id", None) or uuid4()
        )

        if not isinstance(payload, ModelPayloadPostgresUpsertRegistration):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_upsert",
            )
            raise RuntimeHostError(
                f"Expected ModelPayloadPostgresUpsertRegistration, "
                f"got {type(payload).__name__}",
                context=context,
            )

        try:
            # Serialize the record to a dict of column values
            record = payload.record
            if record is None:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=effective_correlation_id,
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="upsert_registration",
                )
                raise RuntimeHostError(
                    "Intent payload has no record — upsert would be lost "
                    "(malformed intent payload)",
                    context=context,
                )

            # model_dump() returns explicit fields + the ``data`` dict.
            # Merge ``data`` into top-level so the projector receives a flat
            # column dict matching the database schema.
            record_dict = record.model_dump()
            data_fields = record_dict.pop("data", {})
            record_dict.update(data_fields)

            # Normalize types for asyncpg: handler sends JSON-serializable
            # strings but asyncpg requires native Python types for UUID,
            # TIMESTAMPTZ, etc. columns.
            record_dict = self._normalize_for_asyncpg(record_dict)

            # Extract entity_id for the projector's aggregate_id parameter.
            # Missing entity_id is a data integrity error — raise so the caller
            # prevents Kafka offset commit rather than silently losing the upsert.
            entity_id = record_dict.get("entity_id")
            if entity_id is None:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=effective_correlation_id,
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="upsert_registration",
                )
                raise RuntimeHostError(
                    "Intent record missing required entity_id field",
                    context=context,
                )

            # Convert to UUID (may arrive as str from model_dump)
            aggregate_id = UUID(entity_id) if isinstance(entity_id, str) else entity_id
            if not isinstance(aggregate_id, UUID):
                aggregate_id = UUID(str(aggregate_id))

            await self._projector.upsert_partial(
                aggregate_id=aggregate_id,
                values=record_dict,
                correlation_id=effective_correlation_id,
                conflict_columns=["entity_id", "domain"],
            )

            logger.info(
                "PostgreSQL upsert executed: entity_id=%s correlation_id=%s",
                str(entity_id),
                str(effective_correlation_id),
            )

        except RuntimeHostError:
            raise
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="intent_effect_postgres_upsert",
            )
            logger.warning(
                "PostgreSQL upsert intent failed: error=%s correlation_id=%s",
                sanitize_error_message(e),
                str(effective_correlation_id),
                extra={
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeHostError(
                "Failed to execute PostgreSQL upsert intent",
                context=context,
            ) from e

    # -------------------------------------------------------------------------
    # Column type sets for asyncpg normalization.
    #
    # These sets MUST stay in sync with schema_registration_projection.sql.
    # When wiring a NEW intent type that writes to registration_projections,
    # check if it introduces columns not listed here and add them:
    #
    #   1. Open schema_registration_projection.sql
    #   2. Find all UUID and TIMESTAMPTZ columns in the CREATE TABLE statement
    #   3. If your new intent writes to any of those columns, ensure the column
    #      name appears in the appropriate set below
    #   4. Add a test case in tests/unit/runtime/test_intent_effect_postgres_upsert.py
    #      that verifies normalization for the new columns
    #
    # CURRENT INTENT COVERAGE:
    #   - postgres.upsert_registration (HandlerNodeIntrospected)
    #     UUID cols: entity_id, last_applied_event_id, correlation_id
    #     TIMESTAMPTZ cols: ack_deadline, registered_at, updated_at
    #
    # All schema UUID and TIMESTAMPTZ columns are included in the sets
    # below. The normalizer is a no-op for None values, so including
    # columns not yet written by any intent type is safe and prevents
    # silent corruption when new intent types are wired.
    # -------------------------------------------------------------------------

    # UUID columns in registration_projections.
    # Schema reference: entity_id UUID, last_applied_event_id UUID, correlation_id UUID
    # Validated by: tests/unit/runtime/test_intent_effect_postgres_upsert.py
    _UUID_COLUMNS: frozenset[str] = frozenset(
        {"entity_id", "last_applied_event_id", "correlation_id"}
    )

    # TIMESTAMPTZ columns in registration_projections.
    # Schema reference: all TIMESTAMPTZ columns from schema_registration_projection.sql
    # Validated by: tests/unit/runtime/test_intent_effect_postgres_upsert.py
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

    @staticmethod
    def _normalize_for_asyncpg(
        record: dict[str, object],
    ) -> dict[str, object]:
        """Normalize record values from JSON-serializable types to asyncpg-native types.

        Handlers produce JSON-serializable values (string UUIDs, ISO datetime
        strings). asyncpg requires native Python types (UUID, datetime) for
        parameterized queries against PostgreSQL UUID and TIMESTAMPTZ columns.

        Args:
            record: Dict of column name → value from model_dump().

        Returns:
            New dict with UUID and datetime columns converted to native types.
        """
        normalized: dict[str, object] = {}
        for key, value in record.items():
            if value is None:
                normalized[key] = value
            elif key in IntentEffectPostgresUpsert._UUID_COLUMNS:
                normalized[key] = (
                    UUID(str(value)) if not isinstance(value, UUID) else value
                )
            elif key in IntentEffectPostgresUpsert._TIMESTAMP_COLUMNS:
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


__all__: list[str] = ["IntentEffectPostgresUpsert"]
