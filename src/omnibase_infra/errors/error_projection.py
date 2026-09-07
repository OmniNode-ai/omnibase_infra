# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ProjectionError — raised by NodeProjectionEffect on projection failure.

Projection failure blocks Kafka publish (OMN-2363 / OMN-2510).  This error
class makes that failure explicit and carries enough context for the runtime
to log a full incident report before routing to retry / dead-letter.

Error hierarchy:
    ModelOnexError (omnibase_core)
    └── RuntimeHostError (omnibase_infra)
        └── ProjectionError   ← this module

Related:
    - OMN-2508: NodeProjectionEffect (omnibase_spi)
    - OMN-2510: Runtime wires projection before Kafka publish
    - error_infra.py: RuntimeHostError base class
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.errors.error_infra import RuntimeHostError

if TYPE_CHECKING:
    from omnibase_infra.models.errors.model_infra_error_context import (
        ModelInfraErrorContext,
    )


class ProjectionError(RuntimeHostError):
    """Raised when a synchronous projection write fails.

    The runtime catches this error in DispatchResultApplier and:
        1. Skips Kafka publish entirely.
        2. Routes the originating message to retry / dead-letter handling.
        3. Logs the failure with projector_key, event_type, and exception details.

    Attributes:
        originating_event_id: UUID of the event that triggered the projection.
        projection_type: The projection table / projector class that failed.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.errors.error_projection import ProjectionError
        >>> from omnibase_infra.models.errors.model_infra_error_context import (
        ...     ModelInfraErrorContext,
        ... )
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>>
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="projection_effect.execute",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise ProjectionError(
        ...     "NodeRegistration projection write failed — connection refused",
        ...     context=context,
        ...     originating_event_id=uuid4(),
        ...     projection_type="NodeRegistration",
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        originating_event_id: UUID | None = None,
        projection_type: str | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ProjectionError with structured projection context.

        Args:
            message: Human-readable description of the projection failure.
            context: Infrastructure context (transport_type, operation,
                correlation_id).
            originating_event_id: UUID of the event that triggered this
                projection.  Included in logs for correlation across services.
            projection_type: The projection table or projector class name.
                Helps operators quickly identify which projector failed.
            **extra_context: Additional context fields forwarded to
                RuntimeHostError for structured logging.
        """
        # Inject projection-specific fields into extra_context before
        # forwarding to RuntimeHostError (same pattern as RepositoryError).
        if originating_event_id is not None:
            extra_context["originating_event_id"] = str(originating_event_id)
        if projection_type is not None:
            extra_context["projection_type"] = projection_type

        super().__init__(message, error_code=None, context=context, **extra_context)

        # Store typed attributes for programmatic access
        self.originating_event_id = originating_event_id
        self.projection_type = projection_type


class ProjectionTenantContextError(ProjectionError):
    """Raised when a tenant projection has no valid authenticated authority.

    OMN-15421. Tenant-scoped projection tables compare their UUID tenant key
    with the transaction-local ``app.tenant_id`` setting. The adapter accepts
    only an opaque capability minted after canonical signed-envelope verification
    and an authoritative signer-to-tenant binding check. Ordinary security-context
    fields, gateway metadata, request/payload values, environment values, empty
    strings, slugs, and shared sentinels are never authority or fallbacks.

    Distinct from the generic :class:`ProjectionError` so callers and operators
    can tell a tenant-attribution refusal apart from a connection or schema
    failure; the two have completely different remediations.
    """


class ProjectionNotMaterializedError(ProjectionError):
    """A projection consumed an event and wrote no row for a NON-content reason.

    OMN-17379. The auto-wired projection dispatch callback caught every handler
    exception, logged one ERROR line, best-effort-DLQ'd, and returned ``None``.
    A callback that returns normally IS an ACK: the consume boundary reads "no
    exception" as success and the offset advances. The fact the event carried is
    then gone from the projection forever while every external surface still
    reports health.

    Live proof on the ``.201`` dev lane (2026-08-31): ``pr_merged_events`` held
    28 rows whose newest was 2026-08-03 while its consumer group sat at
    ``Stable / TOTAL-LAG 0 / CURRENT-OFFSET 97 = LOG-END``. Rewinding the group
    to offset 94 and letting the real wired path re-consume 94→96 produced three

        InsufficientPrivilege: permission denied for sequence
        pr_merged_events_projection_cursor_seq

    errors, three quarantine records, ZERO rows, and a committed offset back at
    97. 230 merged PRs' worth of facts were acknowledged into nothing.

    The distinction this type encodes is the one the old code did not make:

    * A **content** failure (a ``ValidationError`` on a malformed payload) is the
      EVENT's defect. Redelivery can never fix it, so DLQ-and-advance stays
      correct and this type is NOT raised.
    * A **write-path** failure — insufficient privilege, a dead connection, a
      missing relation, a wiring bug that denies the handler its adapter — is the
      RUNTIME's defect. The event is valid and still owed a row, so the offset
      must be withheld and Kafka must redeliver once the write path is repaired.

    Raising it is the whole mechanism: ``EventBusKafka._dispatch_to_subscriber``
    classifies this type as offset-unsafe unconditionally, which rewinds the fetch
    position to the failed message's own offset (the OMN-15232
    ``_rewind_after_unpersisted_dlq`` path). That is the only action that works
    under ``enable_auto_commit=True``, where merely declining to commit does
    nothing and the offset advances anyway.

    The consequence is deliberate and is the point: a projection whose write path
    is broken now STALLS with visible lag instead of running green at lag 0. A
    stalled feed is a detectable feed.
    """


class ProjectionQueryRowBudgetError(ProjectionNotMaterializedError):
    """A projection read matched more rows than the seam will materialise.

    OMN-17888. ``ProjectionDatabaseOperations._execute_query`` emitted
    ``SELECT *`` with no row bound for any caller and then materialised the
    result set twice, both copies alive at once
    (``[dict(record) for record in cursor.fetchall()]``). Measured in the
    deployed container against the live ``omnidash_analytics`` database, one
    call over the ``session_id`` holding 91,633 of
    ``public.session_replay_snapshots``' 94,571 rows cost 225.4 MiB: 200.7 MiB
    of driver rows plus 24.8 MiB of plain dicts. ``asyncio.to_thread`` ran the
    blocking handler on the loop's default 32-worker executor, so four to five
    concurrent calls produced the ~1,062 MB RSS step that memcg-OOM-killed
    ``onex-runtime`` on the ``.201`` DEV lane about every three minutes.

    Raising is the deliberate choice over appending a ``LIMIT``. A truncated
    answer is indistinguishable from a complete one at the call site, so a
    ``LIMIT`` would convert an OOM into silently wrong projections -- the
    silent-fallback shape this codebase refuses everywhere else. The refusal
    names the relation, the bound, and the filter keys so the offending caller
    is identifiable from one log line.

    Classified as a write-path failure, not a content failure: the event is
    well-formed and still owed a row, so the offset must be withheld and the
    record redelivered once the caller is repaired. The remedy is never to
    raise the bound -- it is to make the caller ask a bounded question (an
    indexed single-row read, or a paged one), which is what the read it
    replaced already needed.

    IT IS A SUBCLASS OF :class:`ProjectionNotMaterializedError`, AND THAT IS
    THE WHOLE MECHANISM. Until OMN-17888 second pass it was a direct sibling
    under :class:`ProjectionError` and was caught NOWHERE. Every offset-unsafe
    arm in the runtime matches ``ProjectionNotMaterializedError`` by EXACT type
    (``event_bus_kafka._dispatch_to_subscriber``,
    ``message_dispatch_engine._dispatch_one``, and both
    ``handler_wiring`` boundary arms), so a sibling fell through to the generic
    bounded-retry loop, then to ``_route_swallowed_exception`` -> DLQ -> and the
    offset ADVANCED. The paragraph above claimed the opposite of what the code
    did: the event was acknowledged into a dead-letter record and the row it was
    owed was never written -- the exact OMN-17379 swallow this error was
    supposed to be on the safe side of.

    Subclassing rather than widening four ``except`` tuples is deliberate. The
    two types state the SAME fact -- "this projection consumed an event, wrote
    no row, and the cause is the runtime rather than the event" -- so the
    relationship belongs in the hierarchy, where a fifth arm added later
    inherits it, instead of in four call sites that a fifth arm can forget.
    """


__all__ = [
    "ProjectionError",
    "ProjectionNotMaterializedError",
    "ProjectionQueryRowBudgetError",
    "ProjectionTenantContextError",
]
