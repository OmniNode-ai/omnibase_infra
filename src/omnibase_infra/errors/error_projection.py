# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
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
        3. Logs the failure with originating event ID, ModelProjectionIntent,
           and exception details.

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


__all__ = ["ProjectionError"]
