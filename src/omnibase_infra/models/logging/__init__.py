# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Structured Logging Models.

This module provides Pydantic models for structured logging in ONEX
infrastructure components.

Models:
    - **ModelLogContext**: Structured log context replacing ``dict[str, str | int | float]``

Design Principles:
    - **Type Safety**: Strong typing with validation constraints
    - **Immutable**: All models are frozen (thread-safe after creation)
    - **Builder Pattern**: Fluent interface for constructing log contexts
    - **Backwards Compatible**: ``to_dict()`` method for standard logging formatters

Example:
    >>> from omnibase_infra.models.logging import ModelLogContext
    >>> from uuid import uuid4
    >>>
    >>> # Using builder pattern
    >>> ctx = (
    ...     ModelLogContext.for_operation("dispatch")
    ...     .with_correlation_id(str(uuid4()))
    ...     .with_duration_ms(42.5)
    ...     .with_service_name("kafka-event-bus")
    ... )
    >>>
    >>> # Use in logging
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("Event dispatched", extra=ctx.to_dict())
    >>>
    >>> # Using factory methods
    >>> ctx = ModelLogContext.for_event_bus(
    ...     operation="publish",
    ...     service_name="kafka",
    ...     topic="dev.user.events.v1",
    ... )

See Also:
    omnibase_infra.models.dispatch.ModelDispatchLogContext: Dispatch-specific logging context

.. versionadded:: 0.6.0
    Added as part of OMN-1002 Union Reduction Phase 2.
"""

from omnibase_infra.models.logging.model_log_context import ModelLogContext

__all__ = ["ModelLogContext"]
