# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Update Registration Request Model.

This module provides ModelUpdateRegistrationRequest, representing a request
to update an existing registration record.

Architecture:
    This request model encapsulates all parameters needed for the
    update_registration protocol method, following ONEX patterns
    of using Pydantic models instead of primitive parameters.

Related:
    - ModelRegistrationUpdate: The update payload containing fields to update
    - ProtocolRegistrationStorageHandler: Protocol that uses this request
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelRegistrationUpdate,
)


class ModelUpdateRegistrationRequest(BaseModel):
    """Request model for updating a registration record.

    Encapsulates all parameters needed for the update_registration operation,
    enabling type-safe API boundaries and consistent request validation.

    Immutability:
        This model uses frozen=True to ensure requests are immutable
        once created, enabling safe reuse and logging.

    Attributes:
        node_id: ID of the node to update.
        updates: ModelRegistrationUpdate containing fields to update.
            Only non-None fields will be applied.
        correlation_id: Optional correlation ID for tracing.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.nodes.node_registration_storage_effect.models import (
        ...     ModelRegistrationUpdate,
        ... )
        >>> request = ModelUpdateRegistrationRequest(
        ...     node_id=uuid4(),
        ...     updates=ModelRegistrationUpdate(
        ...         endpoints={"health": "http://localhost:8080/health"},
        ...     ),
        ...     correlation_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(
        description="ID of the node to update",
    )
    updates: ModelRegistrationUpdate = Field(
        description="Update payload containing fields to update (non-None fields applied)",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Optional correlation ID for tracing",
    )


__all__ = ["ModelUpdateRegistrationRequest"]
