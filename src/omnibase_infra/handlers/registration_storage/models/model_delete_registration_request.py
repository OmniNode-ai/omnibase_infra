# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Delete Registration Request Model.

This module provides ModelDeleteRegistrationRequest, representing a request
to delete an existing registration record.

Architecture:
    This request model encapsulates all parameters needed for the
    delete_registration protocol method, following ONEX patterns
    of using Pydantic models instead of primitive parameters.

Related:
    - ModelDeleteResult: Result model for delete operations
    - ProtocolRegistrationStorageHandler: Protocol that uses this request
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDeleteRegistrationRequest(BaseModel):
    """Request model for deleting a registration record.

    Encapsulates all parameters needed for the delete_registration operation,
    enabling type-safe API boundaries and consistent request validation.

    Immutability:
        This model uses frozen=True to ensure requests are immutable
        once created, enabling safe reuse and logging.

    Attributes:
        node_id: ID of the node to delete.
        correlation_id: Optional correlation ID for tracing.

    Example:
        >>> from uuid import uuid4
        >>> request = ModelDeleteRegistrationRequest(
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(
        description="ID of the node to delete",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Optional correlation ID for tracing",
    )


__all__ = ["ModelDeleteRegistrationRequest"]
