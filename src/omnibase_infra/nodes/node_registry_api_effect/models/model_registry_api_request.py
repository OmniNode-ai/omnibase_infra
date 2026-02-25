# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for NodeRegistryApiEffect operations.

Ticket: OMN-1441
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from omnibase_infra.enums import EnumRegistrationState


class ModelRegistryApiRequest(BaseModel):
    """Input envelope for registry API effect operations.

    All fields are optional — callers populate only the fields relevant to
    the specific operation being dispatched.  This mirrors the flexible
    parameter surface of the existing FastAPI route handlers.

    Attributes:
        operation: The operation to perform (matches ``io_operations`` in
            ``contract.yaml``).
        correlation_id: Distributed tracing identifier.
        limit: Pagination page size.
        offset: Pagination offset.
        state: Optional registration state filter for node listing.
        node_type: Optional node type filter (effect, compute, …).
        node_id: UUID for single-node lookup operations.
        contract_id: UUID for single-contract lookup operations.
        topic_suffix: Topic suffix for single-topic lookup operations.
    """

    operation: str = Field(
        description="Operation identifier (see contract.yaml io_operations)."
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing.",
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Page size.")
    offset: int = Field(default=0, ge=0, description="Pagination offset.")
    state: EnumRegistrationState | None = Field(
        default=None, description="Optional registration state filter."
    )
    node_type: str | None = Field(
        default=None, description="Optional node type filter."
    )
    node_id: UUID | None = Field(
        default=None, description="Node UUID for single-node lookup."
    )
    contract_id: UUID | None = Field(
        default=None, description="Contract UUID for single-contract lookup."
    )
    topic_suffix: str | None = Field(
        default=None, description="Topic suffix for single-topic lookup."
    )

    model_config = {"extra": "forbid", "frozen": True}


__all__ = ["ModelRegistryApiRequest"]
