# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Query Metadata Model for Service Discovery Operations.

This module provides ModelQueryMetadata, representing metadata about
query execution in service discovery operations.

Architecture:
    ModelQueryMetadata contains information about how a query was executed,
    useful for debugging and observability:
    - backend_type: Type of backend that processed the query
    - query_duration_ms: Time taken to execute the query
    - timestamp: When the query was executed
    - cached: Whether the result was served from cache

Related:
    - ModelDiscoveryResult: Primary result model that uses this metadata
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelQueryMetadata(BaseModel):
    """Metadata about query execution.

    Contains information about how the query was executed,
    useful for debugging and observability.

    Attributes:
        backend_type: Type of backend that processed the query.
        query_duration_ms: Time taken to execute the query in milliseconds.
        timestamp: When the query was executed.
        cached: Whether the result was served from cache.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    backend_type: str = Field(
        ...,
        description="Type of backend that processed the query (consul, k8s, etcd)",
    )
    query_duration_ms: float | None = Field(
        default=None,
        ge=0,
        description="Time taken to execute the query in milliseconds",
    )
    timestamp: datetime | None = Field(
        default=None,
        description="When the query was executed",
    )
    cached: bool = Field(
        default=False,
        description="Whether the result was served from cache",
    )


__all__ = ["ModelQueryMetadata"]
