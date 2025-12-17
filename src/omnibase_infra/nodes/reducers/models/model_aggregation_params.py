# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Aggregation Parameters Model for Dual Registration Workflow.

This module provides ModelAggregationParams for encapsulating the parameters
passed to the result aggregation function in the dual registration reducer.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelAggregationParams(BaseModel):
    """Parameters for result aggregation in dual registration.

    Encapsulates the parameters needed by the _aggregate_results method
    to combine Consul and PostgreSQL registration outcomes.

    Attributes:
        consul_result: Consul registration result (True for success, exception on failure).
        postgres_result: PostgreSQL registration result (True for success, exception on failure).
        node_id: Unique identifier of the registered node.
        correlation_id: Request correlation ID for distributed tracing.
        registration_time_ms: Total registration time in milliseconds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    consul_result: bool | BaseException = Field(
        ...,
        description="Consul registration result (True for success, exception on failure)",
    )
    postgres_result: bool | BaseException = Field(
        ...,
        description="PostgreSQL registration result (True for success, exception on failure)",
    )
    node_id: UUID = Field(..., description="Unique identifier of the registered node")
    correlation_id: UUID = Field(
        ..., description="Request correlation ID for distributed tracing"
    )
    registration_time_ms: float = Field(
        ..., ge=0.0, description="Total registration time in milliseconds"
    )

    @field_validator("consul_result", "postgres_result", mode="before")
    @classmethod
    def validate_result_type(cls, v: bool | BaseException) -> bool | BaseException:
        """Validate that result is either bool or BaseException."""
        if isinstance(v, bool) or isinstance(v, BaseException):
            return v
        raise ValueError(
            f"Result must be bool or BaseException, got {type(v).__name__}"
        )


__all__ = ["ModelAggregationParams"]
