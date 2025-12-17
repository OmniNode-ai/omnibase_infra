# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Context Model for Dual Registration Workflow.

This module provides ModelFSMContext for maintaining context during FSM
execution in the dual registration reducer.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent


class ModelFSMContext(BaseModel):
    """Context maintained during FSM execution.

    Matches context_variables in contracts/fsm/dual_registration_reducer_fsm.yaml.

    Attributes:
        correlation_id: Request correlation ID for distributed tracing.
        node_id: Unique identifier of the node being registered.
        node_type: ONEX node type (effect, compute, reducer, orchestrator).
        introspection_payload: The original introspection event being processed.
        consul_registered: Whether Consul registration succeeded.
        postgres_registered: Whether PostgreSQL registration succeeded.
        success_count: Number of successful registrations (0, 1, or 2).
        consul_error: Error message if Consul registration failed.
        postgres_error: Error message if PostgreSQL registration failed.
        registration_start_time: Start time for performance tracking.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    correlation_id: UUID | None = Field(
        default=None, description="Request correlation ID for distributed tracing"
    )
    node_id: UUID | None = Field(
        default=None, description="Unique identifier of the node being registered"
    )
    node_type: str | None = Field(
        default=None,
        description="ONEX node type (effect, compute, reducer, orchestrator)",
    )
    introspection_payload: ModelNodeIntrospectionEvent | None = Field(
        default=None, description="The original introspection event being processed"
    )
    consul_registered: bool = Field(
        default=False, description="Whether Consul registration succeeded"
    )
    postgres_registered: bool = Field(
        default=False, description="Whether PostgreSQL registration succeeded"
    )
    success_count: int = Field(
        default=0, description="Number of successful registrations (0, 1, or 2)"
    )
    consul_error: str | None = Field(
        default=None, description="Error message if Consul registration failed"
    )
    postgres_error: str | None = Field(
        default=None, description="Error message if PostgreSQL registration failed"
    )
    registration_start_time: float = Field(
        default=0.0, description="Start time for performance tracking"
    )


__all__ = ["ModelFSMContext"]
