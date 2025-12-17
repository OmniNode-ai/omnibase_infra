# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""FSM Context Model for Dual Registration Workflow.

This module provides ModelFSMContext for maintaining context during FSM
execution in the dual registration reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.models.registration import ModelNodeIntrospectionEvent


@dataclass
class ModelFSMContext:
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

    correlation_id: UUID | None = None
    node_id: str | None = None
    node_type: str | None = None
    introspection_payload: ModelNodeIntrospectionEvent | None = None
    consul_registered: bool = False
    postgres_registered: bool = False
    success_count: int = 0
    consul_error: str | None = None
    postgres_error: str | None = None
    registration_start_time: float = 0.0


__all__ = ["ModelFSMContext"]
