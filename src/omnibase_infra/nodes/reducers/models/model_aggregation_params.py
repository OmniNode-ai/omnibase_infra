# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Aggregation Parameters Model for Dual Registration Workflow.

This module provides ModelAggregationParams for encapsulating the parameters
passed to the result aggregation function in the dual registration reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass
class ModelAggregationParams:
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

    consul_result: bool | BaseException
    postgres_result: bool | BaseException
    node_id: str
    correlation_id: UUID
    registration_time_ms: float


__all__ = ["ModelAggregationParams"]
