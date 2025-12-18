# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer Metrics Model for Dual Registration Workflow.

This module provides ModelReducerMetrics for tracking aggregation metrics
across all processed events in the dual registration reducer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelReducerMetrics:
    """Aggregation metrics tracked by the reducer.

    Tracks registration outcomes across all processed events for monitoring
    and observability purposes.

    Note:
        These metrics are stored **in-memory only** and will reset when the
        service restarts. For persistent metrics collection, integrate with
        an external metrics backend (Prometheus, StatsD, etc.).

    Attributes:
        total_registrations: Total number of registration attempts processed.
        success_count: Number of fully successful registrations (both backends).
        failure_count: Number of completely failed registrations (both backends).
        partial_count: Number of partial registrations (one backend succeeded).
    """

    total_registrations: int = 0
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0


__all__ = ["ModelReducerMetrics"]
