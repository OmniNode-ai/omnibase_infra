# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""NodeDeltaMetricsEffect -- per-model performance rollup effect node.

Upserts aggregated metrics into delta_metrics_by_model from completed
delta bundles. Uses ON CONFLICT DO UPDATE to increment counters atomically.

Architecture:
    - node.py: Declarative node shell extending NodeEffect
    - handlers/: PostgreSQL operation handler (upsert_metrics)
    - models/: Payload model for metrics rollup
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Intent routing and I/O definitions

Node Type: EFFECT_GENERIC
Purpose: Execute PostgreSQL I/O for delta metrics rollup.

Related:
    - OMN-3142: Implementation ticket
    - Migration 040: delta_metrics_by_model table
"""

from __future__ import annotations

from omnibase_infra.nodes.node_delta_metrics_effect.handlers import (
    HandlerUpsertMetrics,
)
from omnibase_infra.nodes.node_delta_metrics_effect.node import (
    NodeDeltaMetricsEffect,
)
from omnibase_infra.nodes.node_delta_metrics_effect.registry import (
    RegistryInfraDeltaMetricsEffect,
)

__all__: list[str] = [
    # Node
    "NodeDeltaMetricsEffect",
    # Registry
    "RegistryInfraDeltaMetricsEffect",
    # Handlers
    "HandlerUpsertMetrics",
]
