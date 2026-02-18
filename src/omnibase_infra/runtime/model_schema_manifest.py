# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Schema manifest model and canonical table list for omnibase_infra.

Defines the explicit allowlist of tables owned by omnibase_infra. Used by
the schema fingerprint utility to compute and validate fingerprints against
a known set of tables.

Related:
    - OMN-2087: Handshake hardening -- Schema fingerprint manifest + startup assertion
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSchemaManifest(BaseModel):
    """Frozen manifest declaring tables owned by a service.

    The manifest is the source of truth for which tables are included in
    the schema fingerprint computation. Tables not listed here are ignored
    during fingerprint validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    owner_service: str = Field(..., description="Service that owns these tables")
    tables: tuple[str, ...] = Field(
        ..., description="Explicit allowlist of owned tables"
    )
    schema_name: str = Field(default="public", description="PostgreSQL schema name")


# Code constant -- the canonical table list for omnibase_infra.
# Keep sorted alphabetically for deterministic fingerprint computation.
OMNIBASE_INFRA_SCHEMA_MANIFEST = ModelSchemaManifest(
    owner_service="omnibase_infra",
    schema_name="public",
    tables=(
        "agent_actions",
        "agent_detection_failures",
        "agent_execution_logs",
        "agent_routing_decisions",
        "agent_status_events",
        "agent_transformation_events",
        "baselines_breakdown",
        "baselines_comparisons",
        "baselines_trend",
        "contracts",
        "db_metadata",
        "injection_effectiveness",
        "latency_breakdowns",
        "llm_call_metrics",
        "llm_cost_aggregates",
        "pattern_hit_rates",
        "registration_projections",
        "router_performance_metrics",
        "topics",
    ),
)


__all__ = ["ModelSchemaManifest", "OMNIBASE_INFRA_SCHEMA_MANIFEST"]
