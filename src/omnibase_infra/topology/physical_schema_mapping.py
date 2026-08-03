# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Temporary logical-to-physical schema mapping for application table grants."""

from __future__ import annotations

TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359: frozenset[str] = frozenset(
    {
        "agent_routing_decisions",
        "capability_scores",
        "context_roi_scores",
        "delegation_budget_state",
        "delegation_events",
        "delegation_judge_verdict_events",
        "delegation_shadow_comparisons",
        "dep_health_findings",
        "instruction_eval_aggregate_snapshots",
        "llm_cost_aggregates",
        "pattern_learning_artifacts",
        "projection_delegation_inference_response_text",
        "savings_estimates",
        "skill_execution_snapshots",
    }
)


def physical_grant_schema_for_table(schema: str, table_name: str) -> str:
    """Return the schema where PostgreSQL ACLs currently apply for a table.

    OMN-15359 owns the physical ``public`` -> ``tenant`` move. Until that lands,
    these tenant-domain projection tables remain physically in ``public`` even
    though their contracts and runtime routing are logically tenant-domain.
    """
    if (
        schema == "tenant"
        and table_name in TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    ):
        return "public"
    return schema
