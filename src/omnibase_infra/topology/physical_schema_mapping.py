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
        # OMN-15631: node_delegation_routing_reducer's tenant overlay table.
        # Same bridge as its sibling delegation_events above -- logically
        # tenant-domain per contract.yaml (ADR-0027), physically created bare
        # in `public` because no `tenant` Postgres schema exists on any lane
        # today (live-confirmed 2026-08-19, see delegation_events' own
        # comment). Enumerated here so
        # physical_grant_schema_for_table('tenant',
        # 'delegation_routing_tenant_overlay') resolves to 'public', matching
        # the migration's actual bare CREATE TABLE.
        "delegation_routing_tenant_overlay",
        "delegation_shadow_comparisons",
        "dep_health_findings",
        # OMN-16090: hook_events is tenant-domain by contract and RLS policy,
        # but its node migration physically created the relation bare in
        # public before the tenant schema cutover. Keep ACL checks aligned with
        # the live relation until OMN-15359 moves the family.
        "hook_events",
        "instruction_eval_aggregate_snapshots",
        "llm_cost_aggregates",
        "pattern_learning_artifacts",
        "projection_delegation_inference_response_text",
        # OMN-15533: these node_projection_savings read views were physically
        # created bare in public by migrations 076/078/079, and the dashboard
        # projection_api exposures already declare schema: public. Migration
        # 083 replaces those existing public views without moving authority.
        "projection_delegation_savings",
        "projection_delegation_savings_series",
        "savings_estimates",
        "skill_execution_snapshots",
        # OMN-16316: node_projection_tenant_credentials' BYOK inference-
        # credential ref catalog. Same bridge as delegation_events/
        # delegation_routing_tenant_overlay above -- logically tenant-domain
        # (per-tenant credential-ref rows, house-tenant ruling), physically
        # created bare in `public` because no `tenant` Postgres schema exists
        # on any lane today. Enumerated here so
        # physical_grant_schema_for_table('tenant',
        # 'tenant_inference_credentials') resolves to 'public' and the
        # application_database_sql_gate accepts the migration's bare
        # CREATE TABLE, matching its actual physical location. No RLS in v1,
        # same dev/beta-only posture as its siblings -- promotable later once
        # the tenant-schema RLS foundation (OMN-14894/OMN-15356) lands.
        "tenant_inference_credentials",
    }
)

# OMN-15359 (P2-P4 build). The `omninode_internal` schema now physically exists
# (docker/migrations/forward/098_create_omninode_internal_schema.sql), but the
# tables that are logically OMNINODE_INTERNAL-domain per the shipped topology
# (`omninode_runtime` principal's TABLE grants, identical across all 7 shipped
# database-topology profiles as of this ticket) have not been copied into it —
# every one of them is still physically created, unqualified, in `public` by
# its node migration. This is the exact gap OMN-15426's live readback named:
# handler_wiring issues schema-qualified SQL against `omninode_internal` for
# these relations while they resolve nowhere, producing `relation does not
# exist` rather than a permission error. Enumerated (not a blanket public->
# internal rule) so a *new* table declared against `omninode_internal` after
# this landed resolves to its real target schema, matching the precedent set
# by TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359 above.
#
# `live_events` is REMOVED from this set as of
# docker/migrations/forward/099_create_omninode_internal_live_events.sql --
# it is the first family individually transform-copied out of the bridge.
# Post-099, physical_grant_schema_for_table('omninode_internal', 'live_events')
# must return 'omninode_internal' (no override) so it agrees with the real
# INSERT-target schema handler_wiring._resolve_projection_database_target
# already resolves from the node contract's literal db_io.db_tables[0].schema.
# Leaving it enumerated here after the physical table exists would silently
# reintroduce the drift 099 exists to close: the grant-privilege check would
# keep asserting against `public` while every write already lands in
# `omninode_internal`.
INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359: frozenset[str] = frozenset(
    {
        "baselines_breakdown",
        "baselines_comparisons",
        "baselines_quality_snapshots",
        "baselines_roi_snapshots",
        "baselines_snapshots",
        "baselines_trend",
        "capsule_store",
        "contract_registry",
        "cost_by_repo_snapshots",
        "deployment_evidence_projection",
        "deployment_readiness_projection",
        "event_chain",
        "evidence_correlation_trace_projection",
        "evidence_dashboard_projection",
        "evidence_readiness_aggregate_projection",
        "gate_activity",
        "gate_metrics",
        "generation_events",
        "intent_classification_events",
        "llm_call_metrics",
        "llm_delegation_daily_projection",
        "llm_routing_decisions",
        "mcp_tools",
        "merge_state_transitions",
        "nightly_loop_configs",
        "nightly_loop_decisions",
        "nightly_loop_iterations",
        "node_service_registry",
        "overnight_session_phases",
        "overnight_sessions",
        "pr_lifecycle_ledger_entries",
        "pr_merged_events",
        "receipt_gate_rows",
        "renderer_capability_projection",
        "sandbox_decisions",
        "session_outcomes",
        "session_replay_snapshots",
        "swarm_runs",
        # OMN-16930: node_projection_tenant_registry's runtime-populated
        # slug->UUID mirror. Same bridge as node_service_registry and
        # validation_event_ledger above -- logically OMNINODE_INTERNAL-domain
        # per the node contract (a registry index, no tenant_id column, no
        # RLS), physically created bare in `public` by
        # docker/migrations/forward/nodes/node_projection_tenant_registry/
        # 0000_create_tenant_registry_mirror.sql. Physically-public is not a
        # convenience here, it is load-bearing: the migrate identity
        # (NODE_DB_USER=role_omnidash) holds neither USAGE nor CREATE on
        # `omninode_internal` (live-confirmed 2026-08-10 and recorded in
        # node_projection_live_events/0002's header, where the repair is still
        # a queued OPERATOR action), and node_projection_delegation/0032
        # resolves the mirror through an unqualified `to_regclass` at apply
        # time. Creating this relation in `omninode_internal` would make the
        # apply-time conversion unreachable on exactly the lanes it exists to
        # convert. Enumerated here so physical_grant_schema_for_table(
        # 'omninode_internal', 'tenant_registry_mirror') resolves to 'public'
        # and the application_database_sql_gate accepts the migration's bare
        # CREATE TABLE, matching its real physical location. Promotable once
        # OMN-15359 moves the family.
        "tenant_registry_mirror",
        "traces",
        "voice_sessions",
        # OMN-16385: validation_event_ledger (docker/migrations/forward/
        # 045_create_validation_event_ledger.sql, applied since 2026-02).
        # Same bridge as event_chain/gate_activity/receipt_gate_rows above --
        # logically OMNINODE_INTERNAL-domain durable audit/replay evidence (no
        # tenant_id column; run_id/repo_id/event_type identify a cross-repo
        # validation event, not a tenant), physically created bare in `public`
        # since the table predates the omninode_internal schema (098) by
        # months. Every read/write path
        # (src/omnibase_infra/runtime/db/postgres_validation_ledger_repository.py)
        # queries it unqualified against the connection's default search_path
        # -- actually moving the physical table would break every one of
        # those ~15 call sites, which this PR (a dead pgcrypto CREATE
        # EXTENSION statement removal) does not touch. Enumerated here so the
        # application_database_sql_gate accepts the migration's pre-existing
        # bare CREATE TABLE, matching its real physical location, the same
        # way it already does for node_service_registry and friends.
        "validation_event_ledger",
    }
)


def physical_grant_schema_for_table(schema: str, table_name: str) -> str:
    """Return the schema where PostgreSQL ACLs currently apply for a table.

    OMN-15359 owns the physical ``public`` -> ``tenant``/``omninode_internal``
    moves. Until each relation family is individually transform-copied and
    proven, these tables remain physically in ``public`` even though their
    contracts and runtime routing are logically tenant- or
    omninode_internal-domain.
    """
    if (
        schema == "tenant"
        and table_name in TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    ):
        return "public"
    if (
        schema == "omninode_internal"
        and table_name in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    ):
        return "public"
    return schema
