# OMN-15359 — `omninode_internal` schema, physical build (P2 slice)

Ticket: OMN-15359 (P2-P4 build classified schemas and migrate internal,
control-plane, catalog, and tenant targets). Parent: OMN-15354. Consumer this
PR unblocks: OMN-15426 (P5 cut internal projections to the `omninode_runtime`
identity).

## What this PR built (additive only)

1. **Physical schema.** `docker/migrations/forward/098_create_omninode_internal_schema.sql`
   — `CREATE SCHEMA IF NOT EXISTS omninode_internal` inside `omnidash_analytics`
   (the physical database backing the unified `application` topology
   database). No table created, moved, or altered. Proven live against a real
   ephemeral Postgres cluster:
   `tests/integration/migrations/test_098_omninode_internal_schema_omn15359.py`
   (schema created, zero tables land, idempotent re-apply, rollback drops the
   empty schema, rollback fails closed — `RESTRICT` — once any table has
   landed in it).
2. **Physical-schema bridge, extended to OMNINODE_INTERNAL.**
   `src/omnibase_infra/topology/physical_schema_mapping.py` gains
   `INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359`, mirroring the
   already-shipped `TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359`
   pattern. `physical_grant_schema_for_table` now resolves any of the 41
   listed tables to `public` (their real physical location) instead of the
   schema the contract logically declares, so grant derivation and
   `handler_wiring` stop targeting a relation that does not exist there. This
   is the direct fix for the OMN-15426 live gap: *"handler_wiring.py issues
   schema-qualified SQL against `omninode_internal` for the 41-table
   `omninode_runtime` domain; physically all 41 still live in public...
   Grant on public schema WITHHELD."* (rolling ledger, 2026-08-03T19:2xZ).
3. **Companion topology grant.** `omninode_runtime` did not hold `USAGE ON
   SCHEMA public` in any shipped instance (only `USAGE ON SCHEMA
   omninode_internal`) — a real, separate gap the bridge alone does not close,
   since a table-level GRANT is inert without schema-level USAGE. Added to all
   three source-of-truth instances
   (`src/omnibase_infra/topology/instances/{local,onex-dev,onex-prod}.yaml`),
   mirroring the grant `tenant_projection_writer` already holds. The 41 TABLE
   grants for `omninode_runtime` were regenerated with
   `scripts/generate_application_database_table_grants.py --write` (against
   the live sibling `omnimarket` checkout — 57 `db_io.db_tables` declarations,
   identical to what was already checked in) so they now read `schema:
   public`, matching the bridge and the tenant precedent's shape. The 7
   `docker/catalog/database-topology/*.yaml` projections were regenerated with
   `scripts/render_application_database_topology.py`.

## Relation family covered

**Family:** `omninode_runtime` internal-domain projections (41 tables:
`baselines_*` (6), registry/catalog (`node_service_registry`,
`contract_registry`, `capsule_store`, `mcp_tools`), evidence/deployment
(`deployment_evidence_projection`, `deployment_readiness_projection`,
`evidence_correlation_trace_projection`, `evidence_dashboard_projection`,
`evidence_readiness_aggregate_projection`), telemetry
(`llm_call_metrics`, `llm_delegation_daily_projection`,
`llm_routing_decisions`, `cost_by_repo_snapshots`, `gate_activity`,
`gate_metrics`, `generation_events`, `intent_classification_events`,
`live_events`, `traces`, `merge_state_transitions`, `pr_lifecycle_ledger_entries`,
`pr_merged_events`, `receipt_gate_rows`, `renderer_capability_projection`,
`sandbox_decisions`, `event_chain`), orchestration/session
(`nightly_loop_configs`, `nightly_loop_decisions`, `nightly_loop_iterations`,
`overnight_session_phases`, `overnight_sessions`, `session_outcomes`,
`session_replay_snapshots`, `swarm_runs`, `voice_sessions`,
`skill_execution_snapshots` is TENANT, excluded)).

- **Owner:** `omnibase_infra` (schema owner `owner_omninode_internal` per
  topology; not yet physically created — see Deferred).
- **Producer:** each table's own node migration under
  `docker/migrations/forward/nodes/<node>/` (vendored from `omnimarket`).
- **Consumer:** `omninode_runtime` principal, binding
  `omninode_runtime_service` (`OMNINODE_INTERNAL_DB_URL`).
- **Domain:** `OMNINODE_INTERNAL` (`docker/catalog/database-topology/*.yaml`
  `schemas.omninode_internal.domain`).
- **Migration stream:** `omnibase_infra.application`.
- **Current physical location:** `public`, bridged (this PR does not move
  data).
- **Target:** `omnidash_analytics.omninode_internal`, physically created and
  empty as of this PR.

`delegation_workflow_state` (R-q, OMN-15337) and the OMN-15423 residual
dispositions (`event_bus_events`, legacy `schema_migrations`) are **not**
part of this family and are untouched here — those are separate, still-open
classification lanes on OMN-15337/OMN-15423.

## Deferred (explicitly, not silently)

Per the ticket's own scope text — *"Preserve source relations until
family-level parity and migration proof complete. Do not use `ALTER TABLE
... SET SCHEMA` on sources and do not use blind indefinite dual-write"* — the
following are out of this PR and owned by the P5 cutover tickets
(OMN-15426/OMN-15360) via the OMN-15420 cutover-journal machinery
(`omnibase_infra.migration.cutover`):

- **Physical per-table copy** of the 41 relations from `public` into
  `omninode_internal`, each with its own transformation receipt (counts, key
  sets, hashes, FKs, sequences, grants, owners, policies reconciled) — the
  full acceptance-criteria proof this ticket's parent scope describes.
- **`owner_omninode_internal` / `omninode_runtime` role creation.** Neither
  role is physically created anywhere in the migration corpus yet (verified:
  zero `CREATE ROLE omninode_runtime` / `CREATE ROLE owner_omninode_internal`
  hits across `docker/migrations/`). Creating them safely (RDS-compatible,
  guarded against re-ALTER privilege demands) is its own change, matching the
  care `094_create_app_dashboard_role.sql` took for `app_dashboard` — not
  something to fold into a schema-creation PR.
- **Tenant-side physical move.** `TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359`
  is untouched; the tenant `CREATE SCHEMA` target already exists via prior
  work (OMN-14894/OMN-15655) and its own migration proof.
- **Grant regeneration re-run under the CI-pinned `omnimarket` checkout.**
  This PR's `--write` ran against the live local sibling clone, not the pin
  CI resolves; `scripts/generate_application_database_table_grants.py
  --check --prove` in the OMN-15361 CI job re-verifies against the pinned
  checkout and is the actual gate of record.
