# Managed-staging canary migrations (`docker/migrations/canary/`)

A **separate, manually-applied** migration set for the one-tenant managed-staging
delegation canary. It is **not** the flat `docker/migrations/forward/` sequence and
is **never** auto-applied by PostgreSQL's `docker-entrypoint-initdb.d` mechanism.

Ticket: **OMN-14737** (Lane B / task **B12**).

## Why this is a separate set

The managed RDS instance `omninode-dev-postgres` has `DBName=null`, so logical
databases are **hand-created**. The canary lands its correlation-linked readback in
a **dedicated canary logical DB** (default `omninode_canary_mstg1`, epoch-scoped to
the B7 `mstg1` epoch) — not in `omnibase_infra` and not in `omnidash_analytics`.
Putting the DDL in the flat `forward/` set would (a) auto-apply the canary-only
table into the `omnibase_infra` database on a fresh container init and (b) change
the flat schema fingerprint. Neither is correct. This set mirrors the existing
per-database precedent (`docker/migrations/intelligence/` for `omniintelligence`).

The flat-set guards are **not** affected by files under this directory:

- `scripts/check_schema_fingerprint.py` globs `docker/migrations/forward/*.sql`
  (flat, non-recursive) — files here are excluded, so the committed fingerprint
  stays in sync.
- `scripts/validation/validate_migration_sequence.py` scans
  `docker/migrations/forward` and `src/omnibase_infra/migrations/forward` — this
  directory is a distinct identity space, so the `001` prefix here does **not**
  collide with the flat `001_registration_projection.sql`.

## Layout

```
docker/migrations/canary/
  README.md                                                   # this file
  forward/
    001_create_delivery_replay_canary_projection.sql          # landing table
  rollback/
    rollback_001_delivery_replay_canary_projection.sql        # drop table
```

## What the landing table is

`delivery_replay_canary_projection` — the durable surface the canary (B11) writes
its correlation-linked readback to. Its columns are derived field-by-field from the
deterministic output model `ModelReplayProjection` of the B6 delivery/replay
projection compute node (OMN-14726, omnimarket
`node_delivery_replay_projection_compute`). See the column-level citations in the
forward migration.

## How it is applied (HELD FOR OPERATOR)

This set is applied **manually**, one operator-gated step at a time, per the
provisioning runbook. Nothing here executes autonomously and committing it
authorizes no live DB/cred mutation.

- Provisioning (bring-up): [`docs/runbooks/managed-staging-canary-postgres-provisioning.md`](../../../docs/runbooks/managed-staging-canary-postgres-provisioning.md)
- Teardown / abort / rollback: [`docs/runbooks/managed-staging-canary-teardown-rollback.md`](../../../docs/runbooks/managed-staging-canary-teardown-rollback.md) (B13, step T-5 covers this DB)

## Re-running with a fresh epoch

To mint a provably disjoint surface for a canary re-run, bump the epoch (`mstg2`,
`mstg3`, …) to match the B7 topic/group epoch: create `omninode_canary_mstg2`,
apply `forward/001_*.sql` into it, and scope a `role_canary_mstg2` credential. The
DDL is DB-agnostic — only the target DB name and role change.
