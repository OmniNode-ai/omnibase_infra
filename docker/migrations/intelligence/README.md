# `docker/migrations/intelligence/` — the applied migration set for the `omniintelligence` database

**Ticket:** OMN-15276 (DDL-ownership decision, scope item 2).

## This directory is what the `.201` docker lanes actually apply

The `intelligence-migration` one-shot service mounts **this** directory and applies
every `*.sql` in it, in sorted order, recording basenames in
`omniintelligence.schema_migrations`:

| Surface | Binding |
| --- | --- |
| `docker/docker-compose.infra.yml` (dev / stability-test / prod lanes) | `MIGRATIONS_DIR: /migrations/intelligence`, `../docker/migrations/intelligence:/migrations/intelligence:ro` |
| `docker/docker-compose.judge.yml` (judge lane) | same pair |
| `docker/catalog/services/intelligence-migration.yaml` | same pair |
| `scripts/run-intelligence-migrations.sh` | `MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations/intelligence}"`, `for migration_file in $(ls "${MIGRATIONS_DIR}"/*.sql \| sort)` |

`intelligence-api` depends on this service with `service_completed_successfully`, so a
migration failure here blocks the lane rather than silently degrading it.

## There is a second, drifted tree — do not confuse them

`omniintelligence/deployment/database/migrations/` is a **separate** tree that feeds the
`omniintelligence-migrate` ECR image (`omniintelligence/deployment/docker/Dockerfile.migrate`,
built by `build-and-push-migrate-image.yml` on pushes to `main`) for the cloud k8s
migration Job. No `.201` lane reads it.

The two trees have **already drifted and no gate compares them** (measured 2026-07-27
against `omniintelligence@1af0132e`): of the 26 `.sql` files here, 8 differ byte-for-byte
from the same-named file over there, 3 exist only here, and 5 exist only there.

**This is the trap OMN-15276 exists to name:** the two conflicting `code_entities`
migrations (`025_code_entities.sql` / `025_create_code_entities.sql`) lived in the
omniintelligence tree, so neither was ever applied on any lane —
`omniintelligence.schema_migrations` held 27 identical rows on stability-test and prod,
ending at `025_fix_llm_delegation_call_log_date_index`, and `code_entities` was absent
from 20/20 databases across both lanes. A fix landed in the omniintelligence tree would
have passed CI and changed nothing on any lane.

## Decision

**`docker/migrations/intelligence/` owns the `code_entities` / `code_relationships` DDL.**

- The canonical DDL is `026_create_code_entities.sql` (OMN-15276). It is the only file
  repo-wide that creates either table; the omniintelligence copies were retired in the
  companion PR.
- The number `026` is a **fresh slot in this directory's own namespace** (this set topped
  out at `025_fix_llm_delegation_call_log_date_index`). Nothing already recorded in
  `schema_migrations` was renumbered — the runner keys on basename, so renumbering an
  applied migration would re-apply it under a new id.

### Known gaps this decision does not close

1. **Cloud parity.** The `omniintelligence-migrate` ECR image no longer carries any
   `code_entities` DDL. Because the table has never existed on any probed lane, this
   removes an unapplied file rather than regressing a live schema — but cloud lanes will
   need this DDL (or a real sync path between the two trees) before the code-intelligence
   nodes run there. Follow-up, not slice 1.
2. **No duplicate-prefix gate covers this directory.**
   `scripts/validation/validate_migration_sequence.py` scans only
   `docker/migrations/forward/` and `src/omnibase_infra/migrations/forward/`. That is why
   the four-month-old triple-claimed `025` prefix never fired a gate — the check exists,
   but not over this path. `023` is duplicated *here* today for the same reason
   (`023_create_debug_intelligence_tables.sql` / `023_create_dispatch_eval_results.sql`),
   and both are already applied, so it cannot be fixed by renumbering.
   `tests/unit/migrations/test_code_entities_canonical_ddl.py` installs a **ratchet**:
   `023` is the sole grandfathered duplicate and any *new* duplicate prefix in this
   directory fails. Promoting that ratchet into the pre-commit/CI sequence validator is
   the enforcement follow-up.
3. **`schema_fingerprint.sha256` in this directory is stale and unverified.** It records
   `migration_file_count: 25` at `2026-04-30`; the directory held 26 files before this
   change. `scripts/check_schema_fingerprint.py` defaults to `docker/migrations/forward`
   and has no caller pointed here, so the artifact is decorative. Left untouched
   deliberately — stamping it would manufacture a green signal for a check nothing runs.

## Adding a migration here

1. Take the next free `NNN` **in this directory** (`ls docker/migrations/intelligence/`).
2. Write idempotent SQL (`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`) — warm
   lanes re-run the runner on every bring-up.
3. Never renumber an applied file. `schema_migrations` keys on the basename; a rename
   re-applies the SQL under a new id.
4. Retiring a superseded migration = **delete the file** (the OMN-13124 idiom). The runner
   discovers files on disk, so a removed migration is simply never applied on fresh
   volumes and leaves no dangling `schema_migrations` reference. The
   `docker/migrations/skip-manifest.yaml` tombstone mechanism belongs to
   `run-forward-migrations.sh` and is not read by `run-intelligence-migrations.sh`.
