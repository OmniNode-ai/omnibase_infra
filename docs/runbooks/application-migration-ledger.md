# Application migration ledger

**Ticket:** OMN-15413
**Target database:** `omnidash_analytics`
**Canonical relation:** `platform_catalog.schema_migrations`

This runbook defines the deterministic migration-history boundary for the
unified application database. It does not merge the separate
`omnibase_infra.public.schema_migrations` service ledger into the application
database.

## Canonical model

The selected application ledger is the existing checksum-capable
`public.node_schema_migrations` table. The bootstrap moves that table into
`platform_catalog` in one PostgreSQL transaction and keeps its relation OID,
rows, timestamps, and owner. It then adds the contract dimensions required by
the deployment topology:

| Column | Meaning |
| --- | --- |
| `migration_stream` | Exact producer stream, such as `node:<node>` |
| `owner` | Checked-in producer owner; never inferred from the login role |
| `domain` | `tenant` or `omninode_internal` for executable node migrations |
| `version` | Exact historical runner identity |
| `checksum` | Lowercase SHA-256 |
| `checksum_kind` | `content_sha256` or quarantined `legacy_attestation` |
| `applied_at` | Original application timestamp |
| `provenance` | Deterministic source relation/file identity |

The primary key is `(migration_stream, domain, version)`. An active node row
must match the checked-in file SHA-256 exactly. A `legacy_attestation` row can
never satisfy an active migration probe.

## Historical import policy

Source ledgers remain intact. Import is additive to the selected canonical
ledger and rerunning it must produce an identical row set.

| Source | Authority | Canonical treatment |
| --- | --- | --- |
| `public.node_schema_migrations(version, applied_at, checksum)` | Applied node set | Move in place; require exact manifest identity and file SHA-256 |
| `public.omnimarket_schema_migrations` | Applied projection set | Normalize `(node_name, filename)` to the exact node version; require manifest checksum; reject overlap with node history |
| Filename-only `public.schema_migrations(filename, applied_at)` | Applied legacy set without checksums | Preserve source and import a deterministic source-record attestation under `legacy:filename-only` |
| Cloud `public.schema_migrations(version, applied_at, checksum)` | Applied cloud set | Preserve source and import under exact producer stream `omninode-cloud` |
| Cloud `public.migrations_log` | Audit/attempt evidence only | Enrich matching applied rows through the checked-in alias map; a log-only alias is fatal |

Filename-only and cloud records currently use domain
`legacy_unclassified`. This is an explicit non-executable quarantine, not a
claim that the migrations target `tenant`. OMN-15423 must provide
per-artifact topology domains before those rows can be promoted. Cloud SQL is
cross-domain, so a blanket domain default is prohibited.

## Deterministic procedure

1. Validate every vendored node SQL file against
   `_ledger/application-migrations.tsv` or an explicit ticketed block in
   `_ledger/application-migration-blocks.tsv`.
2. Stop before migration DDL if an unresolved block is not protected by the
   existing operator fence.
3. Select and extend the checksum-capable node ledger in one transaction.
4. Import filename-only and omnimarket source rows without updating or deleting
   either source relation.
5. Read cloud applied state and audit aliases under one repeatable-read source
   snapshot; import it transactionally into the application ledger.
6. Probe active migrations by stream, domain, version, owner, checksum kind,
   and checksum. Apply and record only an absent, fully declared file.
7. Run the same process a second time and require an identical ledger signature
   with zero newly applied files.

Never insert, update, or delete migration-ledger rows by hand. A missing row is
an indeterminate apply: stop the lane, retain the source catalogs/logs, and
repair through a reviewed deterministic import or a new migration.

## Validation

```bash
uv run python scripts/validation/validate_application_migration_manifest.py
uv run python scripts/validation/validate_application_migration_manifest.py --require-complete
uv run pytest tests/unit/scripts/validation/test_application_migration_manifest.py \
  tests/integration/migrations/test_application_migration_ledger_omn15413.py -q
docker compose -f docker/legacy-rds-fixture/compose.yml up \
  --build --abort-on-container-exit --exit-code-from proof
```

The first command validates the checked-in surface and reports explicit
blocks. The completion command remains RED while any block exists. The
PostgreSQL 16 integration suite proves fresh and sanitized legacy execution
twice plus checksum, unknown-stream, and double-declaration RED controls. The
Docker fixture must also use `--build` so it contains the changed runner and
ledger artifacts.

## Landing holds

- OMN-15423 still leaves `delegation_judge_verdict_events` unclassified. Its
  unfenced `0016` migration makes the complete runner preflight RED.
- The Kubernetes runner still targets `public.node_schema_migrations`. This
  change cannot land until its exact stacked update targets
  `platform_catalog.schema_migrations`; otherwise it would recreate a second
  ledger after the in-place move.
- No live database query, deployment, grant/RLS change, cutover, or destructive
  action is part of this ticket.
