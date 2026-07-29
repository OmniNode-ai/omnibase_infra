# Sanitized legacy-RDS fixture (OMN-15422)

This directory is a wholly synthetic PostgreSQL 16 reproduction surface for the
one-application-database program. It contains no dump-derived value, customer data,
credential, secret, or live database observation. Its provenance is limited to the
committed/static evidence named in `fixture-manifest.json`.

Run the proof with rebuilt images:

```bash
docker compose -f docker/legacy-rds-fixture/compose.yml \
  up --build --abort-on-container-exit --exit-code-from proof
docker compose -f docker/legacy-rds-fixture/compose.yml \
  down --volumes --remove-orphans
```

The harness runs each detector against a safe control and a seeded RED control. It
then applies the real vendored shape-reconciliation migrations twice, runs the real
forward runner twice on a clean install, and executes the real legacy upgrade twice.

The legacy runner is expected to stop at the `filename` versus `migration_id` ledger
boundary owned by OMN-15413. The harness succeeds only when that exact blocker is
observed; any other failure or an unreviewed surprise success fails the proof. This
ticket does not implement ledger import, grants, RLS rollout, topology rendering,
data transformation, cutover, or destructive cleanup.
