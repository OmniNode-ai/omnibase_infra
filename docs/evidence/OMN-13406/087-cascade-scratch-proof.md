# OMN-13406 — 087 CASCADE scratch acceptance proof

Throwaway Postgres 16 (`docker run --rm postgres:16` on .201), torn down after the run.
Proves: (a) WARM DB (decoy + dependent view) → hardened 087 succeeds, drops both;
(b) COLD DB (no decoy) → clean no-op; (neg) OLD bare DROP (no CASCADE) → Postgres refuses
(reproduces the original blocker that the CASCADE fixes).

```
[harness] starting throwaway postgres:16 container (--rm) ...
[harness] postgres ready
[harness] executing proof inside container ...
====================================================================
OMN-13406 087 CASCADE scratch proof  (2026-06-21T10:58:30Z)
postgres server version: 16.14 (Debian 16.14-1.pgdg13+1)
hardened 087 DROP line under test:
45:DROP TABLE IF EXISTS public.delegation_events CASCADE;
====================================================================

### Scenario (a): WARM DB (decoy table + dependent view) -> hardened 087 (CASCADE)
NOTICE:  database "scratch_a" does not exist, skipping
  pre:  table_exists=t  view_exists=t  decoy_rows=40
  --- running hardened 087.sql against scratch_a ---
DROP TABLE
  087 exit: 0 (SUCCESS)
psql:/work/087.sql:45: NOTICE:  drop cascades to view projection_delegation_summary
  post: table_dropped=t  view_dropped=t
  VERDICT (a): PASS — decoy table AND dependent view both dropped.

### Scenario (b): COLD DB (no decoy) -> hardened 087 (clean no-op)
NOTICE:  database "scratch_b" does not exist, skipping
  pre:  delegation_events_absent=t
  --- running hardened 087.sql against scratch_b ---
DROP TABLE
  087 exit: 0 (SUCCESS / no-op)
psql:/work/087.sql:45: NOTICE:  table "delegation_events" does not exist, skipping
  post: delegation_events_absent=t
  VERDICT (b): PASS — clean no-op on cold DB.

### Negative control: WARM DB + OLD bare DROP (no CASCADE) -> expected FAIL
NOTICE:  database "scratch_neg" does not exist, skipping
  --- running OLD bare 'DROP TABLE IF EXISTS public.delegation_events;' (ON_ERROR_STOP) ---
ERROR:  cannot drop table delegation_events because other objects depend on it
DETAIL:  view projection_delegation_summary depends on table delegation_events
HINT:  Use DROP ... CASCADE to drop the dependent objects too.
  bare DROP exit: NONZERO (EXPECTED — Postgres refused; confirms the original blocker)
  VERDICT (neg): PASS — bare DROP reproduces the warm-DB failure the CASCADE fixes.

====================================================================
OVERALL: PASS — (a) CASCADE drops decoy+view, (b) cold no-op, (neg) bare DROP fails.
[harness] proof exit code: 0
[harness] tearing down throwaway container ...
[harness] container removed.
```
