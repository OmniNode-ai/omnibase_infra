# Generated application ACL proof

This rebuilt PostgreSQL 16 harness proves the generated one-database ACL and
default-privilege matrix without touching a deployed database. The seed is wholly
synthetic: it contains no dump-derived value, customer identifier, credential, secret,
or live catalog observation.

Run it with rebuilt images:

```bash
docker compose -f docker/application-acl-proof/compose.yml \
  up --build --abort-on-container-exit --exit-code-from proof
docker compose -f docker/application-acl-proof/compose.yml \
  down --volumes --remove-orphans
```

The same catalog comparator must detect seeded database/schema/object ownership,
`PUBLIC`, broad-object, per-column, grant-option, unsafe-default, PG16 membership,
undeclared-principal, and cross-domain defects before the matrix is
applied. A transient RED control also proves that an undeclared runtime-owned object
cannot hide outside the typed inventory. The harness covers tables, views,
materialized views, sequences, functions, procedures,
enum/domain/composite/range/multirange types, and the `public` schema. It then applies
the generated SQL twice, proves exact catalog
and behavioral access, restores the generated pre-change fixture snapshot including
grantors and membership options, proves a byte-for-byte semantic snapshot match,
reapplies twice, and proves the final state. Routine targets include their PostgreSQL
identity arguments so overloads cannot collapse or receive a name-wide grant.
The generated script has typed `scaffold` and `full` phases. A scaffold can become
`READY` while the full phase remains honestly `BLOCKED` on objects that have not yet
been materialized. Conversely, FULL can be `READY` while SCAFFOLD is `BLOCKED` when
the exact catalog census proves existing roles need attribute hardening; FULL checks
that observed pre-state and repairs it in its transaction, while the additive phase
refuses to conceal it. The scaffold creates absent-but-catalog-proven workload/owner
roles and the three empty target schemas, adds only grants for roles already proven
safe (or newly created safe), and establishes deny-by-default future-object ACLs
only for owner roles it creates.
It preserves existing role attributes, memberships, database ownership, legacy
`CONNECT`, and legacy `public` access; exact revocation and ownership hardening are
FULL-only after the activity/catalog gates pass. A disposable fresh-database lane
proves its wrong-database guard, atomic failure, idempotent apply, legacy
CONNECT/read/write preservation, zero object mutation, hostile collation/type
collision rejection, the later FULL denial, and exact non-cascading removal before
copy. Every script asserts
`current_database()` before its first mutation.

Both phases run through the real `psql` path inside one explicit transaction; an
injected mid-apply error must leave the entire pre-change snapshot unchanged. The
rollback path is independently failure-injected and transactional. It restores
grantor chains topologically for database/schema/object/column ACLs and PG16 role
memberships, then removes only still-empty schemas and dependency-free roles—never
with `CASCADE`. Raw `pg_default_acl` row presence is part of the snapshot: rollback
normalizes PostgreSQL's implicit function/type `PUBLIC` defaults before replaying
captured deviations, and behavioral checks create future objects to prove those
built-ins return. An unrelated-schema default-ACL sentinel remains byte-for-byte and
behaviorally unchanged through scaffold, FULL, and rollback. The same run revokes
all database privileges before re-granting exact
`CONNECT` access and proves positive/negative connection isolation across the eight
approved non-system databases: `omnidash_analytics`, `omninode_cloud`, `keycloak`,
`omnibase_infra`, `omniintelligence`, `omniclaude`, `omnimemory`, and `umami`.

`generated/prechange-fixture-acl.json` is explicitly a sanitized fixture rollback
artifact, not a live RDS ACL capture or a full-day `(datname, usename)` sample. The
locked real-source candidate remains `BLOCKED` and emits no production SQL until an
authorized exact principal census (including explicit cluster-global presence and
absence evidence), immutable source-locked catalog/activity query and result blobs,
a complete 24-hour-or-longer activity window, an
independent typed principal/domain policy, exact function signatures and object-kind
counts, materialized target-location evidence (source and target may coexist), and
the named catalog/activity/ownership blockers are resolved.
