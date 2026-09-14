<!-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Action-authorization nonce-claim boundary (OMN-17486)

Migration 107 owns a dedicated PostgreSQL boundary for one-time, canonical
pre-execution action-authorization claims. It is separate from all existing
first-effect and ledger records. A successful claim is evidence only: it never
sets `execute_enabled` and never authorizes an action by itself.

The only runtime operation is the atomic
`action_authorization_claim.claim_action_authorization` function. It receives
the exact canonical fields, persists only the nonce digest (never the raw
nonce), and returns one of `CLAIMED`, `ALREADY_CONSUMED`, `EXPIRED`,
`MISMATCH`, or `ERROR`.

## How one claim wins

The function validates, then runs exactly one write: an `INSERT ... ON CONFLICT
DO NOTHING ... RETURNING`. The winner is decided by the three unique indexes on
`authorization_id`, `nonce_digest` and `request_digest`, not by a lock the
function takes or a snapshot it reads. One returned row means this caller wrote
the claim, and its state is its outcome. Zero returned rows means a conflicting
claim exists and is already committed, because `ON CONFLICT DO NOTHING` waits
out an in-flight inserter before it yields; the function then reads that row
once and classifies it as `MISMATCH`, `EXPIRED` or `ALREADY_CONSUMED`.

There is no retry loop. An earlier revision of this function looped: the
conflict branch answered `ALREADY_CONSUMED`, fell through to the insert, caught
its own `unique_violation` and went round again, so every repeat claim of a
request spun forever with no wait event and no lock to attribute it to. That is
the whole reason the shape above is one statement — a loop inside a
`SECURITY DEFINER` function that every lane's migration runner applies is a
liveness hazard for every caller, not only for concurrent ones.

If the conflicting row cannot be read after a conflict, the function raises
`40001` rather than guessing. That is reachable only above `READ COMMITTED`,
where the caller's snapshot predates the winner's commit, and a retry in a new
transaction resolves it.

## Rights posture

`claim_action_authorization` is `SECURITY DEFINER` and owned by the role that
applies the migration; the restricted principal never holds table rights. The
migration revokes everything on the schema, the table and both functions from
`PUBLIC` and from `rsd_action_authorization_claim`, then grants that role
`USAGE` on the schema and `EXECUTE` on the claim function alone.
`rsd_action_authorization_claim` is created `NOLOGIN NOSUPERUSER NOCREATEDB
NOCREATEROLE NOINHERIT NOBYPASSRLS NOREPLICATION`, and the migration refuses to
proceed if a role of that name already exists with any of those attributes. A
`BEFORE UPDATE OR DELETE` trigger makes every claim row immutable once written.

## OMN-17462 integration seam

The stable client API is
`claim_action_authorization_via_unix_socket(socket_path=..., request=...)`.
The equivalent local CLI is:

```text
onex-action-authorization-claim --socket /absolute/path/resolved-by-approved-overlay
```

It accepts one canonical request JSON document on standard input and emits the
typed, redacted claim result on standard output. The socket path, PostgreSQL
connection reference, restricted principal, socket-owner UID, and authorized
peer Unix UID must be resolved by an approved overlay; this package reads no
environment or fallback configuration. The socket-owner UID protects the
non-symlink parent directory and final socket inode. The authorized peer UID is
separate and is checked through Linux `SO_PEERCRED` before any request bytes are
read.

For the combined stacked OMN-17462 + OMN-17486 branch, the consumer recipe is:

1. Resolve the OMN-17486 typed overlay and construct the restricted claim port.
2. Submit the canonical action request to the client/CLI immediately before the
   protected action boundary.
3. Continue only when the returned outcome is `CLAIMED`; hard-deny every other
   outcome, including `ERROR`.
4. Keep the existing OMN-17462 hard-deny in force until this composition and its
   combined-branch integration test land together.

No OMN-17462 consumer is wired in this candidate. That dependency is explicit:
the two candidate commits must be stacked before any consumer claim-before-action
proof can be asserted.

## Migration evidence

The existing forward runner records this migration in
`public.schema_migrations` with the file's SHA-256 as its checksum. This
migration relies on that runner as it already stands and changes none of it.
The separate runner checksum hardening that an earlier draft of this work
carried is deliberately not in this change: it alters how every migration on
every lane is recorded and belongs to its own ticket.

The PostgreSQL 16 concurrency, restart and ACL proofs under
`tests/integration/runtime/action_authorization_claim/` run against a throwaway
cluster built by the shared `ephemeral_postgres` fixture, and they were run.
Every pool in those files sets `statement_timeout`, so a liveness defect in the
claim function fails the run instead of hanging it. An earlier revision of this
document recorded these proofs as unverifiable in an offline sandbox; that was
wrong, and it hid the livelock described above for thirteen days.
