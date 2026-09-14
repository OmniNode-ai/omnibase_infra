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
nonce), locks/replays an existing identity when present, and returns one of
`CLAIMED`, `ALREADY_CONSUMED`, `EXPIRED`, `MISMATCH`, or `ERROR`.

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

The forward runner calculates the migration file's SHA-256 immediately before
execution, verifies that the bytes did not change before or during application,
and records that exact digest in `public.schema_migrations`. A later replay of
a 64-hex recorded digest must match the current bytes or fail closed. Historical
non-digest runner markers preserve their prior skip semantics and are not
silently reclassified as content evidence.

PostgreSQL 16 concurrency/restart/ACL tests are included but remain unverified
in this offline sandbox; no local database or Unix-domain-socket integration
test was attempted here.
