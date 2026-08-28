<!--
SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
SPDX-License-Identifier: MIT
-->

# Headless secret seeding (`onex skill seed_secrets`)

**Ticket:** OMN-16897
**Node:** `node_secret_seed_effect` (`src/omnibase_infra/nodes/node_secret_seed_effect/`)

Put a secret into Infisical from the command line. No web UI, no interactive
login, no browser. One command, a typed receipt on stdout.

---

## The one thing to understand before using this

**The command never takes a secret value as an argument.**

`onex skill` turns CLI arguments into the backing node's input payload, and a
node payload is serialised onto the event bus and into the event log. A
`--value` flag would therefore write your key into durable storage on its way
to Infisical. There is no such flag and the request model rejects one.

Values travel by **file**: `--source-path` names a local dotenv-style file (or
`-` for stdin). The handler reads it at execution time and the values never
leave that call — not into the receipt, not into a log line, not into an error
message. Verification after a write is a **name listing**; this node has no
code path that can read a stored secret value.

---

## The immediate use case: landing a new GLM key

`LLM_GLM_API_KEY` is already declared in `config/shared_key_registry.yaml`
under `/shared/llm/`, so that is where it goes.

### 1. Put the key in a file, not on the command line

```bash
umask 077
cat > /tmp/glm-seed.env <<'EOF'
LLM_GLM_API_KEY=<the new key>
EOF
```

Writing it with a heredoc rather than `echo` keeps it out of your shell
history. Delete the file when you are done (step 5).

Prefer no file at all? Pipe it and use the stdin sentinel:

```bash
printf 'LLM_GLM_API_KEY=%s\n' "$NEW_KEY" | \
  uv run onex skill seed_secrets --source-path - ...
```

### 2. Export the machine identity

```bash
export INFISICAL_CLIENT_ID='<machine identity client id>'
export INFISICAL_CLIENT_SECRET='<machine identity client secret>'
```

Both are required. If either is missing the run stops with
`AUTH_UNAVAILABLE`, names the missing **variable**, and writes nothing. There
is no fallback identity — see "the one-time operator step" below if you do not
have these yet.

### 3. Dry run first (this is the default)

```bash
uv run onex skill seed_secrets \
  --source-path /tmp/glm-seed.env \
  --infisical-host http://192.168.86.201:8881 \
  --project-id "$ONEX_PLATFORM_PROJECT_ID" \
  --environment-slug prod \
  --secret-path /shared/llm/ \
  --keys LLM_GLM_API_KEY
```

Omitting `--execute` means dry run. The receipt reports which names *would* be
created and which *would* be updated, resolved from a name listing, and issues
zero writes.

```json
{
  "verdict": "dry_run",
  "success": true,
  "created_names": ["LLM_GLM_API_KEY"],
  "updated_names": [],
  "dry_run": true
}
```

### 4. Apply

Add `--execute`:

```bash
uv run onex skill seed_secrets \
  --source-path /tmp/glm-seed.env \
  --infisical-host http://192.168.86.201:8881 \
  --project-id "$ONEX_PLATFORM_PROJECT_ID" \
  --environment-slug prod \
  --secret-path /shared/llm/ \
  --keys LLM_GLM_API_KEY \
  --execute
```

`verdict: "seeded"` with the name in `verified_names` means the write landed
and was confirmed present by name readback. Anything else is a failure — see
the verdict table.

### 5. Clean up

```bash
shred -u /tmp/glm-seed.env 2>/dev/null || rm -f /tmp/glm-seed.env
unset INFISICAL_CLIENT_SECRET
```

### 6. Restart whatever reads it

Seeding writes to Infisical; it does not restart anything. The runtime
prefetches config at boot, so a running lane keeps the old value until it is
restarted through its normal deploy path.

---

## Flags

| Flag | Required | Meaning |
|---|---|---|
| `--source-path` | yes | Local dotenv-style file, or `-` for stdin. Names where the values are; never carries them. |
| `--infisical-host` | yes | Absolute `http(s)` URL of the target instance. |
| `--project-id` | yes | Project **UUID**, not the project name. |
| `--environment-slug` | yes | e.g. `dev`, `prod`. |
| `--secret-path` | yes | Target folder, e.g. `/shared/llm/`. |
| `--keys` | no | Comma-separated allowlist. Empty seeds every name in the source. |
| `--execute` | no | Opt in to writing. Absent = dry run. |

Every addressing flag is required with no default on purpose. This estate runs
three separate Infisical instances (see below) and a defaulted host, project,
environment or path would seed a real key somewhere nobody meant to touch.
There is deliberately **no `--dry-run` flag**: `onex skill` boolean args are
presence flags that can only ever be set to true, so a `--dry-run` defaulting
to true could never be turned off. Writing is the opt-in instead.

### Which instance?

| Instance | Host | Serves |
|---|---|---|
| `.201` dev lane | `http://192.168.86.201:8881` | `.201` dev runtime lane, local dev bootstrap |
| `.201` stability lane | `http://192.168.86.201:8880` | `.201` stability-test lane only (own DB) |
| in-cluster k8s | `http://infisical.dev.svc.cluster.local:8080` | onex-dev and onex-prod namespaces |

Project names live in `config/infisical_projects.yaml`; the shared platform
project is `onex-platform`. You need its **UUID**, which the projects file does
not carry — retrieve it from the instance (`infisical projects list`, or the
project settings page) and keep it in your shell profile.

---

## Verdicts

| Verdict | `success` | Means |
|---|---|---|
| `seeded` | ✅ | Every name written and confirmed present by name readback. |
| `dry_run` | ✅ | Plan computed, zero writes issued. |
| `auth_unavailable` | ❌ | Machine identity missing or rejected. Nothing written, no fallback identity attempted. |
| `source_unreadable` | ❌ | Source file missing or malformed. Reports a **line number**, never line content. |
| `no_keys` | ❌ | Nothing to seed — empty source, or `--keys` naming keys the source lacks. |
| `store_unreachable` | ❌ | Instance unreachable or its name listing failed. Identity is probably fine; check the host. |
| `write_failed` | ❌ | At least one write was rejected. Every name is still reported; successful writes are not rolled back. |
| `verify_failed` | ❌ | A write was accepted but the name did not appear on readback. |

Two of these are worth calling out.

**`no_keys` is a failure.** A seed run that seeded nothing must not read green,
because "green" is how a typo in `--keys` would otherwise present.

**`write_failed` does not abort at the first error.** Every name is attempted
and the receipt lists what landed and what did not, so a partial seed is fully
visible rather than something you have to reconstruct.

---

## The one-time operator step (honest version)

**Minting the machine identity may require the Infisical web UI once, per
instance.** Everything after that is headless forever.

A machine identity cannot bootstrap itself: you need an authenticated
administrator to create the identity, grant it access to the project, and
issue Universal Auth credentials. That first act is a human one.

You may already have this. Check before doing anything manual:

```bash
env | cut -d= -f1 | grep -E '^INFISICAL_(CLIENT_ID|CLIENT_SECRET|ADDR|PROJECT_ID)$'
grep -o 'INFISICAL_[A-Z_]*' ~/.omnibase/.env 2>/dev/null | sort -u
```

Both commands print **names only** — never pipe these files anywhere that
prints values.

If the credentials already exist, you are done; skip to the use case above. If
they do not, either:

- run `scripts/bootstrap-infisical.sh`, which orchestrates identity
  provisioning against an instance you can already authenticate to; or
- create the identity in the UI once: **Organization → Access Control →
  Identities → Create**, auth method **Universal Auth**, then grant it the
  project role that permits secret writes, and copy the client id/secret.

Give the identity write access to the specific project and environment you
intend to seed, not organization-wide admin.

---

## Design notes

- **Reuse, not a new write path.** The handler composes
  `InfisicalSecretStore.set_secret` (OMN-10557) over `AdapterInfisical`'s
  `create_secret`/`update_secret` (OMN-2286) — the same path
  `omnimarket.projection.credential_publisher` has used in production for
  customer BYOK intake. Nothing about how secrets are written changed; this
  ticket added the missing canonical node in front of it.
- **Idempotent.** `set_secret` is update-then-create, so re-running the same
  command is a clean update, not a duplicate.
- **No bus topics.** The contract declares no `subscribe_topics` and no
  `publish_topics`. A declared subscribe topic is how the runtime auto-wires a
  live consumer, which would make a secret-writing node triggerable by anyone
  who can publish. Invocation is local and operator-initiated only.
- **Read side unchanged.** Runtime reads still go through `HandlerInfisical`
  and the config prefetcher; config values still hold *refs*, not literals
  (OMN-16451). This node is the write-side complement of that discipline.

## See also

- `config/shared_key_registry.yaml` — which key belongs at which `/shared/<transport>/` path
- `config/infisical_projects.yaml` — project registry
- `scripts/bootstrap-infisical.sh` — first-time identity provisioning
- `docs/patterns/service_catalog.md` — where Infisical runs in each lane
