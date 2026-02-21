# Handoff: Infisical Automated Provisioning

**Branch**: `jonah/infisical-automated-provisioning`
**Ticket**: OMN-2287
**Date**: 2026-02-21

---

## Summary

This session implemented automated Infisical provisioning for `omnibase_infra`. The previous `setup-infisical-identity.sh` was a stub with no real logic. It has been replaced by `scripts/provision-infisical.py`, a fully automated 8-step provisioning script that takes a fresh Infisical instance to a fully provisioned state with a single command.

Infisical is running at `http://localhost:8880` and has been seeded with 19 secrets across `/shared/<transport>/` folder paths in the `prod` environment.

---

## Quick Start from Scratch

Use this sequence on a new machine or after dropping the Infisical database.

### Prerequisites

- Docker running
- `.env` file present with at minimum `POSTGRES_PASSWORD` and the bootstrap vars from `.env.example`
- `uv` installed

### Step 1: Start Infisical

```bash
docker compose \
  -p omnibase-infra-runtime \
  -f /Users/jonah/.omnibase/infra/deployed/0.7.0/docker/docker-compose.infra.yml \
  --profile secrets \
  up -d infisical
```

Wait for Infisical to be healthy (check `http://localhost:8880`).

### Step 2: Run full bootstrap

```bash
bash scripts/bootstrap-infisical.sh
```

This script:
1. Waits for Infisical to become healthy
2. Waits for PostgreSQL to become healthy
3. Waits for Valkey to become healthy
4. Calls `uv run python scripts/provision-infisical.py` (Step 4 of bootstrap)
5. Re-sources `.env` to pick up the new Infisical credentials
6. Runs `uv run python scripts/seed-infisical.py --import-env docs/env-example-full.txt --set-values`

On completion, `.env` will contain:
```
INFISICAL_ADDR=http://localhost:8880
INFISICAL_CLIENT_ID=<generated>
INFISICAL_CLIENT_SECRET=<generated>
INFISICAL_PROJECT_ID=<generated>
```

And `.infisical-admin-token` will be written to the repo root (gitignored).

### Step 3: Verify

```bash
# Check secrets are in Infisical
uv run python scripts/seed-infisical.py --dry-run

# Or open the UI
open http://localhost:8880
```

---

## What Was Built

### `scripts/provision-infisical.py` (new)

Replaces the stub `setup-infisical-identity.sh`. Fully idempotent: if all four Infisical credentials (`INFISICAL_ADDR`, `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`, `INFISICAL_PROJECT_ID`) are already present in `.env`, the script exits cleanly without doing anything.

The 8 steps it performs:

| Step | Action | API |
|------|--------|-----|
| 1 | Bootstrap admin user + org | `POST /api/v1/admin/bootstrap` |
| 2 | Create project `omninode-infra` | Projects API |
| 3 | Create `/shared/<transport>/` folder structure in dev/staging/prod | Folders API |
| 4 | Create `onex-runtime` machine identity with Universal Auth | Identities API |
| 5 | Generate client credentials (client_id + client_secret) | Universal Auth API |
| 6 | Add identity to project as `admin` role | Project Members API |
| 7 | Write all four Infisical vars to `.env` | (file write) |
| 8 | Save admin token to `.infisical-admin-token` | (file write) |

### `scripts/bootstrap-infisical.sh` (modified)

- Step 4 now calls `provision-infisical.py` instead of the stub
- Step 5 re-sources `.env` after provisioning so subsequent steps see the new credentials, then runs `seed-infisical.py --import-env docs/env-example-full.txt --set-values`

### `scripts/validation/validate_clean_root.py` (modified)

Added `.infisical-identity` and `.infisical-admin-token` to the allowlist of permitted root-level files.

### `.gitignore` (modified)

Added `.infisical-admin-token` entry.

---

## Current Secrets in Infisical

19 secrets seeded into the `prod` environment across the `/shared/` folder tree:

| Path | Key | Value |
|------|-----|-------|
| `/shared/consul/` | `CONSUL_HOST` | `192.168.86.200` |
| `/shared/consul/` | `CONSUL_PORT` | `28500` |
| `/shared/consul/` | `CONSUL_SCHEME` | from `.env` |
| `/shared/consul/` | `CONSUL_ACL_TOKEN` | empty |
| `/shared/db/` | `POSTGRES_DSN` | `postgresql://postgres:***@omninode-bridge-postgres:5436/omnibase_infra` |
| `/shared/db/` | `POSTGRES_POOL_MIN` | empty |
| `/shared/db/` | `POSTGRES_POOL_MAX` | empty |
| `/shared/db/` | `POSTGRES_TIMEOUT_MS` | empty |
| `/shared/http/` | `HTTP_BASE_URL` | empty |
| `/shared/http/` | `HTTP_TIMEOUT_MS` | empty |
| `/shared/http/` | `HTTP_MAX_RETRIES` | empty |
| `/shared/mcp/` | `MCP_SERVER_HOST` | empty |
| `/shared/mcp/` | `MCP_SERVER_PORT` | empty |
| `/shared/graph/` | `GRAPH_HOST` | empty |
| `/shared/graph/` | `GRAPH_PORT` | empty |
| `/shared/graph/` | `GRAPH_PROTOCOL` | empty |
| `/shared/env/` | `SLACK_WEBHOOK_URL` | from `.env` |
| `/shared/env/` | `SLACK_BOT_TOKEN` | empty |
| `/shared/env/` | `SLACK_CHANNEL_ID` | empty |

Empty values must be filled via the Infisical UI at `http://localhost:8880` or via another seed pass once the values are known.

---

## Current `.env` State

The following variables are now in `.env` as a result of this session:

```
INFISICAL_ADDR=http://localhost:8880
INFISICAL_CLIENT_ID=aa952406-19a0-4da9-9c97-42658b2d1135
INFISICAL_CLIENT_SECRET=<set>
INFISICAL_PROJECT_ID=1efd8d15-99f3-429b-b973-3b10491af449
```

### Variables NOT yet migrated to Infisical

These remain as flat `.env` vars and the rationale for each:

| Variable | Reason |
|----------|--------|
| `POSTGRES_PASSWORD` | Bootstrap dependency — Infisical itself needs postgres to start. Circular. Stays in `.env` permanently. |
| `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DATABASE`, `POSTGRES_USER` | Still read by `ConfigSessionStorage` (`src/omnibase_infra/services/session/config_store.py`). Tracked in OMN-2065. |
| `KAFKA_BOOTSTRAP_SERVERS` | Not declared in any node contract, so contract discovery never picks it up. Bootstrap-level concern. Stays in `.env` until explicitly decided otherwise. |
| `CONSUL_HOST`, `CONSUL_PORT` | Now in Infisical at `/shared/consul/` but still in `.env`. Can be removed once containers read from Infisical. |
| `VAULT_ADDR`, `VAULT_TOKEN` | Not yet migrated. |

---

## Remaining Work

### 1. Wire `INFISICAL_ADDR` into running containers

`ConfigPrefetcher` activates only when `INFISICAL_ADDR` is set in the container environment. Currently the running containers have it blank, so prefetch does not activate. Add to `docker-compose.yml` environment blocks for:

- `omninode-runtime`
- `omninode-runtime-effects`
- worker containers

```yaml
environment:
  INFISICAL_ADDR: ${INFISICAL_ADDR}
  INFISICAL_CLIENT_ID: ${INFISICAL_CLIENT_ID}
  INFISICAL_CLIENT_SECRET: ${INFISICAL_CLIENT_SECRET}
  INFISICAL_PROJECT_ID: ${INFISICAL_PROJECT_ID}
```

### 2. Fill in empty secrets

Many keys were created with empty values because the actual values are not in `.env` or `.env.example`. Fill them via the UI at `http://localhost:8880` or run another seed pass after populating the source values:

```bash
uv run python scripts/seed-infisical.py --import-env docs/env-example-full.txt --set-values
```

### 3. Migrate `ConfigSessionStorage` (OMN-2065)

`src/omnibase_infra/services/session/config_store.py` still reads flat `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_DATABASE`. Once this is migrated to read `POSTGRES_DSN` from Infisical, those four flat vars can be removed from `.env`.

### 4. Remove redundant flat vars from `.env`

After containers are reading from Infisical:
- Remove `CONSUL_HOST`, `CONSUL_PORT` (now in `/shared/consul/`)
- Remove `VAULT_ADDR`, `VAULT_TOKEN` (once migrated)

### 5. Decide on `KAFKA_BOOTSTRAP_SERVERS`

Two options:
- **Keep in `.env`**: treat it as a bootstrap concern, same as `POSTGRES_PASSWORD`
- **Move to Infisical**: add a `/shared/kafka/` folder, add `KAFKA_BOOTSTRAP_SERVERS` to contract discovery, declare it in node contracts

### 6. Add 13 unmapped handler types to `TransportConfigMap`

The contract config discovery system does not know about these handler types, so their config keys are never auto-discovered. Add mappings in `runtime/config_discovery/transport_config_map.py` for:

- `architecture_validation`
- `intent`
- `auth_gate`
- `memgraph`
- `ledger_projection` (appears twice)
- `mock`
- `repo_state`
- `runtime_target`
- `toolchain`
- `rrh_storage`
- `rrh_validate`
- `validation_ledger_projection`

### 7. Per-repo Infisical onboarding

Each other repo needs the four Infisical credentials added to its `.env`. Service-specific secrets should use per-service paths:

```
/services/omniclaude/db/POSTGRES_DSN
/services/omniintelligence/kafka/KAFKA_BOOTSTRAP_SERVERS
```

Repos to update: `omniclaude`, `omniintelligence`, `omniarchon`, `omninode_bridge`, `omnidash`, `omnidash2`, `omnibase_core`.

---

## Gotchas

**Bootstrap is one-time only.** `POST /api/v1/admin/bootstrap` returns HTTP 400 on subsequent calls. The admin token is saved to `.infisical-admin-token` for re-runs. If this file is lost, you must drop and recreate the `infisical_db` database before provisioning again.

**`onex-runtime` identity requires `admin` role.** The same machine identity is used for both seeding (write) and runtime prefetch (read). `viewer` is insufficient for seeding. Downgrading to `viewer` after seeding is complete is an option if least-privilege is desired.

**Secrets must be in the `prod` environment.** `ModelInfisicalAdapterConfig` defaults to `environment_slug="prod"`. Secrets written to `dev` or `staging` will not be found by the runtime unless the slug is overridden.

**Folders must be pre-created.** Infisical does not auto-create folders on secret write. The provision script creates `/shared/<transport>/` folders in all three environments (dev, staging, prod) during Step 3.

**Compose project name.** All containers use project name `omnibase-infra-runtime`. Always pass `-p omnibase-infra-runtime` to `docker compose` commands to avoid creating duplicate containers under a different project name.

**`infisical-sdk` API.** `update_secret_by_name` uses `current_secret_name` as the first positional argument, not `secret_name`.

---

## Files Changed in This Branch

```
scripts/provision-infisical.py              (new)
scripts/bootstrap-infisical.sh              (modified)
scripts/validation/validate_clean_root.py   (modified)
.gitignore                                  (modified)
```
