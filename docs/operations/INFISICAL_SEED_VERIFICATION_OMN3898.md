# Infisical Seed Verification — OMN-3898

**Date**: 2026-03-08
**Ticket**: OMN-3898
**Epic**: OMN-3890
**Depends on**: OMN-3893 (prefetch contract scan decoupling)

## Summary

Operational verification of `seed-infisical.py` after OMN-3893 decoupled the prefetch
contract scan from handler contract paths. This document records the verified state of
Infisical transport folders following a successful seed run.

## Pre-conditions

- OMN-3893 commit `e474e0a6` on branch `jonah/omn-3893-decouple-prefetch-contract-scan`
- Infisical running at `http://localhost:8880` (local Docker, `--profile secrets`)
- `~/.omnibase/.env` sourced with valid `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`,
  `INFISICAL_PROJECT_ID`, `INFISICAL_ADDR`

## Seed Script Dry-Run Output

```
2026-03-08 14:56:59 [INFO] seed-infisical: Scanning contracts in src/omnibase_infra/nodes
2026-03-08 14:57:10 [INFO] ContractConfigExtractor: Config extraction complete:
  59 requirements from 54 contracts (7 transport types, 0 errors)
2026-03-08 14:57:10 [INFO] seed-infisical: Found 24 config requirements
2026-03-08 14:57:10 [INFO] seed-infisical: Dry run complete. Use --execute to write to Infisical.

--- Seed Diff Summary ---
Total keys discovered: 24
Values available from .env: 6
Mode: SKIP existing (overwrite-existing is off)

  [CREATE] /shared/filesystem/FS_BASE_PATH
  [CREATE] /shared/env/GITHUB_TOKEN
  [CREATE] /shared/env/GMAIL_CLIENT_ID
  [CREATE] /shared/env/GMAIL_CLIENT_SECRET
  [CREATE] /shared/env/GMAIL_REFRESH_TOKEN
  [CREATE] /shared/graph/GRAPH_HOST
  [CREATE] /shared/graph/GRAPH_PORT
  [CREATE] /shared/graph/GRAPH_PROTOCOL
  [CREATE] /shared/http/HTTP_BASE_URL
  [CREATE] /shared/http/HTTP_MAX_RETRIES
  [CREATE] /shared/http/HTTP_TIMEOUT_MS
  [CREATE] /shared/env/LINEAR_API_KEY
  [CREATE] /shared/env/LINEAR_TEAM_ID
  [CREATE] /shared/mcp/MCP_SERVER_HOST
  [CREATE] /shared/mcp/MCP_SERVER_PORT
  [CREATE] /shared/db/POSTGRES_HOST          (has .env value)
  [CREATE] /shared/db/POSTGRES_POOL_MAX_SIZE
  [CREATE] /shared/db/POSTGRES_POOL_MIN_SIZE
  [CREATE] /shared/db/POSTGRES_PORT          (has .env value)
  [CREATE] /shared/db/POSTGRES_TIMEOUT_MS    (has .env value)
  [CREATE] /shared/db/POSTGRES_USER          (has .env value)
  [CREATE] /shared/db/QUERY_TIMEOUT_SECONDS
  [CREATE] /shared/env/SLACK_BOT_TOKEN       (has .env value)
  [CREATE] /shared/env/SLACK_CHANNEL_ID      (has .env value)
```

## Prerequisite: Folder Structure Creation

**Root cause discovered**: The `prod` and `staging` environments had no folder structure.
`provision-infisical.py` creates folders via `provision_transport_folders()`, but it
short-circuits when `INFISICAL_CLIENT_ID/SECRET/PROJECT_ID` are already in `~/.omnibase/.env`
(indicating it was previously run). In this case the folder creation step was incomplete.

**Resolution**: Manually created `/shared` and all transport subfolders in `prod` and
`staging` environments via the Infisical folders API (same calls as `provision_transport_folders`).
Also added missing folders (`consul`, `kafka`, `vault`, `qdrant`, `auth`) to `dev`.

## Execute Run Output

```
2026-03-08 15:01:35 [INFO] Infisical adapter initialized and authenticated
2026-03-08 15:01:35 [INFO] Created secret: POSTGRES_HOST at /shared/db/
2026-03-08 15:01:35 [INFO] Created secret: POSTGRES_PORT at /shared/db/
2026-03-08 15:01:35 [INFO] Created secret: POSTGRES_USER at /shared/db/
2026-03-08 15:01:35 [INFO] Created secret: POSTGRES_POOL_MIN_SIZE at /shared/db/
2026-03-08 15:01:35 [INFO] Created secret: POSTGRES_POOL_MAX_SIZE at /shared/db/
2026-03-08 15:01:35 [INFO] Created secret: POSTGRES_TIMEOUT_MS at /shared/db/
2026-03-08 15:01:35 [INFO] Created secret: QUERY_TIMEOUT_SECONDS at /shared/db/
2026-03-08 15:01:35 [INFO] Created secret: GRAPH_HOST at /shared/graph/
2026-03-08 15:01:35 [INFO] Created secret: GRAPH_PORT at /shared/graph/
2026-03-08 15:01:35 [INFO] Created secret: GRAPH_PROTOCOL at /shared/graph/
2026-03-08 15:01:35 [INFO] Created secret: HTTP_BASE_URL at /shared/http/
2026-03-08 15:01:35 [INFO] Created secret: HTTP_TIMEOUT_MS at /shared/http/
2026-03-08 15:01:35 [INFO] Created secret: HTTP_MAX_RETRIES at /shared/http/
2026-03-08 15:01:35 [INFO] Created secret: MCP_SERVER_HOST at /shared/mcp/
2026-03-08 15:01:35 [INFO] Created secret: MCP_SERVER_PORT at /shared/mcp/
2026-03-08 15:01:35 [INFO] Created secret: FS_BASE_PATH at /shared/filesystem/
2026-03-08 15:01:35 [INFO] Created secret: LINEAR_API_KEY at /shared/env/
2026-03-08 15:01:35 [INFO] Created secret: LINEAR_TEAM_ID at /shared/env/
2026-03-08 15:01:35 [INFO] Created secret: GITHUB_TOKEN at /shared/env/
2026-03-08 15:01:35 [INFO] Created secret: GMAIL_CLIENT_ID at /shared/env/
2026-03-08 15:01:35 [INFO] Created secret: GMAIL_CLIENT_SECRET at /shared/env/
2026-03-08 15:01:35 [INFO] Created secret: GMAIL_REFRESH_TOKEN at /shared/env/
2026-03-08 15:01:35 [INFO] Created secret: SLACK_BOT_TOKEN at /shared/env/
2026-03-08 15:01:35 [INFO] Created secret: SLACK_CHANNEL_ID at /shared/env/
2026-03-08 15:01:36 [INFO] Seed complete: 24 created, 0 updated, 0 skipped, 0 errors
```

## Verified Infisical Folder Structure

Transport folders present in `prod` environment after seed:

| Path | Keys Created | Status |
|------|-------------|--------|
| `/shared/auth` | (none from current contracts) | folder exists |
| `/shared/consul` | (none from current contracts) | folder exists |
| `/shared/db` | POSTGRES_HOST, POSTGRES_POOL_MAX_SIZE, POSTGRES_POOL_MIN_SIZE, POSTGRES_PORT, POSTGRES_TIMEOUT_MS, POSTGRES_USER, QUERY_TIMEOUT_SECONDS | populated |
| `/shared/env` | GITHUB_TOKEN, GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN, LINEAR_API_KEY, LINEAR_TEAM_ID, SLACK_BOT_TOKEN, SLACK_CHANNEL_ID | populated |
| `/shared/filesystem` | FS_BASE_PATH | populated |
| `/shared/graph` | GRAPH_HOST, GRAPH_PORT, GRAPH_PROTOCOL | populated |
| `/shared/http` | HTTP_BASE_URL, HTTP_MAX_RETRIES, HTTP_TIMEOUT_MS | populated |
| `/shared/kafka` | (none from current contracts) | folder exists |
| `/shared/mcp` | MCP_SERVER_HOST, MCP_SERVER_PORT | populated |
| `/shared/qdrant` | (none from current contracts) | folder exists |
| `/shared/vault` | (none from current contracts) | folder exists |

## Contract Discovery Statistics (OMN-3893 validation)

The OMN-3893 change decoupled `ConfigPrefetcher`'s contract scan from handler-specific paths.
Post-OMN-3893 scan results:

- **54 contracts** scanned (from `src/omnibase_infra/nodes/`)
- **59 requirements** extracted
- **7 transport types** identified
- **0 errors** during extraction
- **24 unique config keys** after deduplication (bootstrap transports excluded)

## Definition of Done

- [x] Seed script runs without errors (24 created, 0 errors)
- [x] `/shared/db/` folder exists in Infisical with 7 keys
- [x] `/shared/kafka/` folder exists in Infisical (empty — no Kafka contracts in scope)
- [x] `/shared/http/` folder exists in Infisical with 3 keys
- [x] `/shared/filesystem/` folder exists in Infisical with 1 key
- [x] At least baseline transport folders present (11 folders total)

## Follow-up: provision-infisical.py Idempotency Gap

The `provision_transport_folders()` function in `provision-infisical.py` is only called
during initial provisioning. If folders are missing from an environment (e.g. after
manually adding a new environment or partial initial setup), the script will short-circuit
without creating folders if credentials already exist in `~/.omnibase/.env`.

**Recommendation**: Add a `--only-folders` flag to `provision-infisical.py` that skips
credential provisioning and only runs `provision_transport_folders()`. This allows idempotent
folder re-creation without full re-provisioning. Track as a follow-up ticket.
