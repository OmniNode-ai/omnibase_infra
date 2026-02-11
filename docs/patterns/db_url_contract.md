# Per-Service Database URL Contract

> **OMN-2146** | Phase 0.2 (Fix the foundation)
> Part of: DB-Per-Repo -- Break Coupling, Separate Databases, Deploy to Cloud

## Overview

Each OmniNode service owns its own PostgreSQL database and connects via a
dedicated role. The canonical connection method is a single `*_DB_URL`
environment variable containing a full PostgreSQL DSN.

## The Contract

| Environment Variable | Database | Role |
|---|---|---|
| `OMNIBASE_INFRA_DB_URL` | `omnibase_infra` | `role_omnibase_infra` |
| `OMNIINTELLIGENCE_DB_URL` | `omniintelligence` | `role_omniintelligence` |
| `OMNICLAUDE_DB_URL` | `omniclaude` | `role_omniclaude` |
| `OMNIMEMORY_DB_URL` | `omnimemory` | `role_omnimemory` |
| `OMNINODE_CLOUD_DB_URL` | `omninode_cloud` | `role_omninode_cloud` |
| `OMNIDASH_ANALYTICS_DB_URL` | `omnidash_analytics` | `role_omnidash` |

### URL Format

```
postgresql://<role>:<password>@<host>:<port>/<database>
```

Example:

```bash
OMNIBASE_INFRA_DB_URL=postgresql://role_omnibase_infra:s3cret@db.example.com:5432/omnibase_infra
```

## Resolution Order

`ModelPostgresPoolConfig.from_env()` requires a single `*_DB_URL` variable:

1. **`OMNIBASE_INFRA_DB_URL`** - Full DSN (required). Host, port, user,
   password, and database are parsed from the URL.
2. **Error** - If the variable is not set, a `ValueError` is raised
   with a clear message. There is no silent fallback.

> **Note**: The test helper `PostgresConfig.from_env()` (in
> `tests/helpers/util_postgres.py`) additionally falls back to individual
> `POSTGRES_*` variables (`POSTGRES_HOST`, `POSTGRES_PORT`, etc.) for
> convenience in local development. The production `from_env()` does not.

## Fail-Fast Behavior

The implicit `omninode_bridge` default has been removed. Code that previously
connected to the shared `omninode_bridge` database by default now raises an
error unless explicitly configured.

The conceptual change is:

```python
# Before (OMN-2146)
database = os.getenv("POSTGRES_DATABASE", "omninode_bridge")  # silent coupling

# After (OMN-2146) — production code requires OMNIBASE_INFRA_DB_URL with no
# fallback.  The test helper (PostgresConfig.from_env) additionally falls
# back to individual POSTGRES_* variables for local convenience.
db_url = os.getenv("OMNIBASE_INFRA_DB_URL")
if not db_url:
    raise ValueError("Set OMNIBASE_INFRA_DB_URL")
```

## Migration from `omninode_bridge`

### For `.env` files

```bash
# Old
POSTGRES_DATABASE=omninode_bridge

# New (preferred)
OMNIBASE_INFRA_DB_URL=postgresql://role_omnibase_infra:<pw>@<host>:5432/omnibase_infra

# New (legacy individual vars)
POSTGRES_DATABASE=omnibase_infra
```

### For Docker Compose

Docker Compose defaults have been updated from `omninode_bridge` to
`omnibase_infra`:

```yaml
# docker-compose.infra.yml
POSTGRES_DATABASE: ${POSTGRES_DATABASE:-omnibase_infra}
```

### For Tests

Integration tests check `OMNIBASE_INFRA_DB_URL` first, then fall back to
individual `POSTGRES_*` variables. Tests skip gracefully when no database
is configured.
