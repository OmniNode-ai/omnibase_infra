# Smoke Test: Zero Repo .env

## Overview

OMN-2483: Verify zero-repo-env state works end-to-end after P0–P4-B are complete.

## Prerequisites

- OMN-2477 (Remove ConfigSessionStorage env prefix) must be merged
- OMN-2481 (~/.omnibase/.env expanded) must be complete with real values
- No repo .env files present

## Test Results (OMN-2474 epic run, 2026-02-22)

### Step 1: Shell env

```
POSTGRES_HOST=192.168.86.200        ✓
POSTGRES_PASSWORD=__placeholder__   ⚠ needs real value
KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29092  ✓
INFISICAL_ADDR=http://localhost:8880  ✓
```

### Step 2: ConfigSessionStorage

BLOCKED: OMN-2477 (PR #391) not yet merged into main.
ConfigSessionStorage still uses `env_prefix="OMNIBASE_INFRA_SESSION_STORAGE_"` on origin/main.
Re-run after PR #391 is merged and POSTGRES_PASSWORD is filled with real value.

### Step 3: Unit tests

Not run (Step 2 failed — blocked on OMN-2477 merge).

### Step 4: Docker compose

Not run (requires Steps 1-3 to pass and real POSTGRES_PASSWORD).

## How to Re-run

After OMN-2477 is merged and ~/.omnibase/.env has real values:

```bash
# Step 1: verify env
source ~/.omnibase/.env && echo "POSTGRES_HOST=$POSTGRES_HOST"

# Step 2: ConfigSessionStorage
cd /path/to/omnibase_infra
uv run python3 -c "
from omnibase_infra.services.session.config_store import ConfigSessionStorage
cfg = ConfigSessionStorage()
print('host:', cfg.postgres_host)
print('database:', cfg.postgres_database)
print('dsn_safe:', cfg.dsn_safe)
"

# Step 3: unit tests
uv run pytest tests/ -m unit -n auto 2>&1 | tail -5

# Step 4: service boot
docker compose --env-file ~/.omnibase/.env -p omnibase-infra-runtime up -d runtime
sleep 10
docker logs omnibase-infra-runtime-runtime-1 2>&1 | grep -E "(Connected|ERROR|WARN)" | head -20
```
