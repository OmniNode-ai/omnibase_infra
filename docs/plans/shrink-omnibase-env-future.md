# Future: Shrink ~/.omnibase/.env to 5 Bootstrap Lines

## Overview

OMN-2484: The permanent end-state of OMN-2481 (transitional expanded .env).
Not yet executable.

## Prerequisite (UNMET)

All docker-compose services must be wired to fetch from Infisical on boot
(INFISICAL_ADDR + machine identity credentials at container startup).

Until this wiring is complete, `~/.omnibase/.env` must remain expanded
(OMN-2481 state) to provide all shared vars to services.

## Target State

Once prerequisite is met, `~/.omnibase/.env` shrinks to:

```bash
# OmniNode — bootstrap only (cannot come from Infisical)
POSTGRES_PASSWORD=<set from secure source>
INFISICAL_ADDR=http://localhost:8880
INFISICAL_CLIENT_ID=<set from secure source>
INFISICAL_CLIENT_SECRET=<set from secure source>
INFISICAL_PROJECT_ID=<set from secure source>
```

Everything else is fetched by containers from Infisical `/shared/*` at startup.

## Why These 5 Stay

- `POSTGRES_PASSWORD` — circular bootstrap dep: Infisical needs Postgres to start
- `INFISICAL_ADDR` / `INFISICAL_CLIENT_*` / `INFISICAL_PROJECT_ID` — Infisical needs
  these to authenticate; they cannot come from itself

## Steps (when prerequisite is met)

1. Verify all runtime containers boot using only Infisical-fetched credentials
2. Remove all non-bootstrap vars from `~/.omnibase/.env`
3. Verify: `echo $POSTGRES_HOST` → empty in new terminal
4. Verify services still start (getting POSTGRES_HOST from Infisical, not shell)
