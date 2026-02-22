# Delete Repo .env Files

## Overview

OMN-2482: Delete all per-repo `.env` files across OmniNode repos.
Only safe after P0–P4-A are complete (enforcement active, registry-driven tooling live,
~/.omnibase/.env expanded).

## Status (as of OMN-2474 epic run)

No `.env` files found in any OmniNode repos on this machine (`/Users/jonah/Code/`).
This task is effectively already done for the local environment.

If repos on `/Volumes/PRO-G40/Code/` have `.env` files, run the script below.

## Script

```bash
#!/bin/bash
# Step 1: Dry-run — verify only shared keys would be stripped
REPOS=(omnibase_infra omnibase_infra2 omniclaude omniintelligence omniintelligence2 omnidash omninode_bridge omniarchon)
CODE_ROOT=/Volumes/PRO-G40/Code  # Update if repos are elsewhere

for repo in "${REPOS[@]}"; do
    env_file="$CODE_ROOT/$repo/.env"
    [[ -f "$env_file" ]] && echo "EXISTS: $env_file"
done

# Step 2: Strip shared keys from repos that still have them
for repo in omniintelligence omnidash; do
    [[ -d "$CODE_ROOT/$repo" ]] && python3 ~/.omnibase/scripts/onboard-repo.py "$CODE_ROOT/$repo" --dry-run
done

# Step 3: Delete all remaining .env files
for repo in "${REPOS[@]}"; do
    env_file="$CODE_ROOT/$repo/.env"
    [[ -f "$env_file" ]] && rm "$env_file" && echo "Removed: $env_file"
done

# Step 4: Verify none remain
for repo in "${REPOS[@]}"; do
    [[ -f "$CODE_ROOT/$repo/.env" ]] && echo "STILL EXISTS: $repo/.env"
done && echo "Check complete."
```

## Safety Checks

Before running Step 3, verify these keys are NOT in the strip list:
- `POSTGRES_DATABASE` (identity_default, should never be in shared)
- `POSTGRES_PASSWORD` (bootstrap_only)
- `INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET` (bootstrap_only)

If any of these appear in the strip list — **stop and fix the registry**.
