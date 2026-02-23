# onboard-repo.py Setup

## Overview

`~/.omnibase/scripts/onboard-repo.py` strips shared env vars from a repo's `.env` file.
It reads `shared_key_registry.yaml` (not `~/.omnibase/.env`) to determine which keys are shared.

This document records the implementation for OMN-2479.

## Home-Dir Files Created

### `~/.omnibase/scripts/onboard-repo.py`

Strips shared keys from a repo `.env` using the registry. Key behavior:

- Reads `shared` section only — `bootstrap_only` and `identity_defaults` are excluded
- Uses `OMNIBASE_SHARED_KEY_REGISTRY` env var with fallback to `~/.omnibase/infra/shared_key_registry.yaml`
- Raises `FileNotFoundError` (not silent empty set) when registry missing
- Sanity-checks that `POSTGRES_PASSWORD` and `INFISICAL_*` are never stripped

### `~/.omnibase/infra/shared_key_registry.yaml`

Copy of `config/shared_key_registry.yaml` from omnibase_infra. Update when registry changes.

## Required Shell Config

Add to `~/.zshrc`:

```bash
export OMNIBASE_SHARED_KEY_REGISTRY=/path/to/omnibase_infra/config/shared_key_registry.yaml
```

Replace `/path/to/omnibase_infra` with the actual path to your omnibase_infra checkout.

## Usage

```bash
# Dry-run to see what would be stripped
python3 ~/.omnibase/scripts/onboard-repo.py /path/to/repo --dry-run

# Apply stripping
python3 ~/.omnibase/scripts/onboard-repo.py /path/to/repo
```

## Verification

POSTGRES_PASSWORD must NOT appear in the strip list:

```bash
python3 ~/.omnibase/scripts/onboard-repo.py /path/to/repo --dry-run 2>&1 | grep -v POSTGRES_PASSWORD
```

INFISICAL_* secrets must NOT appear in the strip list.
