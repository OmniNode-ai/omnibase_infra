# Zero Repo .env Files Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all per-repo `.env` files so every clone of every repo works identically
with zero local configuration drift.

**Architecture:** All shared infrastructure credentials live in `~/.omnibase/.env` (sourced at
shell startup via `~/.zshrc`). Service identity config (database name) is hardcoded as a default
in each Settings class. A versioned `shared_key_registry.yaml` in this repo defines which keys
are shared, so onboarding tooling never depends on the mutable `~/.omnibase/.env` for its key
list. Infisical handles runtime secret distribution for containers.

**Tech Stack:** Python 3.12, pydantic-settings, YAML, pre-commit, uv

**Priority Order:**
- **P0** — Enforcement first (prevents backsliding before anything is deleted)
- **P1** — Identity defaults in code (kills drift at the source)
- **P2** — Registry before automation (stable foundation for tooling)
- **P3** — Tooling changes (deterministic, registry-driven stripping)
- **P4** — Delete repo `.env` files (only safe once P0–P3 are done)
- **P5** — Infisical writes (real side-effects happen last)

---

## Background

`~/.zshrc` already runs:
```bash
if [[ -f "$HOME/.omnibase/.env" ]]; then
    set -a; source "$HOME/.omnibase/.env"; set +a
fi
```

pydantic-settings resolution order: **env vars > .env file**. Any var exported by shell startup
takes precedence over the `.env` file. This means if all shared vars are in `~/.omnibase/.env`,
no repo `.env` is needed for those vars.

The one remaining issue is `ConfigSessionStorage` uses `env_prefix="OMNIBASE_INFRA_SESSION_STORAGE_"`,
making it blind to standard `POSTGRES_*` shell env vars. Removing the prefix fixes this.

**Canonical sources of truth:**
- `shared_key_registry.yaml` + Infisical (`/shared/*`) — canonical for shared secrets
- `~/.omnibase/.env` — local dev convenience mirror only, never canonical
- Settings class `default=` values — canonical for per-repo identity config (e.g., `POSTGRES_DATABASE`)

---

## Task 1 (P0): Remove `.env` from Allowed Root Files — Enforcement First

Make it structurally impossible to accidentally commit `.env` or re-introduce it as a pattern.
Enforcement must happen **before** any deletion.

**Files:**
- Modify: `scripts/validation/validate_clean_root.py`

**Step 1: Find existing tests**

```bash
grep -r "validate_clean_root\|validate_root" tests/ -l
```

**Step 2: Remove `.env` and `.envrc` from `ALLOWED_ROOT_FILES`**

Remove `.env` from `ALLOWED_ROOT_FILES`. Keep `.env.example` — it is safe to commit because
`.gitignore` negates it with `!.env.example`. `.env.template` was also removed from the allowlist
because `.gitignore` does not negate it (only `!.env.example` is negated), so `.env.template`
would be uncommittable anyway and the allowlist entry would be dead code.

Also remove `.env` from `ALLOWED_ROOT_DIRECTORIES` if present (`.env` is not a virtualenv name —
this is a confusing artifact from earlier in the codebase).

While editing: also remove `.envrc` from allowed files if present, or explicitly add it to the
denied list. direnv creep is the same problem under a different filename — if you're not using
direnv intentionally, block it here.

**Step 3: Run existing validate_clean_root tests**

```bash
uv run pytest tests/ -k "clean_root or validate_root" -v
```

All should pass.

**Step 4: Run the script directly against the repo**

```bash
python3 scripts/validation/validate_clean_root.py
```

If `.env` exists locally, you'll see a violation. That's correct — the script now enforces the policy.

**Step 5: Commit**

```bash
git add scripts/validation/validate_clean_root.py
git commit -m "feat(validation): remove .env from allowed root files — enforce zero-repo-env policy"
```

---

## Task 2 (P0): Add Pre-commit Hook for `.env` Detection

Defense-in-depth: catch `.env` files at commit time, not just during validation runs.

**Files:**
- Modify: `.pre-commit-config.yaml`

**Step 1: Add a hook that rejects staged `.env` files**

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: no-env-file
      name: Reject committed .env files (anywhere in tree)
      entry: bash -c 'if git diff --cached --name-only | grep -qE "(^|/)\.env$"; then echo "ERROR: .env file staged — use ~/.omnibase/.env instead (subdir .env files also rejected)"; exit 1; fi'
      language: system
      pass_filenames: false
      stages: [commit]
```

The regex `(^|/)\.env$` matches `.env` at the root AND at any subpath (`config/.env`,
`docker/.env`, `secrets/.env`, etc.). Drift frequently sneaks in via subdirectories when the
root hook blocks only `^\.env$`.

**Step 2: Test the hook**

```bash
pre-commit run no-env-file --all-files
```

Expected: passes (no `.env` staged).

**Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hook to reject committed .env files"
```

---

## Task 3 (P1): Fix `ConfigSessionStorage` Env Prefix — Identity Defaults in Code

> **STATUS: COMPLETED** — The `env_prefix` removal has been implemented on branch
> `jonah/omn-2287-zero-repo-env-phase2`. `ConfigSessionStorage` now uses `env_prefix=""`
> and `env_file=None`, so it reads standard `POSTGRES_*` vars from the shell environment.
> Tests were updated to use unprefixed env var names. The historical plan text below is
> preserved for reference.

**Files:**
- Modify: `src/omnibase_infra/services/session/config_store.py`

**Background:**

Currently `ConfigSessionStorage` uses `env_prefix="OMNIBASE_INFRA_SESSION_STORAGE_"`. This means:
- `postgres_host` reads `OMNIBASE_INFRA_SESSION_STORAGE_POSTGRES_HOST` — not in `~/.omnibase/.env`
- `postgres_password` reads `OMNIBASE_INFRA_SESSION_STORAGE_POSTGRES_PASSWORD` — not in `~/.omnibase/.env`

Removing the prefix fixes both:
- `postgres_host` reads `POSTGRES_HOST` from shell env ✓
- `postgres_password` reads `POSTGRES_PASSWORD` from shell env ✓
- `postgres_database` defaults to `"omnibase_infra"` in code — no env var needed ✓

**Step 1: Find and run existing tests first**

```bash
uv run pytest tests/ -k "session_storage or config_store" -v 2>&1 | tail -30
```

Note current test count and pass/fail state.

**Step 2: Update `model_config` in `config_store.py`**

Change from:
```python
model_config = SettingsConfigDict(
    env_prefix="OMNIBASE_INFRA_SESSION_STORAGE_",
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
)
```

To:
```python
model_config = SettingsConfigDict(
    env_prefix="",
    env_file=None,       # No .env file — reads from shell env (sourced via ~/.omnibase/.env)
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
)
```

`env_file=None` is intentional. Leaving it as `".env"` is an attractive nuisance: pydantic-settings
will silently read any `.env` it finds in the working directory, undermining the zero-repo-env policy
and causing "works here, fails in CI" behavior.

Also update module docstring and class docstring to remove references to the old prefix. Document
that `postgres_database` defaults to `"omnibase_infra"` in code.

**Step 3: Update tests that set the prefixed env var**

```bash
grep -r "OMNIBASE_INFRA_SESSION_STORAGE" tests/
```

For each match, rename to the unprefixed version:
```python
# Before
os.environ["OMNIBASE_INFRA_SESSION_STORAGE_POSTGRES_PASSWORD"] = "test"

# After
os.environ["POSTGRES_PASSWORD"] = "test"
```

**Step 4: Run tests**

```bash
uv run pytest tests/ -k "session_storage or config_store" -v 2>&1 | tail -30
```

Expected: same count, all passing.

**Step 5: Run full unit suite to confirm no regressions**

```bash
uv run pytest tests/ -m unit -n auto 2>&1 | tail -5
```

Expected: all passing (~3330 tests).

**Step 6: Commit**

```bash
git add src/omnibase_infra/services/session/config_store.py
git commit -m "fix(session): remove env prefix so ConfigSessionStorage reads standard POSTGRES_* vars"
```

---

## Task 4 (P2): Create `shared_key_registry.yaml`

The authoritative, versioned definition of which keys are "shared across all repos". Tooling reads
this — never reads `~/.omnibase/.env` to determine key names.

**Registry semantics:**
- `shared` — eligible for Infisical seeding AND for stripping from repo `.env` files
- `bootstrap_only` — excluded from Infisical seeding (circular dependency); listed here for
  awareness only; **lives outside repos** (home dir or deployment secrets) — never stripped
  *because it must never be in a repo `.env` in the first place*
- `identity_defaults` — never in any `.env`; always as `default=` in Settings class code

**Files:**
- Create: `config/shared_key_registry.yaml`

**Step 1: Create the config directory**

```bash
mkdir -p /Volumes/PRO-G40/Code/omnibase_infra/config
```

**Step 2: Create the registry file**

```yaml
# shared_key_registry.yaml
#
# Authoritative list of keys shared across all OmniNode repos.
# Keys come from ~/.omnibase/.env (shell env) at dev time,
# and from Infisical /shared/<transport>/ at runtime.
#
# Semantics:
#   shared         — eligible for Infisical seeding + repo .env stripping
#   bootstrap_only — excluded from Infisical seeding (circular bootstrap dep);
#                    listed here for awareness only; never strip from repos
#   identity_defaults — never in any .env; hardcoded as default= in Settings classes
#
# Used by:
#   - ~/.omnibase/scripts/onboard-repo.py  (strip shared keys from repo .env)
#   - scripts/register-repo.py seed-shared  (seed /shared/* in Infisical)

version: "1.0"

shared:
  "/shared/db/":
    - POSTGRES_HOST
    - POSTGRES_PORT
    - POSTGRES_USER
    - POSTGRES_DSN
    - POSTGRES_POOL_MIN_SIZE  # renamed; was POSTGRES_POOL_MIN
    - POSTGRES_POOL_MAX_SIZE  # renamed; was POSTGRES_POOL_MAX
    - POSTGRES_TIMEOUT_MS

  "/shared/kafka/":
    - KAFKA_BOOTSTRAP_SERVERS
    - KAFKA_HOST_SERVERS
    - KAFKA_REQUEST_TIMEOUT_MS

  "/shared/consul/":
    - CONSUL_HOST
    - CONSUL_PORT
    - CONSUL_SCHEME
    - CONSUL_ACL_TOKEN
    - CONSUL_ENABLED

  "/shared/vault/":
    - VAULT_ADDR
    # VAULT_TOKEN: per-service only — excluded from /shared/. Each service must be
    # provisioned under /services/<repo>/vault/VAULT_TOKEN with a token scoped to
    # its own Vault policy. See shared_key_registry.yaml for the rationale.

  "/shared/llm/":
    - LLM_CODER_URL
    - LLM_CODER_FAST_URL
    - LLM_EMBEDDING_URL
    - LLM_DEEPSEEK_R1_URL
    - EMBEDDING_MODEL_URL
    - VLLM_SERVICE_URL
    - VLLM_DEEPSEEK_URL
    - VLLM_LLAMA_URL
    - ONEX_TREE_SERVICE_URL
    - METADATA_STAMPING_SERVICE_URL
    - REMOTE_SERVER_IP

  "/shared/auth/":
    - OPENAI_API_KEY
    - GOOGLE_API_KEY
    - GEMINI_API_KEY
    - Z_AI_API_KEY
    - Z_AI_API_URL
    - SERVICE_AUTH_TOKEN
    - GH_PAT

  "/shared/valkey/":
    - VALKEY_PASSWORD

  "/shared/env/":
    - SLACK_WEBHOOK_URL
    - SLACK_BOT_TOKEN
    - SLACK_CHANNEL_ID

# Keys that must NEVER be seeded into Infisical (circular bootstrap dependency —
# Infisical needs these to start, so they cannot come FROM Infisical).
# These live OUTSIDE repos: POSTGRES_PASSWORD in ~/.omnibase/.env,
# INFISICAL_* service secrets in ~/.omnibase/.env or deployment secrets.
# They must never appear in any repo .env file.
bootstrap_only:
  - POSTGRES_PASSWORD
  - INFISICAL_ADDR
  - INFISICAL_CLIENT_ID
  - INFISICAL_CLIENT_SECRET
  - INFISICAL_PROJECT_ID
  - INFISICAL_ENCRYPTION_KEY
  - INFISICAL_AUTH_SECRET

# Keys that are per-repo identity config.
# Must NEVER appear in any .env file — hardcoded as default= in Settings classes.
identity_defaults:
  - POSTGRES_DATABASE
```

**Step 3: Verify YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('config/shared_key_registry.yaml'))"
```

Expected: no output (parse succeeds).

**Step 4: Commit**

```bash
git add config/shared_key_registry.yaml
git commit -m "feat: add shared_key_registry.yaml as authoritative shared key definition"
```

---

## Task 5 (P3): Update `onboard-repo.py` to Read Registry

**Files:**
- Modify: `~/.omnibase/scripts/onboard-repo.py`

**Step 1: Replace `shared_keys()` function**

```python
# Prefer env override so this script isn't tied to Jonah's volume path.
# Set OMNIBASE_SHARED_KEY_REGISTRY=/path/to/config/shared_key_registry.yaml
# to point at any checkout.
_DEFAULT_REGISTRY = Path.home() / ".omnibase" / "infra" / "shared_key_registry.yaml"
REGISTRY_PATH = Path(
    os.environ.get("OMNIBASE_SHARED_KEY_REGISTRY", str(_DEFAULT_REGISTRY))
)


def shared_keys(registry: Path | None = None) -> set[str]:
    """Load shared key names from the registry YAML.

    Only returns keys from the 'shared' section — bootstrap_only and
    identity_defaults are intentionally excluded (must not be stripped).
    """
    import yaml

    path = registry or REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Registry not found at {path}. "
            f"Set OMNIBASE_SHARED_KEY_REGISTRY or copy config/shared_key_registry.yaml "
            f"to {_DEFAULT_REGISTRY}"
        )

    data = yaml.safe_load(path.read_text())
    keys: set[str] = set()
    for folder_keys in data.get("shared", {}).values():
        keys.update(folder_keys)
    # bootstrap_only and identity_defaults are NOT included — must not be stripped
    return keys
```

No silent fallback to an empty set — fail fast with a clear message. An empty fallback would silently
skip stripping keys and leave repos with drift that's invisible until someone notices.

`OMNIBASE_SHARED_KEY_REGISTRY` env var override removes the hardcoded machine path. Typical usage:
```bash
export OMNIBASE_SHARED_KEY_REGISTRY=/Volumes/PRO-G40/Code/omnibase_infra/config/shared_key_registry.yaml
# or add to ~/.zshrc alongside the .omnibase/.env source
```

Remove the old `SHARED_ENV` constant and old `shared_keys()` function.

**Step 2: Test against omniintelligence**

```bash
python3 ~/.omnibase/scripts/onboard-repo.py /Volumes/PRO-G40/Code/omniintelligence
```

Expected: shows shared keys to remove (POSTGRES_HOST, OPENAI_API_KEY, GH_PAT, etc.) but NOT
`POSTGRES_PASSWORD` or `INFISICAL_*` service secrets.

**Step 3: Test against omnidash**

```bash
python3 ~/.omnibase/scripts/onboard-repo.py /Volumes/PRO-G40/Code/omnidash
```

Verify only shared keys are flagged.

**Step 4: No commit** — this file lives in `~/.omnibase/scripts/`, not in the repo.

---

## Task 6 (P3): Update `register-repo.py` to Read Registry

**Files:**
- Modify: `scripts/register-repo.py`

**Step 1: Replace hardcoded `SHARED_PLATFORM_SECRETS` dict**

```python
_REGISTRY_PATH = _PROJECT_ROOT / "config" / "shared_key_registry.yaml"


def _load_registry() -> dict[str, list[str]]:
    """Load shared key registry from YAML. Fails fast if registry is missing."""
    import yaml
    if not _REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry not found: {_REGISTRY_PATH}\n"
            "Run Task 4 first to create config/shared_key_registry.yaml"
        )
    data = yaml.safe_load(_REGISTRY_PATH.read_text())
    return {folder: keys for folder, keys in data.get("shared", {}).items()}


def _bootstrap_keys() -> frozenset[str]:
    """Keys excluded from Infisical seeding (circular bootstrap dependency)."""
    import yaml
    data = yaml.safe_load(_REGISTRY_PATH.read_text())
    return frozenset(data.get("bootstrap_only", []))
```

No silent empty returns — a missing registry must be a loud failure, not a partial seed that
creates ghost paths in Infisical.

In `cmd_seed_shared`, replace:
```python
for folder, keys in SHARED_PLATFORM_SECRETS.items():
```
with:
```python
for folder, keys in _load_registry().items():
```

**Step 2: Verify dry-run still works**

```bash
cd /Volumes/PRO-G40/Code/omnibase_infra
source ~/.omnibase/.env
uv run python scripts/register-repo.py seed-shared
```

Expected: same keys as before (now from registry), dry-run output, no errors.

**Step 3: Commit**

```bash
git add scripts/register-repo.py
git commit -m "refactor: register-repo.py reads shared_key_registry.yaml instead of hardcoded dict"
```

---

## Task 7A (P4, Transitional): Expand `~/.omnibase/.env` to Full Shared Vars

> **Transitional step only.** This file will shrink back to 5 bootstrap lines once all repos and
> containers read from Infisical directly (Task 7B). Do not treat this expanded version as the
> permanent end-state.
>
> **Security note:** The template below shows key *names* only. Fill values from your secure
> source (1Password, Infisical admin UI, or the existing per-repo `.env` files before deletion).
> Never paste real secret values into plan documents.

**Files:**
- Modify: `~/.omnibase/.env` (home dir, never in any repo)

**Step 1: Write the expanded content (fill `<...>` placeholders from secure source)**

```bash
# =============================================================================
# OmniNode Platform — Shared Infrastructure Credentials
# ~/.omnibase/.env
#
# Sourced automatically by ~/.zshrc:
#   set -a; source ~/.omnibase/.env; set +a
#
# TRANSITIONAL: This is a dev-time mirror of /shared/* in Infisical.
# It will shrink to 5 bootstrap lines once all containers read from Infisical.
#
# NEVER add repo-specific vars here (POSTGRES_DATABASE, etc.)
# NEVER commit this file anywhere.
# =============================================================================

# PostgreSQL (host-accessible: direct IP, external port)
POSTGRES_HOST=192.168.86.200
POSTGRES_PORT=5436
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<set from secure source>

# Kafka/Redpanda
KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29092
KAFKA_HOST_SERVERS=192.168.86.200:29092

# Consul
CONSUL_HOST=192.168.86.200
CONSUL_PORT=28500
CONSUL_ENABLED=true

# Vault (VAULT_ADDR is shared platform-wide; VAULT_TOKEN is per-developer/per-service
# and NOT seeded to Infisical /shared/ — each service gets its own token scoped to
# its Vault policy, provisioned under /services/<repo>/vault/VAULT_TOKEN)
VAULT_ADDR=http://omninode-bridge-vault:8200
VAULT_TOKEN=<set from secure source — this is YOUR developer token, not a shared value>

# Infisical (shared across all repos — same project, same machine identity)
INFISICAL_ADDR=http://localhost:8880
INFISICAL_CLIENT_ID=<set from secure source>
INFISICAL_CLIENT_SECRET=<set from secure source>
INFISICAL_PROJECT_ID=<set from secure source>

# LLM endpoints
LLM_CODER_URL=http://192.168.86.201:8000
LLM_CODER_FAST_URL=http://192.168.86.201:8001
LLM_EMBEDDING_URL=http://192.168.86.200:8100
LLM_DEEPSEEK_R1_URL=http://192.168.86.200:8101
EMBEDDING_MODEL_URL=http://192.168.86.201:8002
VLLM_SERVICE_URL=http://192.168.86.201:8002
VLLM_DEEPSEEK_URL=http://192.168.86.201:8000/v1
VLLM_LLAMA_URL=http://192.168.86.201:8001/v1

# Platform services
ONEX_TREE_SERVICE_URL=http://192.168.86.200:8058
METADATA_STAMPING_SERVICE_URL=http://192.168.86.200:8057
REMOTE_SERVER_IP=192.168.86.200

# API keys (fill from Infisical /shared/auth/ or secure source)
OPENAI_API_KEY=<set from secure source>
GOOGLE_API_KEY=<set from secure source>
GEMINI_API_KEY=<set from secure source>
Z_AI_API_KEY=<set from secure source>
Z_AI_API_URL=<set from secure source>
SERVICE_AUTH_TOKEN=<set from secure source>
GH_PAT=<set from secure source>

# Valkey
VALKEY_PASSWORD=<set from secure source>

# Slack (optional)
SLACK_WEBHOOK_URL=
SLACK_BOT_TOKEN=
SLACK_CHANNEL_ID=
```

**Step 2: Verify the file is sourced in a new shell**

```bash
echo "POSTGRES_HOST=$POSTGRES_HOST"
echo "INFISICAL_ADDR=$INFISICAL_ADDR"
echo "LLM_CODER_URL=$LLM_CODER_URL"
```

Expected:
```
POSTGRES_HOST=192.168.86.200
INFISICAL_ADDR=http://localhost:8880
LLM_CODER_URL=http://192.168.86.201:8000
```

**Step 3: No commit** — this file lives in `~`, not in any repo.

---

## Task 7B (Future): Final Minimal `~/.omnibase/.env`

> **Do this AFTER** all repos and runtime containers read from Infisical directly.
> This is the permanent end-state of Task 7A, not the current step.

The file shrinks back to 5 bootstrap lines:

```bash
# OmniNode — bootstrap only (cannot come from Infisical)
POSTGRES_PASSWORD=<set from secure source>
INFISICAL_ADDR=http://localhost:8880
INFISICAL_CLIENT_ID=<set from secure source>
INFISICAL_CLIENT_SECRET=<set from secure source>
INFISICAL_PROJECT_ID=<set from secure source>
```

Everything else (POSTGRES_HOST, KAFKA_*, LLM_*, etc.) is read by containers from Infisical at startup.

---

## Task 8 (P4): Delete All Repo `.env` Files — Safely

Run only **after** Tasks 1–7A are complete.

**Step 1: Strip redundant keys from omniintelligence and omnidash first**

```bash
# Dry-run — confirm only shared keys are flagged (NOT POSTGRES_PASSWORD or INFISICAL_*)
python3 ~/.omnibase/scripts/onboard-repo.py /Volumes/PRO-G40/Code/omniintelligence
python3 ~/.omnibase/scripts/onboard-repo.py /Volumes/PRO-G40/Code/omnidash
```

Before applying, **print a diff of what would be removed** to catch over-stripping:

```bash
# For each repo, show exactly what keys would be stripped vs kept
for repo in omniintelligence omnidash; do
    echo "=== $repo ==="
    python3 ~/.omnibase/scripts/onboard-repo.py /Volumes/PRO-G40/Code/$repo 2>&1
    echo ""
done
```

Verify the "would strip" list contains only shared keys. If `POSTGRES_DATABASE`, `INFISICAL_CLIENT_ID`,
or any `INFISICAL_*` service secret appears in the strip list, **stop** — the registry or script
has a bug.

Then apply:

```bash
python3 ~/.omnibase/scripts/onboard-repo.py /Volumes/PRO-G40/Code/omniintelligence --apply
python3 ~/.omnibase/scripts/onboard-repo.py /Volumes/PRO-G40/Code/omnidash --apply
```

**Step 2: List `.env` files before deleting — OmniNode repos only**

```bash
REPOS=(omnibase_infra omnibase_infra2 omniclaude omniintelligence omniintelligence2 omnidash omninode_bridge omniarchon)
for repo in "${REPOS[@]}"; do
    env_file="/Volumes/PRO-G40/Code/$repo/.env"
    [[ -f "$env_file" ]] && echo "Would remove: $env_file"
done
```

Review the list, then delete:

```bash
for repo in "${REPOS[@]}"; do
    env_file="/Volumes/PRO-G40/Code/$repo/.env"
    if [[ -f "$env_file" ]]; then
        rm "$env_file"
        echo "Removed: $env_file"
    fi
done
```

Also clean up `.env.env.bak` files left by onboard-repo.py:

```bash
for repo in "${REPOS[@]}"; do
    bak="/Volumes/PRO-G40/Code/$repo/.env.env.bak"
    [[ -f "$bak" ]] && rm "$bak" && echo "Removed: $bak"
done
```

**Step 3: Verify none remain in OmniNode repos**

```bash
for repo in "${REPOS[@]}"; do
    [[ -f "/Volumes/PRO-G40/Code/$repo/.env" ]] && echo "STILL EXISTS: $repo/.env"
done
echo "Check complete."
```

Expected: only "Check complete." line — no `STILL EXISTS` lines.

**Step 4: No commit** — `.env` files are gitignored, so nothing to commit.

---

## Task 9 (P5): Smoke Test Everything Works Without `.env`

**Step 1: Verify shell env in a new terminal**

```bash
echo "POSTGRES_HOST=$POSTGRES_HOST"
echo "POSTGRES_PASSWORD=${POSTGRES_PASSWORD:0:6}..."
echo "KAFKA_BOOTSTRAP_SERVERS=$KAFKA_BOOTSTRAP_SERVERS"
echo "INFISICAL_ADDR=$INFISICAL_ADDR"
```

Expected:
```
POSTGRES_HOST=192.168.86.200
POSTGRES_PASSWORD=omniod...
KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29092
INFISICAL_ADDR=http://localhost:8880
```

**Step 2: Verify ConfigSessionStorage instantiates correctly**

```bash
cd /Volumes/PRO-G40/Code/omnibase_infra
uv run python3 -c "
from omnibase_infra.services.session.config_store import ConfigSessionStorage
cfg = ConfigSessionStorage()
print('host:', cfg.postgres_host)
print('database:', cfg.postgres_database)
print('dsn_safe:', cfg.dsn_safe)
"
```

Expected:
```
host: 192.168.86.200
database: omnibase_infra
dsn_safe: postgresql://postgres:***@192.168.86.200:5436/omnibase_infra
```

**Step 3: Run the full unit test suite**

```bash
uv run pytest tests/ -m unit -n auto 2>&1 | tail -5
```

Expected: all passing.

**Step 4: Real service boot smoke test**

Start one service that previously depended on repo `.env` and verify it connects:

```bash
cd /Volumes/PRO-G40/Code/omnibase_infra

# Explicitly pass the home-dir env file so compose doesn't silently fall back to
# a repo .env that no longer exists. If POSTGRES_* are in your shell env (via
# ~/.zshrc sourcing), this is belt-and-suspenders — but it proves compose works
# without a repo-local .env file.
docker compose \
    --env-file ~/.omnibase/.env \
    -p omnibase-infra-runtime \
    up -d runtime 2>&1 | tail -20

# Wait for startup
sleep 10

# Check Postgres and Kafka connectivity
docker logs omnibase-infra-runtime-runtime-1 2>&1 | grep -E "(Connected|ERROR|WARN)" | head -20
```

Expected: connection success lines, no credential errors.

If compose was previously reading from repo `.env` implicitly (via docker compose's automatic
`.env` file discovery), this step proves the explicit `--env-file` path works. Once confirmed,
update the `Makefile` or compose wrapper scripts to always pass `--env-file ~/.omnibase/.env`.

**Step 5: Final commit for this branch**

```bash
cd /Volumes/PRO-G40/Code/omnibase_infra
git add -u
git commit -m "feat(OMN-2287): zero repo .env files — shared vars via shell env, identity defaults in code"
```

---

## Summary of Changes

| File | Action | Priority | Why |
|------|--------|----------|-----|
| `scripts/validation/validate_clean_root.py` | Remove `.env` from allowed | P0 | Structural enforcement |
| `.pre-commit-config.yaml` | Add no-env-file hook | P0 | Commit-time enforcement |
| `src/.../services/session/config_store.py` | Remove env prefix | P1 | Reads standard `POSTGRES_*` |
| `config/shared_key_registry.yaml` | Create | P2 | Stable versioned source of truth |
| `~/.omnibase/scripts/onboard-repo.py` | Read registry | P3 | Deterministic key stripping |
| `scripts/register-repo.py` | Read registry | P3 | Registry-driven Infisical seeding |
| `~/.omnibase/.env` | Expand (transitional) | P4 | Shell env provides all shared vars |
| All repo `.env` files | Delete (allowlisted) | P4 | The goal — zero drift across clones |

## What Stays the Same

- `.env.example` stays (documents bootstrap vars for new developers)
- `docs/env-example-full.txt` stays (full legacy reference)
- `INFISICAL_ENCRYPTION_KEY` / `INFISICAL_AUTH_SECRET` are Infisical container service secrets —
  they live in `~/.omnibase/.env` (home dir) or deployment secrets, **never in any repo `.env`**
- `POSTGRES_DATABASE` stays as a `default=` in `ConfigSessionStorage` — never an env var

## Guardrail: omnidash DB Access

omnidash must not query upstream Postgres directly. It uses event bus + projections. If an
`omnidash_analytics` read-model DB is ever added, it must be an explicit projection persistence
implementation — not a "convenient" connection using globally-available credentials. Enforcement:
omnidash's Settings class must not include `postgres_host` or `postgres_dsn` unless it is for
its own read model, and that must be documented at the field level.
