# Handoff: OMN-2287 Phase 2 — Zero Repo `.env` Files (Local Review Cycle)

**Date:** 2026-02-23
**Author:** Jonah (automated local review, 23 iterations)

---

## Branch & Status

| Field | Value |
|-------|-------|
| **Branch** | `jonah/omn-2287-zero-repo-env-phase2` |
| **Base** | `main` |
| **Current commit** | `4169e51d` |
| **Total commits on branch** | 43 (2 foundational + 41 review fixes) |
| **Pre-commit hooks** | Passing on every commit |

**What this branch does:** Implements the registry-driven, zero-repo-env architecture from `docs/plans/2026-02-21-zero-repo-env-files.md`. Specifically: removes the `OMNIBASE_INFRA_SESSION_STORAGE_` env prefix from `ConfigSessionStorage`, creates `config/shared_key_registry.yaml` as the authoritative key list, rewrites `scripts/register-repo.py` to read the registry, adds `QUERY_TIMEOUT_SECONDS` and `CONSUL_ENABLED` to the transport map, adds valkey/qdrant keys to the registry, and sets runtime service containers to use `${INFISICAL_ADDR:-}` (opt-in empty default) in `docker-compose.infra.yml`.

---

## Work Completed This Session

A local review cycle (23 iterations, 41 `fix(review):` commits) was run against the two foundational commits. Every fix was committed individually with the pre-commit suite passing.

### Foundational Commits

| Commit | Description |
|--------|-------------|
| `849bea36` | `fix(session): remove env prefix so ConfigSessionStorage reads standard POSTGRES_* vars` |
| `3ba75581` | `feat(OMN-2287): zero-repo-env phase 2 — registry YAML, config_store prefix, compose fix` |

### Review Fix Commits (41 total, oldest first)

| Commit | Priority | Description |
|--------|----------|-------------|
| `a983f9d9` | major/minor | restore INFISICAL_ADDR opt-in, add registry guards, fix stale comment |
| `31471986` | major/minor | move KAFKA_BOOTSTRAP_SERVERS to bootstrap_only, registry guards, refactor |
| `aee79700` | minor | add QDRANT keys to registry, fix `_read_registry_data` docstring, clean up path |
| `e2d49bf6` | major/minor | registry data guards, stale comments, subparser formatter |
| `89c04c92` | minor | utf-8 encoding on open, docstring clarity, header semantics note |
| `4fb5b43a` | minor | docstring cleanup, POSTGRES_DSN comment, DSN encoding caveat |
| `0a461626` | minor | add URL-encoding caveat to OMNIINTELLIGENCE_DB_URL in compose |
| `dd599406` | minor | add migration note for KAFKA_BOOTSTRAP_SERVERS move to bootstrap_only |
| `fc245390` | minor | cmd_seed_shared identity_defaults guard, TODO migration comment |
| `3d9204f8` | major/minor | ticket ref typo, ambient-env note, registry type guards, compose version hint |
| `1c328acb` | minor | ValueError consistency, registry section guards, deduplicate registry_path |
| `1e37ecf3` | minor | string element validation, `_REGISTRY_PATH` constant, compose warning specificity |
| `cdb4e9d5` | minor | isinstance list guard before `all()` in `_bootstrap_keys`/`_identity_defaults` |
| `45f6ae65` | minor | POSTGRES_POOL_*_SIZE names, remove dead KAFKA_HOST_SERVERS, plan doc completion note |
| `a8b23cf3` | major/minor | update test assertions for POSTGRES_POOL_*_SIZE, remove KAFKA_BOOTSTRAP_SERVERS from transport map |
| `2de258af` | major/minor | pool env aliases, dead code, QDRANT_URL, docstring clarity |
| `f695f7d4` | major/minor | VAULT_TOKEN per-service, single-read registry, missing test assertion, warning clarity |
| `28d6cf69` | major/minor | VAULT_TOKEN intent docs, bootstrap subcategories, empty-list guard, pool alias test |
| `ff8c7d0d` | minor | annotate `yaml.safe_load` return in `_read_registry_data` |
| `0ebc9237` | **major** | add `populate_by_name=True` for AliasChoices mypy compliance |
| `0d65798b` | minor | clear ambient POSTGRES_* vars in pool defaults test |
| `4a2e8445` | major/minor | test env isolation, valkey keys, QUERY_TIMEOUT_SECONDS, compose notes |
| `71a9082b` | **major** | mypy type annotations in register-repo.py, YAML comment style |
| `20765a04` | major/minor | narrow `_upsert_secret` exception, cast return type, remove vacuous encoding, add missing ValidationError test |
| `d6661cb2` | major/minor | narrow auth indicators, fix test kwarg, QUERY_TIMEOUT_SECONDS comment, valkey ordering |
| `b20d0b48` | major/minor | remove broad auth substring, move `_AUTH_INDICATORS` to module-level, KAFKA_HOST_SERVERS comment, remove dead test setenv |
| `989634f8` | major/minor | sync cause-chain not-found patterns, drop HTTP 400 from folder-create suppress set, test env isolation |
| `e0d8c1ba` | major/minor | document HTTP 400 intent, soften adapter claim, add KAFKA_GROUP_ID and KAFKA_ACKS to registry |
| `2e27f937` | major/minor | URL scheme validation in `_load_infisical_adapter`, CONSUL_ENABLED in transport map, test env cleanup, update docstring |
| `14748aef` | **major** | move INFISICAL_ADDR validation to command entry points, remove ValueError from adapter factory |
| `ded6f61f` | minor | document VAULT_ADDR prefetch gap, clarify `_load_infisical_adapter` validation contract |
| `b238d01e` | minor | move onboard-repo INFISICAL_ADDR validation after dry-run gate, add QUERY_TIMEOUT_SECONDS to DATABASE transport keys |
| `519886ad` | major/minor | explicit auth re-raise before not-found logic in `_upsert_secret`, clarify QUERY_TIMEOUT_SECONDS comment |
| `0c106d1e` | major/minor | remove redundant secret-not-found check, add QDRANT_URL to registry and transport map, migration re-seed note |
| `39d9d4a3` | **major** | add INFISICAL_PROJECT_ID pre-flight check to `cmd_seed_shared` |
| `2958a71f` | minor | add QDRANT_URL transport map test, QUERY_TIMEOUT_SECONDS env var resolution test |
| `9ffa30cb` | minor | add empty-list guards to `_bootstrap_keys` and `_identity_defaults` |
| `3a794e89` | minor | print SystemExit string messages to stderr before returning 1 in command handlers |
| `f030c60e` | minor | swap guard ordering in `_bootstrap_keys`/`_identity_defaults` — empty-list check before element-type check |
| `4169e51d` | minor | simplify SystemExit handler, clarify `_upsert_secret` auth guard comment |

---

## Current State of Changed Files

| File | Lines +/- | What changed |
|------|-----------|--------------|
| `config/shared_key_registry.yaml` | +133 | **New file.** Authoritative YAML definition of all 39 shared keys across 8 Infisical folders, plus `bootstrap_only` and `identity_defaults` sections. |
| `docker/docker-compose.infra.yml` | +33/-0 | Runtime service containers use `${INFISICAL_ADDR:-}` (empty default — Infisical is opt-in); operators set `INFISICAL_ADDR` in `~/.omnibase/.env` to enable Infisical at runtime. Adds clarifying comments on URL-encoding requirements for DSN variables. |
| `docs/plans/2026-02-21-zero-repo-env-files.md` | +35/-0 | Status updates: Task 3 marked COMPLETED, migration re-seed note added, plan doc completion note added. |
| `scripts/register-repo.py` | +412/-182 | Rewrote to read `shared_key_registry.yaml` instead of a hardcoded dict; added `_load_registry()`, `_bootstrap_keys()`, `_identity_defaults()`; added INFISICAL_PROJECT_ID pre-flight check; narrowed exception handling; added mypy type annotations; refactored `cmd_seed_shared` and `cmd_onboard_repo`. |
| `src/.../runtime/config_discovery/transport_config_map.py` | +8/-1 | Added `CONSUL_ENABLED` to CONSUL keys, `QUERY_TIMEOUT_SECONDS` to DATABASE keys, `QDRANT_URL` to QDRANT keys, `KAFKA_GROUP_ID`/`KAFKA_ACKS` to KAFKA keys; removed `KAFKA_BOOTSTRAP_SERVERS` (bootstrap-only, not Infisical-sourced). |
| `src/.../services/session/config_store.py` | +79/-0 | Removed `OMNIBASE_INFRA_SESSION_STORAGE_` prefix; set `env_prefix=""`, `env_file=None`; added `populate_by_name=True`; added `AliasChoices` for `pool_min_size`/`pool_max_size` to map `POSTGRES_POOL_MIN_SIZE`/`POSTGRES_POOL_MAX_SIZE`; added `pool_sizes` cross-field validator; updated all docstrings. |
| `src/.../runtime/config_discovery/test_transport_config_map.py` | +18/-1 | Added tests for `CONSUL_ENABLED`, `QDRANT_URL`; updated assertions for `QUERY_TIMEOUT_SECONDS`; cleaned up test env isolation. |
| `tests/unit/services/session/__init__.py` | +3 | New file to make session test directory a Python package. |
| `tests/unit/services/session/test_config_store.py` | +120 | **New file.** 120-line test module: instantiation from env vars, default values, `dsn_safe` masking, `pool_sizes` validator, `AliasChoices` resolution for pool fields, ambient POSTGRES_* env isolation. |

---

## What Was Fixed (by Category)

### Bugs Fixed

- **Auth downgrade logic in `_upsert_secret`**: cause-chain inspection was not re-raising auth errors before falling through to the not-found branch, silently swallowing real auth failures as "secret not found".
- **Redundant not-found check**: a second not-found check after the auth branch was unreachable dead code.
- **Broad auth substring matching**: using a wide string like `"authentication"` matched too many unrelated errors; replaced with `_AUTH_INDICATORS` module-level constant with narrow, exact strings.
- **INFISICAL_ADDR validation placement**: was inside `_load_infisical_adapter()` (the factory), which meant a `ValueError` could escape from deep inside `cmd_onboard_repo` with no context. Moved validation to the command entry points.
- **HTTP 400 in folder-create suppress set**: the `_create_folder()` helper was suppressing HTTP 400 (bad request) alongside HTTP 409 (conflict/already-exists). 400 is a real error that should propagate; only 409 should be suppressed.
- **Missing INFISICAL_PROJECT_ID pre-flight**: `cmd_seed_shared` would proceed until it hit the Infisical API and fail with a confusing error if `INFISICAL_PROJECT_ID` was unset. Added a pre-flight check at command entry.
- **`populate_by_name=True` missing**: `AliasChoices` on pool fields failed mypy `[pydantic-alias]` without this setting. Mypy-compliant now.

### Type Safety Improvements

- Annotated all function return types in `register-repo.py` (mypy was reporting `Any` returns).
- Annotated `yaml.safe_load()` return as `dict[str, object]` in `_read_registry_data`.
- Narrowed `_upsert_secret` exception type from broad `Exception` to specific Infisical SDK exceptions.
- Removed vacuous `str(e)` encoding (already a str; cast is a no-op that confused mypy).
- Added explicit `cast()` on return value where `infisical-sdk` returns `Any`.

### Registry Completeness

- Added `KAFKA_BOOTSTRAP_SERVERS` to `bootstrap_only` section (it cannot come from Infisical — circular startup dep).
- Added `QDRANT_URL` to `config/shared_key_registry.yaml` (alongside existing `QDRANT_HOST`/`QDRANT_PORT`/`QDRANT_API_KEY`).
- Added `KAFKA_GROUP_ID` and `KAFKA_ACKS` to registry (per-service Kafka consumer config).
- Moved `KAFKA_HOST_SERVERS` to `bootstrap_only` — it is a host-scripts alias for `KAFKA_BOOTSTRAP_SERVERS`, not an Infisical-managed key.
- Added `CONSUL_ENABLED` to registry and to `transport_config_map.py`.
- Added `QUERY_TIMEOUT_SECONDS` to `transport_config_map.py` DATABASE keys (it is a per-service pool setting, distinct from the shared `POSTGRES_TIMEOUT_MS`).
- Added valkey ordering fix: `VALKEY_HOST`, `VALKEY_PORT`, `VALKEY_DB` are in the registry alongside `VALKEY_PASSWORD`.
- `VAULT_TOKEN` explicitly excluded from `/shared/vault/` — documented as per-service only in both the registry and plan doc.

### Test Coverage Additions

- New test file: `tests/unit/services/session/test_config_store.py` (120 lines).
  - `ConfigSessionStorage` instantiation from env vars
  - Default values
  - `dsn_safe` password masking
  - `pool_min_size > pool_max_size` cross-field validator
  - `AliasChoices` resolution for `POSTGRES_POOL_MIN_SIZE`/`POSTGRES_POOL_MAX_SIZE`
  - Ambient POSTGRES_* env isolation (monkeypatch clears ambient vars)
- Added `QDRANT_URL` test case to `test_transport_config_map.py`.
- Added `QUERY_TIMEOUT_SECONDS` env var resolution test.
- Fixed test kwarg mismatch in `cmd_onboard_repo` test.
- Added missing `ValidationError` test for pool cross-field validator.
- Added `pool_min > pool_max` validator test.
- All tests isolate their env var mutations with `monkeypatch` (no ambient `POSTGRES_*` leakage between tests).

### Security / Validation Guards

- Empty-list guard in `_bootstrap_keys()` and `_identity_defaults()`: if the registry section is missing or empty, return `frozenset()` without calling `all()` on an empty list (which would vacuously return `True` and skip a type check).
- List type check before element-type check (`isinstance(keys, list)` before `all(isinstance(k, str) for k in keys)`).
- URL scheme validation added to `_load_infisical_adapter`: `INFISICAL_ADDR` must start with `http://` or `https://`.
- `cmd_seed_shared` now validates `INFISICAL_PROJECT_ID` is set before making any API calls.
- SystemExit messages now go to stderr (not stdout) before returning exit code 1.

### Usability Improvements

- `register-repo.py` subparser now uses `RawDescriptionHelpFormatter` so multi-line help text is not reflowed.
- Stale comments updated throughout `register-repo.py`.
- Docstrings updated to reference `shared_key_registry.yaml` instead of the old hardcoded dict.
- Added `_REGISTRY_PATH` module-level constant (was duplicated as a local in multiple functions).
- Single-read of registry YAML per invocation (was read twice in some code paths).
- Plan doc updated: Task 3 status note added, migration re-seed note added, registry-growth caveat added.

---

## Pending Work (from the Plan)

| Priority | Status | Task |
|----------|--------|------|
| P0 | **NOT DONE** | Remove `.env` from `validate_clean_root.py` allowed files |
| P0 | **NOT DONE** | Add `no-env-file` pre-commit hook |
| P5 | **PENDING** | Smoke test: verify services boot without repo `.env` |
| — | **PENDING** | Delete repo `.env` (only after smoke test passes) |
| — | **PENDING** | Run `onboard-repo` for omniclaude, omniintelligence, omnidash |
| — | **PENDING** | Migrate `ConfigSessionStorage` to read `POSTGRES_DSN` from Infisical (OMN-2065) |
| — | **PENDING** | Add 13 unmapped handler types to `TransportConfigMap` |
| — | **PENDING** | Wire `ConfigPrefetcher.prefetch_for_contracts()` into actual startup path |
| — | **FUTURE** | Shrink `~/.omnibase/.env` back to 5 bootstrap lines (Task 7B — after all containers read Infisical directly) |

> **BLOCKING REQUIREMENT — P0 enforcement tasks must not slip to a follow-up PR.**
> Without these two tasks the zero-repo-env policy declared by this branch is active but completely unenforced: a developer can commit a `.env` file and no hook will catch it. Both tasks ("Remove `.env` from `validate_clean_root.py` allowed files" and "Add `no-env-file` pre-commit hook") MUST land on `main` before or concurrently with this branch merging. They are independent of this branch and have no runtime impact, so there is no technical reason to delay them. Do not merge this branch while either P0 task remains open.

**Notes on P0 tasks:** Tasks 1 and 2 from the plan (enforcement hooks) were scoped out of this branch. They are pure enforcement additions with no runtime impact and can land independently on `main` before or after this branch merges.

**On the 13 unmapped handler types:** `TransportConfigMap` currently maps 12 transport types. The following `EnumInfraTransportType` members have no entry and return an empty key tuple: `HTTP` (partial), `GRPC`, `MCP`, `FILESYSTEM`, `GRAPH`, `RUNTIME`, `INMEMORY`. Of these, `INMEMORY` and `RUNTIME` are intentionally empty. The others need real keys defined.

---

## Next Steps

In priority order:

1. **Open PR for this branch** — all hooks pass, 43 commits, ready for review.
   > **Note:** P0 enforcement tasks (adding the `no-env-file` pre-commit hook and removing `.env` from `validate_clean_root.py`) must land before or immediately after this branch merges to close the enforcement gap. Without P0, the zero-repo-env policy is active but unenforced — developers could commit a `.env` file and the hook would not catch it.
2. **P0 enforcement (separate PR)** — add `no-env-file` pre-commit hook and remove `.env` from `validate_clean_root.py`. These are independent of this branch and should not be blocked on it.
3. **Smoke test (P5)** — start the compose stack with `--env-file ~/.omnibase/.env` and no repo `.env`. Verify `ConfigSessionStorage` instantiates correctly from shell env. See `docs/plans/2026-02-21-zero-repo-env-files.md` Task 9 for the step-by-step.
4. **Delete repo `.env`** — only after smoke test passes. Run `~/.omnibase/scripts/onboard-repo.py` first (dry-run, then `--apply`) to strip shared keys from omniclaude, omniintelligence, omnidash.
5. **Re-seed Infisical** — the registry gained `QDRANT_URL`, `KAFKA_GROUP_ID`, `KAFKA_ACKS` after the initial seed run. Re-run `seed-shared --execute` to pick up the new keys. Command:
   ```bash
   set -a; source ~/.omnibase/.env; set +a
   uv run python scripts/register-repo.py seed-shared --execute
   ```
6. **Wire `ConfigPrefetcher`** — `prefetch_for_contracts()` is instantiated but never called with a real contracts directory during startup. This is the final step before removing the env-var fallback entirely.

---

## Known Intentional Decisions

These decisions were questioned during the review cycle and confirmed as deliberate. Saved here to avoid the same back-and-forth in code review.

| Decision | Rationale |
|----------|-----------|
| `extra="ignore"` on `ConfigSessionStorage` | Intentional. The class reads from the global shell env (`env_prefix=""`), which contains many unrelated vars (LLM endpoints, Slack tokens, etc.). `extra="forbid"` would reject every invocation. `extra="ignore"` is the correct posture when reading from a broad environment namespace. |
| `INFISICAL_ADDR` validation after dry-run gate | `cmd_onboard_repo` validates `INFISICAL_ADDR` only when `--execute` is set (after the dry-run gate). Dry-run should work without a live Infisical instance so developers can preview changes offline. Failing validation on dry-run would break this use case. |
| `VAULT_ADDR` in registry without a `VAULT` transport type | `VAULT_ADDR` is the shared platform address (the Vault server URL). It lives in `/shared/vault/`. `VAULT_TOKEN` is per-service and deliberately excluded from `/shared/` — each service needs its own scoped token. The registry comment documents this asymmetry. |
| HTTP 400 excluded from folder-create suppress set (409 only) | HTTP 409 (Conflict) means the folder already exists — safe to suppress. HTTP 400 (Bad Request) means the request was malformed — should propagate so the caller knows something is wrong with the input. These are distinct conditions with distinct meanings. |
| `KAFKA_BOOTSTRAP_SERVERS` in `bootstrap_only`, not `shared` | Containers need this key to start Kafka consumers, but the value differs between Docker-internal (`omninode-bridge-redpanda:9092`) and host scripts (`192.168.86.200:29092`). Seeding a single value to Infisical would be wrong for one of the two contexts. Lives in `~/.omnibase/.env` and compose env for now. |
| `QUERY_TIMEOUT_SECONDS` in both transport map and `shared_key_registry.yaml` | `QUERY_TIMEOUT_SECONDS` is in `/shared/db/` as a platform-wide default (seeded to Infisical as `/shared/db/QUERY_TIMEOUT_SECONDS`). Services that need a different value should set it per-service under `/services/<repo>/db/`. It is distinct from `POSTGRES_TIMEOUT_MS`, which is the connection-level timeout (milliseconds); `QUERY_TIMEOUT_SECONDS` is the query-level timeout (seconds). |
