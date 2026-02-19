> **Navigation**: [Home](../index.md) > [Decisions](README.md) > ADR-007 Vault to Infisical Migration

# ADR-007: Vault to Infisical Migration

## Status

Accepted

## Date

2026-02-19

## Context

The ONEX infrastructure originally used HashiCorp Vault (via the `hvac` Python
client) as the secret management backend. `HandlerVault` implemented secret
read, write, list, delete, and token-renewal operations against Vault's KV v2
secrets engine.

Several pressures made Vault's continued use unsustainable in the ONEX
development environment:

**Operational complexity.** Vault requires an external unsealing step after
every restart. For a development and CI environment where containers restart
frequently, this creates friction and manual intervention. Infisical runs
unsealed by default in Docker Compose deployments and exposes a simpler
bootstrap surface.

**Licensing trajectory.** HashiCorp relicensed Vault under the Business Source
License in 2023. While the BSL permits most internal use, the directional
uncertainty motivated evaluating purpose-built open-source alternatives.

**Developer experience.** Vault's path convention (`/secret/data/<path>`) and
token lifecycle management (renewal, TTL, token types) added ceremony around
every secret read. Infisical's model (project → environment → secret path)
maps more naturally to the `ONEX_ENV / ONEX_SERVICE` namespace model already
used in ONEX topic naming and service discovery.

**Bootstrap integration.** OMN-2287 added contract-driven config discovery that
scans ONEX node contract YAML files for transport type declarations and
pre-fetches the corresponding secrets from a central store. Infisical's REST
SDK integrates cleanly into this prefetch pattern. Vault's token-based
authentication required separate machinery to obtain and refresh credentials
before any secret could be read.

Work proceeded in three tickets:

- **OMN-2286**: `AdapterInfisical`, `HandlerInfisical`, and the config
  resolution layer added alongside the existing Vault handler.
- **OMN-2287**: Contract-driven config discovery, Infisical seed script, and
  bootstrap orchestration that populates Infisical from the existing `.env` at
  first launch.
- **OMN-2288**: `HandlerVault` removed; all secret resolution references updated
  to point at `HandlerInfisical`.

## Decision

**Replace HashiCorp Vault with Infisical as the sole secret management backend.**

The Vault handler (`HandlerVault`), its models, its error types (`InfraVaultError`,
`error_vault`), and the `MixinVaultInitialization` / `MixinVaultSecrets` /
`MixinVaultToken` / `MixinVaultRetry` mixin family are removed without a
compatibility shim. No forwarding stubs remain.

### What Infisical provides

- **Universal secrets management** via a project/environment/path hierarchy
- **Machine identity authentication** (client ID + client secret; no tokens
  to renew)
- **REST SDK** (`infisical-python`) as the adapter layer
- **TTL caching** at the handler level (default 5 minutes, configurable)
- **Batch secret fetch** via `infisical.get_secrets_batch` operation
- **Handler-level circuit breaker** via `MixinAsyncCircuitBreaker`

### Handler surface

`HandlerInfisical` supports three operations:

| Operation | Description |
|-----------|-------------|
| `infisical.get_secret` | Fetch a single secret by name |
| `infisical.list_secrets` | List secret keys at a path |
| `infisical.get_secrets_batch` | Fetch multiple secrets by name in one call |

The handler delegates raw SDK calls to `AdapterInfisical` (in
`adapters/_internal/`) and owns all cross-cutting concerns: caching, circuit
breaking, audit logging. Secret values are always returned as `SecretStr` at
the resolution boundary and are never logged.

### Path convention change

| System | Path format | Example |
|--------|-------------|---------|
| Vault | `/secret/data/<service>/<key>` | `/secret/data/omnibase-infra/POSTGRES_PASSWORD` |
| Infisical (shared) | `/shared/<transport>/KEY` | `/shared/database/POSTGRES_PASSWORD` |
| Infisical (per-service) | `/services/<service>/<transport>/KEY` | `/services/omnibase-infra/database/POSTGRES_PASSWORD` |

The Infisical path convention is defined by `TransportConfigMap` in
`runtime/config_discovery/transport_config_map.py`, keyed on
`EnumInfraTransportType` values.

### Error type changes

`InfraVaultError` (subclass of `InfraConnectionError`) is removed. The
`EnumInfraTransportType.VAULT` value is retired; errors from Infisical use
`EnumInfraTransportType.INFISICAL`. The `SecretResolutionError` class
(already in the error hierarchy) is reused as the canonical error for secret
fetch failures.

### Bootstrap sequence

The Infisical-backed bootstrap runs in this order:

1. PostgreSQL starts (password from `.env`)
2. Valkey (Redis-compatible cache) starts
3. Infisical starts (`depends_on: postgres + valkey healthy`)
4. Machine identity provisioning (first-time only, via
   `scripts/setup-infisical-identity.sh`)
5. Seed script populates Infisical from contracts and `.env` values
   (`scripts/seed-infisical.py`)
6. Runtime services start; `ConfigPrefetcher` prefetches required secrets
   from Infisical before node startup

The seed script is safe to re-run (idempotent). It supports `--dry-run` mode.

### Opt-in behavior

Config prefetch from Infisical is **opt-in**: `ConfigPrefetcher` only executes
when `INFISICAL_ADDR` is set in the environment. Without that variable, the
runtime falls back to standard environment variable resolution. Local
development that does not run Infisical continues to work without change.

## Consequences

### Positive

- **Simpler bootstrap**: Infisical starts unsealed; no post-restart manual
  steps.
- **Reduced `.env` surface**: The `.env.example` shrank from ~660 lines to
  ~30 bootstrap-only lines. Full pre-Infisical config preserved in
  `docs/env-example-full.txt`.
- **Machine identity auth**: No token renewal machinery required.
- **Contract-driven discovery**: Node contracts declare their transport
  dependencies; config prefetch resolves them before startup without bespoke
  per-node configuration.
- **Codebase simplification**: Four Vault mixins, one Vault handler, one
  Vault error class, and seven Vault model files are removed.

### Negative

- **New external dependency**: Infisical must be healthy before runtime
  services start. The opt-in guard (`INFISICAL_ADDR`) mitigates this for local
  development.
- **No secret write operations**: `HandlerInfisical` is read-only (get, list,
  batch-get). Write operations previously available in `HandlerVault`
  (write, delete) are not exposed. Administrative changes go through the
  Infisical web UI or seed script.
- **Loss of dynamic secret TTLs**: Vault's KV v2 supports per-secret versioning
  and TTLs at the engine level. Infisical's cache TTL is handler-managed and
  applies uniformly to all secrets.

### Neutral

- The `hvac` Python package is removed from dependencies.
- `EnumInfraTransportType.INFISICAL` is the new enum value for secret
  management transport errors.
- Existing consumers that relied on `InfraVaultError` must catch
  `SecretResolutionError` or `InfraConnectionError` instead.

## References

- **OMN-2286**: Infisical adapter, handler, and config resolution layer
- **OMN-2287**: Contract-driven config discovery, seed script, bootstrap
  orchestration
- **OMN-2288**: Remove Vault handler; migrate all secret resolution references
- **Handler**: `src/omnibase_infra/handlers/handler_infisical.py`
- **Adapter**: `src/omnibase_infra/adapters/_internal/adapter_infisical.py`
- **Config discovery**: `src/omnibase_infra/runtime/config_discovery/`
- **Bootstrap script**: `scripts/bootstrap-infisical.sh`
- **CLAUDE.md**: Contract-Driven Config Discovery section
