# Infisical Secrets Guide

> **Status**: Current | **Last Updated**: 2026-02-19

This guide covers how to retrieve secrets at runtime using `HandlerInfisical`, the path convention used in Infisical, the six-step bootstrap sequence for a fresh deployment, how to add a new secret to the seed script, and how to work locally without Infisical running.

For the architecture of config discovery (how contract YAML files drive automatic secret prefetch), see `docs/architecture/CONFIG_DISCOVERY.md`. This guide focuses on developer-facing tasks.

---

## Table of Contents

1. [Why Infisical](#why-infisical)
2. [Infisical Path Convention](#infisical-path-convention)
3. [Bootstrap Sequence](#bootstrap-sequence)
4. [Using HandlerInfisical at Runtime](#using-handlerinfisical-at-runtime)
5. [Fetching a Single Secret](#fetching-a-single-secret)
6. [Fetching Secrets in Batch](#fetching-secrets-in-batch)
7. [Synchronous Access via get_secret_sync](#synchronous-access-via-get_secret_sync)
8. [Handling SecretResolutionError](#handling-secretresolutionerror)
9. [Adding a New Secret to the Seed Script](#adding-a-new-secret-to-the-seed-script)
10. [Local Development Without Infisical](#local-development-without-infisical)
11. [Config Prefetch at Startup](#config-prefetch-at-startup)
12. [Cache Behavior](#cache-behavior)
13. [Common Mistakes](#common-mistakes)

---

## Why Infisical

Before Infisical, every service needed to list its connection strings and API keys in a `.env` file. As the node count grew, `.env` expanded from ~60 lines to over 660 lines. Adding a new node that needed Kafka and PostgreSQL meant updating `.env` for that service, and synchronizing that file across environments.

Infisical centralizes secrets in a structured store with:
- Audit logs on every secret access
- Per-environment values (dev, staging, production) without changing code
- Machine identity authentication (no shared passwords in `.env`)
- TTL-based caching in `HandlerInfisical` (default 5 minutes) to avoid hot paths hitting Infisical on every request
- Circuit breaking if Infisical is temporarily unavailable

The `.env` file is now reduced to ~30 bootstrap-only lines (Postgres password, Infisical credentials). Everything else comes from Infisical.

---

## Infisical Path Convention

All secrets follow a two-level hierarchy. The `<transport>` segment uses the **enum value** (lowercase), not the enum name.

```
Shared (one value for all services):
  /shared/<transport>/KEY

Per-service (can differ between services):
  /services/<service-slug>/<transport>/KEY
```

### Transport slugs

| Transport | Slug in path |
|-----------|-------------|
| PostgreSQL | `db` |
| Kafka/Redpanda | `kafka` |
| Consul | `consul` |
| Infisical itself | `infisical` (bootstrap transport — see below) |
| HTTP | `http` |
| MCP | `mcp` |
| Qdrant | `qdrant` |
| Runtime env vars | `env` |

### Examples

```
/shared/db/POSTGRES_DSN               -- shared connection string for all services
/shared/db/POSTGRES_POOL_MIN          -- shared pool minimum
/shared/db/POSTGRES_POOL_MAX          -- shared pool maximum
/shared/db/POSTGRES_TIMEOUT_MS        -- shared query timeout

/shared/kafka/KAFKA_BOOTSTRAP_SERVERS -- shared broker list

/shared/consul/CONSUL_HOST            -- shared Consul host
/shared/consul/CONSUL_PORT            -- shared Consul port

/shared/env/ENABLE_REAL_TIME_EVENTS   -- explicit env var dependency
/shared/env/NODE_SERVICE_NAME         -- service name flag

/services/omnibase-runtime/db/POSTGRES_DSN   -- runtime-specific override
/services/omniintelligence/db/POSTGRES_DSN   -- intelligence-specific pool config
```

Use per-service paths when a service needs a different value from the shared default — for example, a larger connection pool for a high-throughput service.

### Bootstrap transport exception

`INFISICAL` itself is a "bootstrap transport." Its credentials (`INFISICAL_ADDR`, `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`, `INFISICAL_PROJECT_ID`) must come from the `.env` file. They cannot come from Infisical because Infisical needs them to start. The config discovery system automatically skips `INFISICAL` when building fetch specs.

---

## Bootstrap Sequence

Run these six steps in order for a fresh deployment or when setting up a new environment. The `scripts/bootstrap-infisical.sh` script orchestrates all of them.

```
Step 1:  PostgreSQL starts
         POSTGRES_PASSWORD must be in .env (only credential that stays there)
         All other Postgres config will come from Infisical after seeding

Step 2:  Valkey starts
         Required by Infisical for session token storage
         No manual configuration needed

Step 3:  Infisical starts
         depends_on: postgres (healthy) + valkey (healthy)
         Reads from .env: INFISICAL_ADDR, INFISICAL_CLIENT_ID,
                          INFISICAL_CLIENT_SECRET, INFISICAL_PROJECT_ID

Step 4:  Identity provisioning (first time only)
         Run: scripts/setup-infisical-identity.sh
         Creates two machine identities:
           - runtime  (read-only: only fetches secrets)
           - admin    (read-write: used by seed script)

Step 5:  Seed Infisical from contracts
         Run: uv run python scripts/seed-infisical.py \
                  --contracts-dir src/omnibase_infra/nodes \
                  --import-env .env \
                  --set-values \
                  --execute
         Safe by default (--dry-run shows what would happen without --execute)
         Populates paths under /shared/ and /services/ based on contract transport types

Step 6:  Runtime services start
         Each service runs ConfigPrefetcher on startup
         Prefetched values are written to os.environ
         HandlerInfisical provides caching + circuit breaking
```

After step 6, the running services never read `.env` again (except the bootstrap credentials that stay there).

---

## Using HandlerInfisical at Runtime

`HandlerInfisical` wraps the Infisical SDK with caching and circuit breaking. It lives at `src/omnibase_infra/handlers/handler_infisical.py` and implements the standard handler lifecycle.

### Configuration

```python
infisical_config = {
    "host": "http://infisical:8080",        # INFISICAL_ADDR
    "client_id": "...",                      # INFISICAL_CLIENT_ID
    "client_secret": "...",                  # INFISICAL_CLIENT_SECRET
    "project_id": "...",                     # INFISICAL_PROJECT_ID
    "environment_slug": "dev",               # INFISICAL_ENVIRONMENT or "dev"
    "secret_path": "/shared/db",             # default path scope
    "cache_ttl_seconds": 300,                # 5-minute cache (default)
    "circuit_breaker_enabled": True,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_reset_timeout": 60.0,
}
```

### Initialize and shut down

```python
from omnibase_infra.handlers.handler_infisical import HandlerInfisical

handler = HandlerInfisical(container)
await handler.initialize(infisical_config)

# ... use the handler ...

await handler.shutdown()
```

---

## Fetching a Single Secret

Use the `infisical.get_secret` envelope operation to fetch a single secret by name.

```python
from uuid import uuid4

envelope = {
    "operation": "infisical.get_secret",
    "correlation_id": str(uuid4()),
    "payload": {
        "secret_name": "POSTGRES_DSN",
        # Optional overrides (uses handler defaults if omitted):
        # "project_id": "...",
        # "environment_slug": "prod",
        # "secret_path": "/services/my-service/db",
    },
}
output = await handler.execute(envelope)

# output.result["value"] is the raw secret string
# output.result["source"] is "cache" or "infisical"
dsn = output.result["value"]
```

The returned `value` is a plain string at this point (unwrapped from `SecretStr` for JSON-ledger compatibility). If you need a `SecretStr` at the call site, wrap it:

```python
from pydantic import SecretStr

dsn_secret = SecretStr(output.result["value"])
```

Secret values are never logged by the handler at any log level.

---

## Fetching Secrets in Batch

For startup prefetch where you need multiple secrets, `infisical.get_secrets_batch` is more efficient than calling `infisical.get_secret` in a loop.

```python
envelope = {
    "operation": "infisical.get_secrets_batch",
    "correlation_id": str(uuid4()),
    "payload": {
        "secret_names": [
            "POSTGRES_DSN",
            "POSTGRES_POOL_MIN",
            "POSTGRES_POOL_MAX",
            "POSTGRES_TIMEOUT_MS",
        ],
        # Optional path override:
        # "secret_path": "/services/my-service/db",
    },
}
output = await handler.execute(envelope)

secrets = output.result["secrets"]     # dict[str, str] — name -> value
errors  = output.result["errors"]      # dict[str, str] — name -> error message (if any)
from_cache = output.result["from_cache"]  # int — how many came from cache
from_fetch = output.result["from_fetch"]  # int — how many were fetched live
```

Cache hits are mixed with live fetches transparently. Any secret that was already cached will not incur a network call.

---

## Synchronous Access via get_secret_sync

Some callers cannot use `async/await` (for example, `ConfigPrefetcher` which calls into the handler from a synchronous context). `HandlerInfisical.get_secret_sync()` provides a synchronous interface backed by the same handler-level cache.

```python
from pydantic import SecretStr

value: SecretStr | None = handler.get_secret_sync(
    secret_name="POSTGRES_DSN",
    # Optional overrides:
    # project_id="...",
    # environment_slug="prod",
    # secret_path="/services/my-service/db",
)

if value is None:
    # Handler not initialized, or secret not found
    raise RuntimeError("Could not resolve POSTGRES_DSN")

dsn = value.get_secret_value()
```

`get_secret_sync` checks the cache first. It does not go through the circuit breaker (the circuit breaker is `async`-only). If Infisical is unreachable during a synchronous call, the SDK raises a `RuntimeError` which propagates to the caller.

---

## Handling SecretResolutionError

`SecretResolutionError` is raised when Infisical is reachable but the secret cannot be resolved (wrong name, wrong path, insufficient permissions). It is a subclass of `RuntimeHostError`.

```python
from omnibase_infra.errors import (
    InfraUnavailableError,
    SecretResolutionError,
    ModelInfraErrorContext,
)
from omnibase_infra.enums import EnumInfraTransportType

try:
    output = await handler.execute(envelope)
except InfraUnavailableError:
    # Circuit breaker open — Infisical is temporarily unavailable
    # Fall back to environment variable or fail fast depending on policy
    logger.warning("Infisical circuit open, falling back to env var")
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise
except SecretResolutionError as e:
    # Secret exists in Infisical but could not be fetched
    # Check: correct secret_name? correct secret_path? correct environment_slug?
    logger.error(
        "Secret resolution failed",
        extra={"correlation_id": str(e.context.correlation_id if e.context else "")},
    )
    raise
```

### Error sanitization rules

When logging or re-raising errors involving secret resolution, follow these rules:

Safe to include in error messages: operation names, Infisical path slugs (not values), correlation IDs.

Never include: secret names, secret values, raw paths that might reveal organizational structure.

```python
# WRONG — exposes the secret name and value
raise RuntimeHostError(f"Secret {secret_name}={value} not found")

# CORRECT — operation and correlation ID only
ctx = ModelInfraErrorContext.with_correlation(
    correlation_id=correlation_id,
    transport_type=EnumInfraTransportType.INFISICAL,
    operation="get_secret",
)
raise SecretResolutionError("Secret resolution failed", context=ctx) from e
```

---

## Adding a New Secret to the Seed Script

The seed script (`scripts/seed-infisical.py`) discovers which secrets are needed by scanning contract YAML files. When you add a new transport dependency to a node, the seed script picks it up automatically on the next run.

For explicit env var dependencies not tied to a transport type, add a `dependencies` entry to the node's `contract.yaml`:

```yaml
# In nodes/node_my_effect/contract.yaml
dependencies:
  - type: environment
    env_var: MY_FEATURE_FLAG
    required: true
  - type: environment
    env_var: OPTIONAL_TIMEOUT_MS
    required: false
```

These env vars will be seeded to `/shared/env/MY_FEATURE_FLAG` and `/shared/env/OPTIONAL_TIMEOUT_MS` in Infisical.

After adding the dependency, run the seed script in dry-run mode to preview the change:

```bash
# Preview what the seed script would do (dry-run is the default)
uv run python scripts/seed-infisical.py \
    --contracts-dir src/omnibase_infra/nodes

# Seed the new key (creates key, does not set value without --set-values)
uv run python scripts/seed-infisical.py \
    --contracts-dir src/omnibase_infra/nodes \
    --create-missing-keys \
    --execute

# Seed key and set value from .env
uv run python scripts/seed-infisical.py \
    --contracts-dir src/omnibase_infra/nodes \
    --import-env .env \
    --set-values \
    --execute
```

The seed script is safe by default. Without `--execute` it only prints what would happen. Without `--set-values` it creates keys with empty values. Without `--overwrite-existing` it never changes existing non-empty values.

For a per-service secret that differs from the shared default, set it manually in the Infisical UI at the path `/services/<service-slug>/<transport>/KEY`, or use the Infisical CLI:

```bash
infisical secrets set MY_KEY=my_value \
    --path /services/omnibase-runtime/db \
    --env dev
```

---

## Local Development Without Infisical

Config prefetch is **opt-in**. The `ConfigPrefetcher` and the runtime prefetch path only activate when `INFISICAL_ADDR` is set in the process environment.

When `INFISICAL_ADDR` is not set:
- The system skips prefetch entirely
- Each service reads configuration from the OS environment variables (loaded from `.env` as before)
- No error is raised; the behavior is identical to the pre-Infisical workflow

This means local development works without running Infisical. The full `.env` configuration reference (pre-Infisical, ~660 lines) is preserved at `docs/env-example-full.txt` for migration reference.

To test with Infisical locally:

```bash
# Bring up local Infisical alongside the infrastructure
docker compose --env-file .env -f docker/docker-compose.infra.yml up -d postgres valkey infisical

# Run the seed script
uv run python scripts/seed-infisical.py \
    --contracts-dir src/omnibase_infra/nodes \
    --import-env .env --set-values --execute

# Set INFISICAL_ADDR so the runtime activates prefetch
export INFISICAL_ADDR=http://localhost:8080
```

---

## Config Prefetch at Startup

`ConfigPrefetcher` orchestrates the startup prefetch. It is called once by the runtime before any nodes start handling requests.

```python
from pathlib import Path
from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
    ContractConfigExtractor,
)
from omnibase_infra.runtime.config_discovery.config_prefetcher import ConfigPrefetcher
from omnibase_infra.handlers.handler_infisical import HandlerInfisical

async def bootstrap_config(container, infisical_config: dict) -> None:
    # 1. Initialize HandlerInfisical
    handler = HandlerInfisical(container)
    await handler.initialize(infisical_config)

    # 2. Scan all node contracts for transport dependencies
    extractor = ContractConfigExtractor()
    requirements = extractor.extract_from_paths([
        Path("src/omnibase_infra/nodes/"),
    ])

    # 3. Prefetch matching secrets from Infisical
    prefetcher = ConfigPrefetcher(
        handler=handler,
        service_slug="omnibase-runtime",   # empty string -> use shared paths only
        infisical_required=False,           # True -> missing keys raise errors
    )
    result = prefetcher.prefetch(requirements)

    # 4. Apply to os.environ (existing env vars are never overwritten)
    applied = prefetcher.apply_to_environment(result)

    if result.missing:
        logger.warning("Keys missing from Infisical: %s", result.missing)

    logger.info(
        "Config prefetch complete",
        extra={
            "fetched": result.success_count,
            "missing": len(result.missing),
            "applied_to_env": applied,
        },
    )
```

The priority order for any config key is:

```
1. Existing os.environ value     (highest priority — never overwritten)
2. Value fetched from Infisical
3. Value in .env file            (lowest priority for Infisical-managed keys)
```

This means you can always override any Infisical value for local testing by setting the env var before the service starts.

---

## Cache Behavior

`HandlerInfisical` maintains a handler-level in-memory cache (not the Infisical SDK cache).

| Setting | Default | Description |
|---------|---------|-------------|
| `cache_ttl_seconds` | `300` | How long a cached secret is valid (5 minutes) |
| `MAX_CACHE_SIZE` | `1000` | Maximum cached entries before eviction |

Cache eviction strategy when `MAX_CACHE_SIZE` is exceeded:
1. Evict all expired entries.
2. If still over limit, evict the entries with the earliest expiry time until back under limit.

To force a cache miss (for example, after rotating a secret without restarting the service):

```python
# Invalidate a specific secret
count = handler.invalidate_cache(secret_name="POSTGRES_DSN")

# Invalidate everything
count = handler.invalidate_cache()
```

Cache metrics are available from `handler.describe()`:

```python
info = handler.describe()
print(info["cache_hits"])    # int
print(info["cache_misses"])  # int
print(info["total_fetches"]) # int
```

---

## Common Mistakes

### Logging secret values

```python
# WRONG — secret value ends up in log
logger.info(f"Using DSN: {output.result['value']}")

# CORRECT — log only metadata
logger.info("DSN resolved from Infisical", extra={"source": output.result["source"]})
```

### Using the wrong Infisical path

Fetching from `/shared/db/POSTGRES_DSN` returns the shared value. If a service needs a per-service override, it must use `/services/<slug>/db/POSTGRES_DSN`. The ConfigPrefetcher handles this via `service_slug`. If you call `HandlerInfisical.execute()` directly, pass the explicit `secret_path` in the payload.

### Not checking for None on get_secret_sync

`get_secret_sync` returns `None` when the handler is not initialized or the adapter is unavailable — it does not raise. Always check the return value:

```python
value = handler.get_secret_sync(secret_name="POSTGRES_DSN")
if value is None:
    raise RuntimeError("Handler not initialized or secret unavailable")
```

### Storing Infisical credentials in Infisical itself

`INFISICAL_ADDR`, `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`, and `INFISICAL_PROJECT_ID` must live in `.env`. They are "bootstrap credentials" — Infisical needs them to start. The seed script and `TransportConfigMap` both skip the `INFISICAL` transport to prevent circular bootstrap failures.

### Setting `infisical_required=True` prematurely

When `infisical_required=True`, any key missing from Infisical is recorded as an error in `ModelPrefetchResult.errors` and may block startup. Only set this after all secrets have been seeded. Use `infisical_required=False` (the default) during initial rollout or when the seed script has not yet run.

### Forgetting to call shutdown()

`HandlerInfisical.shutdown()` clears the cache, resets the circuit breaker, and releases the adapter connection. In tests, always call `await handler.shutdown()` in teardown to prevent state leaking between tests.

---

## See Also

- `docs/architecture/CONFIG_DISCOVERY.md` — how contracts drive automatic secret discovery
- `src/omnibase_infra/handlers/handler_infisical.py` — full handler implementation
- `src/omnibase_infra/runtime/config_discovery/` — ContractConfigExtractor, ConfigPrefetcher, TransportConfigMap
- `scripts/seed-infisical.py` — seed script with dry-run support
- `scripts/bootstrap-infisical.sh` — full bootstrap orchestration
- `scripts/setup-infisical-identity.sh` — machine identity creation
- `docs/env-example-full.txt` — pre-Infisical full `.env` reference
- `docs/patterns/security_patterns.md` — error sanitization and secret handling rules
