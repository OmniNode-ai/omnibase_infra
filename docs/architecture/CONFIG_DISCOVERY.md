# Config Discovery Architecture

> **Status**: Current | **Last Updated**: 2026-02-19

Config discovery is the system that reads ONEX contract YAML files at runtime, extracts which infrastructure transports each node depends on, and prefetches those configuration values from Infisical before the node starts handling requests. This eliminates the need for a sprawling `.env` file listing every config key across every service.

Introduced in OMN-2287 (version 0.10.0).

---

## Table of Contents

1. [Problem Solved](#problem-solved)
2. [Architecture Overview](#architecture-overview)
3. [The Three Contract Fields Scanned](#the-three-contract-fields-scanned)
4. [Component Reference](#component-reference)
5. [Infisical Path Convention](#infisical-path-convention)
6. [Bootstrap Sequence](#bootstrap-sequence)
7. [Opt-In Behavior](#opt-in-behavior)
8. [Bootstrap Transport Exception](#bootstrap-transport-exception)
9. [Usage Example](#usage-example)
10. [Environment Override Priority](#environment-override-priority)
11. [See Also](#see-also)

---

## Problem Solved

Before this system, every service that wanted to use, say, Kafka and PostgreSQL had to declare those connection strings explicitly in `.env`. Adding a new node that needed a database meant manually updating `.env` for that service. As the node count grew, `.env` expanded from ~60 lines to over 660 lines.

Config discovery inverts this: the contract file declares what a node needs (`transport_type: db` in metadata), and the runtime extracts that declaration and fetches the matching connection values from Infisical automatically.

---

## Architecture Overview

```
contract.yaml files (multiple, across node directories)
        |
        v
+---------------------------+
|  ContractConfigExtractor  |  -- scans YAML, reads 3 fields only
+---------------------------+
        |
        | ModelConfigRequirements
        v
+--------------------+
|  TransportConfigMap|  -- maps transport type -> Infisical path + key names
+--------------------+
        |
        | List[ModelTransportConfigSpec]
        v
+------------------+
|  ConfigPrefetcher|  -- fetches values from HandlerInfisical
+------------------+
        |
        | ModelPrefetchResult (resolved: dict[str, SecretStr], missing, errors)
        v
+--------------------+
|  os.environ        |  -- apply_to_environment() writes resolved values
+--------------------+
        |
        v
Runtime node starts with correct configuration
```

---

## The Three Contract Fields Scanned

`ContractConfigExtractor` is deliberately narrow. It only reads three specific Pydantic-backed fields from contract YAML — it does not parse arbitrary YAML sections.

### Field 1: `metadata.transport_type`

Declares the primary transport type for a node.

```yaml
# Example: a node that uses PostgreSQL as its primary transport
metadata:
  name: node_registry_reducer
  node_type: REDUCER_GENERIC
  transport_type: db          # <-- extracted here
```

When `transport_type: db` is found, the extractor looks up `EnumInfraTransportType.DATABASE` in `TransportConfigMap` and records that `POSTGRES_DSN`, `POSTGRES_POOL_MIN`, `POSTGRES_POOL_MAX`, and `POSTGRES_TIMEOUT_MS` are required by this contract.

### Field 2: `handler_routing.handlers[].handler_type`

Declares the transport type each individual handler uses. A node can have multiple handlers of different types (e.g., one that writes to PostgreSQL and another that reads from Consul).

```yaml
handler_routing:
  routing_strategy: operation_match
  handlers:
    - operation: register_node
      handler_type: consul          # <-- extracted here
    - operation: persist_node
      handler_type: db              # <-- and here
```

Each `handler_type` value is resolved to an enum and its canonical keys are added to the requirements. The source field path (e.g., `handler_routing.handlers[0].handler_type`) is recorded for traceability.

### Field 3: `dependencies[].type == "environment"`

For explicit env var declarations that don't fit a transport category — feature flags, service names, and similar:

```yaml
dependencies:
  - type: environment
    env_var: ENABLE_REAL_TIME_EVENTS
    required: true
  - type: environment
    env_var: NODE_SERVICE_NAME
    required: false
```

These are fetched from the `/shared/env/` Infisical path and tagged as `transport_type: RUNTIME` in the requirements model.

### String aliases supported

The extractor accepts both the canonical enum values and common aliases:

| Contract string | Resolved to |
|-----------------|-------------|
| `db`, `database`, `postgres`, `postgresql` | `DATABASE` |
| `kafka`, `redpanda` | `KAFKA` |
| `consul` | `CONSUL` |
| `redis`, `valkey` | `VALKEY` |
| `secret`, `secrets` | `INFISICAL` |
| `http` | `HTTP` |
| `mcp` | `MCP` |
| `qdrant` | `QDRANT` |
| `grpc` | `GRPC` |
| `filesystem` | `FILESYSTEM` |
| `graph` | `GRAPH` |

Unknown transport strings are recorded as non-fatal extraction errors (visible in `ModelConfigRequirements.errors`) and scanning continues.

---

## Component Reference

### `ContractConfigExtractor`

Stateless, thread-safe class. The primary entry point for extraction.

```python
extractor = ContractConfigExtractor()

# Single contract file
reqs = extractor.extract_from_yaml(Path("nodes/my_node/contract.yaml"))

# Recursive scan: pass directory -> finds all contract.yaml files under it
reqs = extractor.extract_from_paths([
    Path("src/omnibase_infra/nodes/"),
])

print(f"Found {len(reqs.requirements)} requirements from {len(reqs.contract_paths)} contracts")
print(f"Transport types: {[t.value for t in reqs.transport_types]}")
print(f"Extraction errors: {reqs.errors}")
```

`extract_from_paths` merges results from all scanned contracts using `ModelConfigRequirements.merge()`, which deduplicates transport types and contract paths while preserving insertion order.

### `TransportConfigMap`

Maps each `EnumInfraTransportType` to:
- The slug used in Infisical folder paths (the enum's `.value`, e.g., `"db"` for `DATABASE`)
- The canonical set of configuration key names expected at that path

```python
tcm = TransportConfigMap()

# Shared config: all services share these values
spec = tcm.shared_spec(EnumInfraTransportType.DATABASE)
# -> ModelTransportConfigSpec(
#      infisical_folder="/shared/db/",
#      keys=("POSTGRES_DSN", "POSTGRES_POOL_MIN", "POSTGRES_POOL_MAX", "POSTGRES_TIMEOUT_MS")
#    )

# Per-service config: values that differ between services
spec = tcm.service_spec(
    EnumInfraTransportType.DATABASE,
    service_slug="omnibase-runtime",
)
# -> ModelTransportConfigSpec(
#      infisical_folder="/services/omnibase-runtime/db/",
#      keys=("POSTGRES_DSN", ...)
#    )

# Batch: get specs for a list of transport types
specs = tcm.specs_for_transports(
    [EnumInfraTransportType.DATABASE, EnumInfraTransportType.KAFKA],
    service_slug="my-service",
)
```

Transports with empty key sets (`INMEMORY`, `RUNTIME`) are skipped automatically. Bootstrap transports (`INFISICAL`) are also skipped — they must come from the environment, not from Infisical itself.

### `ConfigPrefetcher`

Orchestrates the fetch from Infisical. Accepts any object implementing `ProtocolSecretResolver` (typically `HandlerInfisical`).

```python
from omnibase_infra.handlers.handler_infisical import HandlerInfisical

handler = HandlerInfisical(container)
await handler.initialize(infisical_config)

prefetcher = ConfigPrefetcher(
    handler=handler,
    service_slug="omnibase-runtime",   # empty -> use shared paths
    infisical_required=False,           # True -> missing keys are errors, not warnings
)

result = prefetcher.prefetch(requirements)
# result.resolved: dict[str, SecretStr]  -- successfully fetched values
# result.missing:  tuple[str, ...]       -- keys not found in Infisical
# result.errors:   dict[str, str]        -- per-key error messages (only when infisical_required=True)

# Apply to process environment (respects existing env values)
applied_count = prefetcher.apply_to_environment(result)
```

**Key ordering rule**: keys already present in `os.environ` are never overwritten. Environment variables always take precedence over Infisical values. This means you can override any Infisical value for local development simply by setting the corresponding env var.

### `ModelTransportConfigSpec`

Frozen Pydantic model representing a single fetch target in Infisical.

```python
ModelTransportConfigSpec(
    transport_type=EnumInfraTransportType.DATABASE,
    infisical_folder="/shared/db/",
    keys=("POSTGRES_DSN", "POSTGRES_POOL_MIN", "POSTGRES_POOL_MAX", "POSTGRES_TIMEOUT_MS"),
    required=False,
    service_slug="",
)
```

### `ModelConfigRequirements`

Frozen Pydantic model that aggregates requirements from one or more contracts. Supports merging:

```python
reqs_a = extractor.extract_from_yaml(path_a)
reqs_b = extractor.extract_from_yaml(path_b)
merged = reqs_a.merge(reqs_b)
# merged.requirements contains all from both
# merged.transport_types is deduplicated, order-preserved
```

### `ModelPrefetchResult`

Frozen Pydantic model returned by `ConfigPrefetcher.prefetch()`. Values are stored as `SecretStr` to prevent accidental logging.

```python
result.resolved        # dict[str, SecretStr]
result.missing         # tuple[str, ...]
result.errors          # dict[str, str]
result.specs_attempted # int -- number of transport specs attempted
result.success_count   # len(resolved)
result.failure_count   # len(missing) + len(errors)
```

### `ProtocolSecretResolver`

Protocol (not a class) defined in `runtime/config_discovery/models/protocol_secret_resolver.py`. Decouples `ConfigPrefetcher` from the `HandlerInfisical` import to avoid circular imports.

```python
class ProtocolSecretResolver(Protocol):
    def get_secret_sync(
        self, secret_name: str, secret_path: str
    ) -> SecretStr | None: ...
```

Any object with `get_secret_sync` satisfies this protocol — useful for testing with mock resolvers.

---

## Infisical Path Convention

All paths follow a two-level hierarchy. The `<transport>` segment is the **enum value** (e.g., `"db"` not `"DATABASE"`):

```
Shared (one value for all services):
  /shared/<transport>/KEY

Per-service (can differ between services):
  /services/<service-slug>/<transport>/KEY
```

Examples:

```
/shared/db/POSTGRES_DSN               -- shared connection string
/shared/kafka/KAFKA_BOOTSTRAP_SERVERS -- shared broker list
/shared/consul/CONSUL_HOST             -- shared Consul host
/shared/env/ENABLE_REAL_TIME_EVENTS   -- explicit env dependencies

/services/omnibase-runtime/db/POSTGRES_DSN   -- per-service override
/services/omniintelligence/db/POSTGRES_DSN   -- different pool for intelligence
```

Multiple instances of the same transport type (e.g., two PostgreSQL connections) are disambiguated via service namespacing.

---

## Bootstrap Sequence

The full bootstrap order for a fresh deployment or after environment changes:

```
Step 1:  PostgreSQL starts
         -- POSTGRES_PASSWORD must be in .env (not Infisical)

Step 2:  Valkey starts
         -- Required by Infisical for session storage

Step 3:  Infisical starts
         -- depends_on postgres + valkey healthy
         -- INFISICAL_ADDR, INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET in .env

Step 4:  Identity provisioning (first time only)
         -- scripts/setup-infisical-identity.sh
         -- Creates two identities:
            - runtime  (read-only)
            - admin    (read-write for seeding)

Step 5:  Seed runs
         -- scripts/seed-infisical.py
         -- Reads all contract.yaml files via ContractConfigExtractor
         -- Populates Infisical from discovered transport keys + .env values
         -- Safe by default (no --force needed); run with --dry-run to preview

Step 6:  Runtime services start
         -- Each service runs ConfigPrefetcher on startup
         -- Prefetched values are written to os.environ
         -- HandlerInfisical provides caching; circuit breaking; audit logging
```

---

## Opt-In Behavior

Config prefetch only activates when `INFISICAL_ADDR` is set in the process environment. Without it, the system silently skips prefetch and each service falls back to standard environment variable resolution.

This means:

- Local development without Infisical: works identically to the previous `.env`-based approach
- Production with Infisical: config is fetched from Infisical on startup
- CI/CD: can run without Infisical by simply omitting `INFISICAL_ADDR`

The `infisical_required` flag on `ConfigPrefetcher` controls behavior when a key is missing from Infisical:
- `False` (default): missing keys are logged as warnings and skipped
- `True`: missing required keys are recorded as errors in `ModelPrefetchResult.errors`

---

## Bootstrap Transport Exception

`INFISICAL` is a "bootstrap transport" — its credentials (`INFISICAL_ADDR`, `INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`, `INFISICAL_PROJECT_ID`) must come from the `.env` file. They cannot come from Infisical itself because Infisical needs them to start.

`TransportConfigMap.is_bootstrap_transport(EnumInfraTransportType.INFISICAL)` returns `True`. The `specs_for_transports` method automatically skips bootstrap transports when building fetch specs.

---

## Usage Example

```python
from pathlib import Path
from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
    ContractConfigExtractor,
)
from omnibase_infra.runtime.config_discovery.config_prefetcher import ConfigPrefetcher
from omnibase_infra.handlers.handler_infisical import HandlerInfisical

async def bootstrap_config(container, infisical_config: dict) -> None:
    # 1. Initialize Infisical handler
    handler = HandlerInfisical(container)
    await handler.initialize(infisical_config)

    # 2. Scan all node contracts
    extractor = ContractConfigExtractor()
    requirements = extractor.extract_from_paths([
        Path("src/omnibase_infra/nodes/"),
    ])

    # 3. Prefetch config values
    prefetcher = ConfigPrefetcher(
        handler=handler,
        service_slug="omnibase-runtime",
    )
    result = prefetcher.prefetch(requirements)

    # 4. Apply to environment (env vars already set are not overwritten)
    applied = prefetcher.apply_to_environment(result)

    print(f"Prefetched {result.success_count} keys, applied {applied} to environment")
    if result.missing:
        print(f"Missing from Infisical: {result.missing}")
```

---

## Environment Override Priority

The full resolution order for any config key:

```
1. os.environ (already set)  <-- highest priority, never overwritten
2. Infisical (prefetched by ConfigPrefetcher)
3. .env file (loaded before bootstrap)  <-- lowest priority for Infisical-managed keys
```

This means a developer can always override any value by setting the corresponding env var before the service starts, without modifying Infisical.

---

## See Also

- `src/omnibase_infra/runtime/config_discovery/` — all source files
- `scripts/bootstrap-infisical.sh` — full bootstrap orchestration script
- `scripts/seed-infisical.py` — populates Infisical from contracts and `.env`
- `scripts/setup-infisical-identity.sh` — creates machine identities
- `docs/env-example-full.txt` — pre-Infisical full `.env` reference (for migration)
- `docs/patterns/security_patterns.md` — error sanitization and secret handling
