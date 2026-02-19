> **Navigation**: [Home](../index.md) > [Patterns](README.md) > Secret Resolver

# Secret Resolution Pattern

## Overview

ONEX infrastructure resolves secrets through two layers:

1. **`HandlerInfisical`** — the EFFECT-pattern handler that talks directly to the
   Infisical API. Owns caching, circuit breaking, and audit logging.
2. **`SecretResolver`** — a high-level runtime utility that wraps `HandlerInfisical`
   (and env/file sources) behind a unified logical-name API.

Most application code interacts with `SecretResolver`. `HandlerInfisical` is used
directly only when you need raw Infisical access (e.g., the seed script, bootstrap
tooling, or the runtime service kernel).

**Key Features**:
- Unified interface for multiple secret sources (env, file, Infisical)
- TTL-based caching with automatic expiration (handler-level and resolver-level)
- Thread-safe and async-safe operations
- Convention fallback when no explicit mapping exists
- Introspection methods that never expose secret values
- Pydantic `SecretStr` wrapping to prevent accidental logging
- Opt-in Infisical integration: set `INFISICAL_ADDR` to enable; omit for env-var fallback

**Core Principle**: Handlers should never call `os.getenv` directly for secrets. Use the resolver instead.

## Quick Start

```python
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.models import (
    ModelSecretResolverConfig,
    ModelSecretMapping,
    ModelSecretSourceSpec,
)

# Configure resolver with explicit mappings
config = ModelSecretResolverConfig(
    mappings=[
        ModelSecretMapping(
            logical_name="database.postgres.password",
            source=ModelSecretSourceSpec(
                source_type="env",
                source_path="POSTGRES_PASSWORD",
            ),
        ),
        ModelSecretMapping(
            logical_name="kafka.sasl.password",
            source=ModelSecretSourceSpec(
                source_type="infisical",
                source_path="/shared/kafka/KAFKA_SASL_PASSWORD",
            ),
        ),
    ],
)

# Initialize resolver
resolver = SecretResolver(config=config)

# Resolve secrets
password = resolver.get_secret("database.postgres.password")
api_key = resolver.get_secret("llm.openai.api_key", required=False)

# Access the actual value (when needed)
raw_password = password.get_secret_value()
```

## Design Philosophy

The SecretResolver follows these core principles:

| Principle | Description |
|-----------|-------------|
| **Dumb and deterministic** | Resolves and caches only. Does not discover, mutate, or manage secrets. |
| **Explicit mappings preferred** | Convention fallback is optional and disabled by default in strict mode. |
| **Bootstrap exceptions** | Infisical machine identity credentials always from env (two-phase initialization). |
| **Never exposes values** | Introspection returns masked paths only. |
| **Thread-safe** | Sync operations use `threading.Lock`, async uses `asyncio.Lock`. |

---

## Naming Conventions

### Canonical Rules

- **Lowercase, dot-separated**: `domain.subsystem.thing`
- **No hyphens, spaces, or camelCase**
- **Avoid "environment" in the name** (deployment concern, not identity)

### Recommended Domains

| Domain | Example | Description |
|--------|---------|-------------|
| `database.postgres.*` | `database.postgres.password` | PostgreSQL credentials |
| `database.postgres.*` | `database.postgres.username` | PostgreSQL users |
| `kafka.*` | `kafka.sasl.password` | Kafka/Redpanda authentication |
| `qdrant.*` | `qdrant.api_key` | Vector DB credentials |
| `infisical.*` | `infisical.client_id` | Infisical bootstrap (env-only) |
| `llm.openai.*` | `llm.openai.api_key` | LLM API keys |
| `llm.anthropic.*` | `llm.anthropic.api_key` | Claude API keys |
| `auth.keycloak.*` | `auth.keycloak.client_secret` | Auth provider secrets |
| `storage.s3.*` | `storage.s3.secret_access_key` | Object storage |
| `consul.*` | `consul.token` | Service discovery tokens |

### Suffix Conventions

| Suffix | Use For | Example |
|--------|---------|---------|
| `.password` | Database passwords | `database.postgres.password` |
| `.username` | Database users | `database.postgres.username` |
| `.token` | Auth tokens | `infisical.client_secret` |
| `.api_key` | API keys | `llm.openai.api_key` |
| `.dsn` | Connection strings | `database.postgres.dsn` |
| `.url` | Service URLs | `kafka.bootstrap.url` |
| `.private_key` | Private keys | `auth.jwt.private_key` |
| `.cert` | Certificates | `kafka.ssl.cert` |
| `.secret` | Generic secrets | `webhook.signing.secret` |

---

## Source Types

### Environment Variables (`env`)

The simplest source type. Maps logical names to environment variables.

```python
ModelSecretSourceSpec(
    source_type="env",
    source_path="POSTGRES_PASSWORD",  # env var name
)
```

**When to use**:
- Local development
- CI/CD pipelines
- Kubernetes deployments with env injection
- Bootstrap secrets (Infisical client ID/secret/addr)

### File-Based (`file`)

For Kubernetes secrets mounted as files at `/run/secrets`.

```python
# Absolute path
ModelSecretSourceSpec(
    source_type="file",
    source_path="/run/secrets/db_password",
)

# Relative path (resolved against secrets_dir)
ModelSecretSourceSpec(
    source_type="file",
    source_path="db_password",  # Resolves to /run/secrets/db_password
)
```

**Configuration**:
```python
config = ModelSecretResolverConfig(
    secrets_dir=Path("/run/secrets"),  # Default
    mappings=[...],
)
```

**When to use**:
- Kubernetes deployments with secret volume mounts
- Docker secrets
- Any file-based secret injection

### Infisical (`infisical`)

Infisical secret management with project-based path selection.

```python
# Per-service secret
ModelSecretSourceSpec(
    source_type="infisical",
    source_path="/services/omninode/database/POSTGRES_PASSWORD",
)

# Shared infrastructure secret
ModelSecretSourceSpec(
    source_type="infisical",
    source_path="/shared/database/POSTGRES_PASSWORD",
)
```

**Path convention**: `/shared/<transport>/KEY` or `/services/<service>/<transport>/KEY`

**When to use**:
- Production secrets with rotation
- Centralized secret management
- Any secrets requiring audit trails

---

## Convention Fallback

When no explicit mapping exists and `enable_convention_fallback=True`, the resolver automatically converts logical names to environment variable names.

### Conversion Rules

```python
# Without prefix
"database.postgres.password" -> "DATABASE_POSTGRES_PASSWORD"

# With prefix "ONEX_"
"database.postgres.password" -> "ONEX_DATABASE_POSTGRES_PASSWORD"
```

### Configuration

```python
config = ModelSecretResolverConfig(
    enable_convention_fallback=True,   # Enable auto-discovery
    convention_env_prefix="ONEX_",     # Optional prefix
    mappings=[],                       # No explicit mappings needed
)

resolver = SecretResolver(config=config)

# This will look for ONEX_DATABASE_POSTGRES_PASSWORD env var
password = resolver.get_secret("database.postgres.password")
```

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| Local development | Enable with `ONEX_` prefix |
| Production | Disable (prefer explicit mappings) |
| Testing | Enable for test fixtures |
| Migration | Enable temporarily while adding mappings |

---

## Caching

### Default TTLs

| Source | Default TTL | Rationale |
|--------|-------------|-----------|
| `env` | 24 hours (86400s) | Environment rarely changes during runtime |
| `file` | 24 hours (86400s) | File contents typically stable |
| `infisical` | 5 minutes (300s) | Supports secret rotation |

### Override TTL Per Secret

```python
ModelSecretMapping(
    logical_name="dynamic.api.key",
    source=ModelSecretSourceSpec(
        source_type="infisical",
        source_path="/shared/api/ROTATING_KEY",
    ),
    ttl_seconds=60,  # 1 minute override
)
```

### Configure Default TTLs

```python
config = ModelSecretResolverConfig(
    default_ttl_env_seconds=86400,           # 24 hours
    default_ttl_file_seconds=86400,          # 24 hours
    default_ttl_infisical_seconds=300,       # 5 minutes
    mappings=[...],
)
```

### Manual Refresh

```python
# Refresh single secret (invalidates cache)
resolver.refresh("database.postgres.password")

# Refresh all cached secrets
resolver.refresh_all()
```

### Cache Statistics

```python
stats = resolver.get_cache_stats()
print(f"Entries: {stats.total_entries}")
print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Refreshes: {stats.refreshes}")
print(f"Expired evictions: {stats.expired_evictions}")
```

---

## Bootstrap Flow

Infisical machine identity credentials must be available **before** the resolver can access Infisical. This creates a two-phase initialization pattern.

### Phase 1: Bootstrap (env-only)

```python
import os

# These MUST come from environment, never from Infisical
infisical_client_id = os.environ.get("INFISICAL_CLIENT_ID")
infisical_client_secret = os.environ.get("INFISICAL_CLIENT_SECRET")
infisical_addr = os.environ.get("INFISICAL_ADDR")
```

### Phase 2: Initialize Infisical Handler

```python
from omnibase_infra.handlers.handler_infisical import HandlerInfisical

infisical_handler = None
if infisical_client_id and infisical_client_secret and infisical_addr:
    infisical_handler = HandlerInfisical(
        infisical_addr=infisical_addr,
        client_id=infisical_client_id,
        client_secret=infisical_client_secret,
    )
```

### Phase 3: Initialize Resolver

```python
resolver = SecretResolver(config=config, infisical_handler=infisical_handler)
```

### Phase 4: All Other Secrets via Resolver

```python
# Now use the resolver for everything else
db_password = resolver.get_secret("database.postgres.password")
api_key = resolver.get_secret("llm.openai.api_key")
```

### Bootstrap Secrets Configuration

The resolver has a built-in list of bootstrap secrets that bypass resolution:

```python
config = ModelSecretResolverConfig(
    bootstrap_secrets=[
        "infisical.client_id",
        "infisical.client_secret",
        "infisical.addr",
    ],  # Default values
    mappings=[...],
)
```

---

## Async API

The resolver provides async variants for all primary methods.

```python
# Async single secret
password = await resolver.get_secret_async("database.postgres.password")

# Async multiple secrets
secrets = await resolver.get_secrets_async([
    "database.postgres.password",
    "kafka.sasl.password",
])
```

**Note**: For Infisical secrets, async uses native async I/O. For env/file secrets, the sync call is wrapped in a thread executor.

---

## Error Handling

```python
from omnibase_infra.errors import SecretResolutionError

# Required secret (raises on not found)
try:
    secret = resolver.get_secret("missing.secret")
except SecretResolutionError as e:
    # Handle missing required secret
    logger.error(
        "Secret not found",
        logical_name=e.logical_name,
        correlation_id=str(e.context.correlation_id),
    )

# Optional secret (returns None on not found)
secret = resolver.get_secret("optional.api.key", required=False)
if secret is None:
    # Use fallback behavior
    pass
```

### Error Context

`SecretResolutionError` includes structured context via the standard `ModelOnexError` interface:

```python
except SecretResolutionError as e:
    # Access context fields via error.model.context dict
    print(e.model.context.get("transport_type"))  # EnumInfraTransportType.RUNTIME
    print(e.model.context.get("operation"))       # "get_secret"
    print(e.model.context.get("logical_name"))    # The requested logical name

    # Correlation ID is at the model level
    print(e.model.correlation_id)                 # UUID for tracing
```

---

## Security Considerations

The SecretResolver implements multiple security controls to protect secrets:

### Path Traversal Prevention

File-based secrets are protected against path traversal attacks:

```python
# BLOCKED: Path traversal attempt
config = ModelSecretResolverConfig(
    mappings=[
        ModelSecretMapping(
            logical_name="attempted.traversal",
            source=ModelSecretSourceSpec(
                source_type="file",
                source_path="../../../etc/passwd",  # BLOCKED
            ),
        ),
    ],
    secrets_dir=Path("/run/secrets"),
)

resolver = SecretResolver(config=config)
# Returns None - path traversal detected and blocked
result = resolver.get_secret("attempted.traversal", required=False)
```

**How it works**:
- Relative paths MUST resolve within `secrets_dir`
- Absolute paths are trusted (explicitly configured by administrator)
- Symlink loops are handled gracefully (return None, not crash)

### Bootstrap Secret Isolation

Bootstrap secrets (needed to initialize Infisical) are isolated from normal resolution:

```python
config = ModelSecretResolverConfig(
    bootstrap_secrets=["infisical.client_id", "infisical.client_secret", "infisical.addr"],
    # ... other settings
)
```

**Bootstrap secrets**:
- Always resolve from environment variables only
- Never routed to Infisical (prevents circular dependency)
- Use convention-based naming: `infisical.client_id` -> `INFISICAL_CLIENT_ID`

### Infisical Integration Security

Infisical integration uses machine identity authentication (client ID + client secret). Credentials are loaded from environment variables during bootstrap and never stored in source files.

**When Infisical handler is NOT configured**:
- Returns `None` (graceful degradation)
- Logs warning with logical name only (no path)

**Path convention enforced**: Paths must match `/shared/<transport>/KEY` or `/services/<service>/<transport>/KEY`.

### Memory Handling

Raw secret values (plain strings) are briefly held in local variables during resolution
before being wrapped in `SecretStr`. Python's garbage collector will reclaim this memory,
but there is no explicit secure memory wiping. This is acceptable for most use cases.

**For high-security environments**:
- Consider using dedicated secret management libraries with secure memory handling
- Use short-lived processes for secret-intensive operations
- Ensure swap is encrypted at the OS level

The brief exposure window is minimized by:
- Immediately wrapping values in `SecretStr` after retrieval
- Never storing raw strings in instance attributes
- Cache stores only `SecretStr` values, not raw strings

### Log Sanitization

The SecretResolver never logs:
- Actual secret values (at any log level, including DEBUG)
- File paths to secret files
- Infisical paths (which could reveal secret structure)

**What IS logged (for debugging)**:
- Logical names (e.g., `database.postgres.password`)
- Error types (e.g., `PermissionError`, `FileNotFoundError`)
- Correlation IDs

### Error Message Security

Error messages are sanitized to prevent information leakage:

```python
# SecretResolutionError includes:
# - Logical name (for debugging)
# - Correlation ID (for tracing)
# - Error type
#
# SecretResolutionError does NOT include:
# - Actual file paths
# - Infisical paths
# - Secret values
# - Internal directory structure
```

### Best Practices

| Do | Do Not |
|----|--------|
| Use explicit mappings for sensitive secrets | Rely on convention fallback for critical secrets |
| Configure bootstrap secrets in `bootstrap_secrets` | Store Infisical credentials in Infisical |
| Use absolute paths only for trusted admin configs | Use relative paths with user-controlled input |
| Check return values for `required=False` calls | Assume secrets always exist |

---

## Introspection (Safe)

Introspection methods are designed to be safe for logging and debugging. They **never expose actual secret values**.

### List Configured Secrets

```python
# Returns logical names only (not values)
names = resolver.list_configured_secrets()
# ["database.postgres.password", "kafka.sasl.password", ...]
```

### Get Source Information

```python
info = resolver.get_source_info("database.postgres.password")

# ModelSecretSourceInfo(
#     logical_name="database.postgres.password",
#     source_type="infisical",
#     source_path_masked="infisical:/shared/***/***",  # Path is masked
#     is_cached=True,
#     expires_at=datetime(2025, 1, 15, 12, 30, 0),
# )
```

### Masked Path Formats

| Source Type | Masked Format |
|-------------|---------------|
| `env` | `env:POSTGRES_PASSWORD` |
| `file` | `file:/run/secrets/***` |
| `infisical` | `infisical:/shared/***/***` |

---

## Migration Guide

### Before (Direct `os.getenv`)

```python
# DON'T DO THIS - secrets scattered across codebase
import os

db_password = os.getenv("POSTGRES_PASSWORD")
api_key = os.getenv("OPENAI_API_KEY")
kafka_pass = os.getenv("KAFKA_SASL_PASSWORD")
```

**Problems**:
- No caching (env lookup on every call)
- No centralized configuration
- No audit trail
- Difficult to rotate secrets
- Hard to migrate to Infisical

### After (SecretResolver)

```python
# DO THIS - centralized secret management
password = resolver.get_secret("database.postgres.password")
api_key = resolver.get_secret("llm.openai.api_key")
kafka_pass = resolver.get_secret("kafka.sasl.password")
```

**Benefits**:
- Centralized configuration
- TTL-based caching
- Easy migration to Infisical
- Introspection without value exposure
- Thread-safe and async-safe

### Migration Steps

1. **Pick canonical dotted name** following naming conventions:
   ```
   POSTGRES_PASSWORD -> database.postgres.password
   OPENAI_API_KEY -> llm.openai.api_key
   ```

2. **Add explicit mapping** (temporary, points to env var):
   ```python
   ModelSecretMapping(
       logical_name="database.postgres.password",
       source=ModelSecretSourceSpec(
           source_type="env",
           source_path="POSTGRES_PASSWORD",
       ),
   )
   ```

3. **Update code** to use resolver:
   ```python
   # Old
   password = os.getenv("POSTGRES_PASSWORD")

   # New
   password = resolver.get_secret("database.postgres.password")
   ```

4. **Move to Infisical** when ready (update mapping):
   ```python
   ModelSecretMapping(
       logical_name="database.postgres.password",
       source=ModelSecretSourceSpec(
           source_type="infisical",
           source_path="/shared/database/POSTGRES_PASSWORD",
       ),
   )
   ```

5. **Delete env alias** - canonical name remains stable throughout

---

## CI Enforcement

New `os.getenv` calls in production paths will be flagged by CI. To migrate:

### Option 1: Use SecretResolver (Preferred)

```python
# Replace os.getenv with resolver
password = resolver.get_secret("database.postgres.password")
```

### Option 2: Allowlist (Temporary)

For bootstrap secrets or legacy code migration, add to `.secretresolver_allowlist`:

```text
# SecretResolver Migration Allowlist
# OMN-764: Centralized secret resolution
#
# Format: filepath:line_number # ticket reason
# Remove entries as files are migrated to use SecretResolver
#
# Bootstrap exceptions (permanent - not in this list):
# - src/omnibase_infra/runtime/secret_resolver.py (resolver needs bootstrap access)
#
# ==============================================================================
# Event Bus
# ==============================================================================
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:507 # OMN-764 KAFKA config from env

# ==============================================================================
# Runtime - Service Kernel
# ==============================================================================
src/omnibase_infra/runtime/service_kernel.py:157 # OMN-764 ONEX_CONTRACTS_DIR
src/omnibase_infra/runtime/service_kernel.py:630 # OMN-764 POSTGRES_HOST
src/omnibase_infra/runtime/service_kernel.py:635 # OMN-764 POSTGRES_USER

# ==============================================================================
# Nodes - Registration Orchestrator Plugin
# ==============================================================================
src/omnibase_infra/nodes/node_registration_orchestrator/plugin.py:214 # OMN-764 POSTGRES_HOST
```

**Allowlist Conventions**:
- Group entries by logical section using `# ===` delimiters
- Use relative paths from repository root
- Include ticket reference (e.g., `OMN-764`) and variable name as the reason
- Remove entries as files are migrated to use SecretResolver
- Bootstrap exceptions (e.g., `secret_resolver.py` itself) are hardcoded in the validator, not in the allowlist

### Option 3: Inline Exclusion Marker

For individual lines that cannot be migrated, use an inline comment marker:

```python
# This env var is documented in external API, cannot change
api_endpoint = os.getenv("EXTERNAL_API_URL")  # ONEX_EXCLUDE: secret_resolver
```

**Important**: The marker must appear in a comment context (after `#`). Markers inside string literals are ignored.

### CI Validator Edge Cases

The `validate_no_direct_env.py` validator has specific behaviors to be aware of:

| Behavior | Description |
|----------|-------------|
| **Alias detection** | Catches `from os import getenv` and `from os import environ` aliases |
| **Inline markers** | `# ONEX_EXCLUDE: secret_resolver` must be in actual comment, not string literal |
| **Bootstrap exceptions** | `secret_resolver.py` is permanently exempt (hardcoded, not allowlisted) |
| **Path matching** | Allowlist uses exact `filepath:line_number` format |
| **Exclusion patterns** | Test files (`test_*.py`, `*_test.py`, `conftest.py`) are automatically excluded |

**Exit Codes**:
- `0`: No violations found
- `1`: Violations found (blocks CI)
- `2`: Configuration error (malformed allowlist entry)

---

## Complete Example

```python
"""Example: Configuring SecretResolver for production."""

import os
from pathlib import Path

from omnibase_infra.handlers.handler_infisical import HandlerInfisical
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.models import (
    ModelSecretResolverConfig,
    ModelSecretMapping,
    ModelSecretSourceSpec,
)


def create_secret_resolver() -> SecretResolver:
    """Create and configure the secret resolver for production."""

    # Phase 1: Bootstrap Infisical credentials (always from env)
    infisical_client_id = os.environ.get("INFISICAL_CLIENT_ID")
    infisical_client_secret = os.environ.get("INFISICAL_CLIENT_SECRET")
    infisical_addr = os.environ.get("INFISICAL_ADDR", "http://192.168.86.200:8200")

    # Phase 2: Initialize Infisical handler if available
    infisical_handler = None
    if infisical_client_id and infisical_client_secret and infisical_addr:
        infisical_handler = HandlerInfisical(
            infisical_addr=infisical_addr,
            client_id=infisical_client_id,
            client_secret=infisical_client_secret,
        )

    # Phase 3: Configure resolver
    config = ModelSecretResolverConfig(
        mappings=[
            # Database credentials from Infisical
            ModelSecretMapping(
                logical_name="database.postgres.password",
                source=ModelSecretSourceSpec(
                    source_type="infisical",
                    source_path="/shared/database/POSTGRES_PASSWORD",
                ),
                ttl_seconds=300,  # 5 minutes for rotation support
            ),
            # Kafka from K8s secret mount
            ModelSecretMapping(
                logical_name="kafka.sasl.password",
                source=ModelSecretSourceSpec(
                    source_type="file",
                    source_path="kafka_password",
                ),
            ),
            # API key from env (local development)
            ModelSecretMapping(
                logical_name="llm.openai.api_key",
                source=ModelSecretSourceSpec(
                    source_type="env",
                    source_path="OPENAI_API_KEY",
                ),
            ),
        ],
        # Enable convention fallback for development
        enable_convention_fallback=os.environ.get("ONEX_ENV") == "development",
        convention_env_prefix="ONEX_",
        # K8s secrets mount point
        secrets_dir=Path("/run/secrets"),
        # Adjust TTLs for production
        default_ttl_infisical_seconds=300,   # 5 minutes
        default_ttl_env_seconds=86400,       # 24 hours
        default_ttl_file_seconds=3600,       # 1 hour (K8s may rotate)
    )

    return SecretResolver(config=config, infisical_handler=infisical_handler)


# Usage
resolver = create_secret_resolver()

# Get secrets
db_password = resolver.get_secret("database.postgres.password")
kafka_password = resolver.get_secret("kafka.sasl.password")
api_key = resolver.get_secret("llm.openai.api_key", required=False)

# Monitor cache performance
stats = resolver.get_cache_stats()
if stats.hit_rate < 0.8:
    logger.warning("Low cache hit rate", hit_rate=stats.hit_rate)
```

---

## ONEX Compliance

The SecretResolver follows ONEX infrastructure patterns:

### Error Handling

Uses the standard `omnibase_infra.errors` hierarchy:

```python
from omnibase_infra.errors import SecretResolutionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Create error context with automatic correlation ID
context = ModelInfraErrorContext.with_correlation(
    transport_type=EnumInfraTransportType.RUNTIME,
    operation="get_secret",
    target_name="secret_resolver",
)

raise SecretResolutionError(
    f"Secret not found: {logical_name}",
    context=context,
    logical_name=logical_name,
)
```

### Type Safety

- Uses `Literal["env", "infisical", "file"]` for `source_type` parameter
- `SecretSourceType` type alias exported for consistent usage across codebase
- `SecretStr` from Pydantic prevents accidental secret logging

### Thread Safety

- Sync operations protected by `threading.Lock`
- Async operations use `asyncio.Lock` for serialization
- No global state - all state encapsulated in resolver instance

### Not a Node

SecretResolver is a **runtime utility**, not an ONEX node. It does not require:
- `contract.yaml` (not a node)
- `ModelONEXContainer` injection (standalone utility)
- Handler routing (direct method calls)

This is intentional - the resolver is a low-level primitive used by nodes and handlers, not a workflow component.

---

## HandlerInfisical

`HandlerInfisical` is the EFFECT-category handler that performs raw secret operations
against the Infisical API. `SecretResolver` delegates to it for Infisical-backed secrets;
you can also call it directly for lower-level use.

**Module**: `omnibase_infra.handlers.handler_infisical`

### Architecture

```
SecretResolver
    └── HandlerInfisical          (EFFECT pattern, owns cache + circuit breaker)
            └── AdapterInfisical  (raw SDK calls only)
```

The handler owns all cross-cutting concerns: TTL caching (default 5 min, up to 1000
entries), circuit breaking (5 consecutive failures → OPEN, 60 s reset timeout), and
audit logging. The adapter is a thin SDK wrapper with no business logic.

### Initialization

`HandlerInfisical` takes an ONEX container and requires a call to `initialize()` before
use. Configuration is provided as a plain `dict` whose fields match
`ModelInfisicalHandlerConfig`:

```python
from omnibase_infra.handlers.handler_infisical import HandlerInfisical
from omnibase_core.container import ModelONEXContainer

handler = HandlerInfisical(container)
await handler.initialize({
    "host": "http://192.168.86.200:8880",   # INFISICAL_ADDR from .env
    "client_id": "...",                      # INFISICAL_CLIENT_ID
    "client_secret": "...",                  # INFISICAL_CLIENT_SECRET
    "project_id": "xxxxxxxx-xxxx-...",       # INFISICAL_PROJECT_ID
    "environment_slug": "prod",              # default: "prod"
    "secret_path": "/",                      # default: "/"
    "cache_ttl_seconds": 300.0,             # default: 5 minutes; 0 = disabled
    "circuit_breaker_threshold": 5,          # default: 5 failures
    "circuit_breaker_reset_timeout": 60.0,  # default: 60 seconds
    "circuit_breaker_enabled": True,         # default: True
})
```

### Supported Operations

All operations go through `execute(envelope)` using one of three operation strings:

| Operation string | Purpose |
|-----------------|---------|
| `infisical.get_secret` | Fetch a single secret by name |
| `infisical.list_secrets` | List secret keys at a path (no values) |
| `infisical.get_secrets_batch` | Fetch multiple secrets in one call |

### Get a Single Secret (async)

```python
from omnibase_infra.errors import SecretResolutionError

envelope = {
    "operation": "infisical.get_secret",
    "correlation_id": str(correlation_id),
    "payload": {
        "secret_name": "POSTGRES_PASSWORD",
        # Optional overrides (use handler defaults when omitted):
        # "project_id": "...",
        # "environment_slug": "prod",
        # "secret_path": "/shared/db",
    },
}

output = await handler.execute(envelope)
# output.result["value"]  -- plain str (re-wrap in SecretStr at call site)
# output.result["source"] -- "cache" or "infisical"
```

### List Secrets at a Path

```python
envelope = {
    "operation": "infisical.list_secrets",
    "correlation_id": str(correlation_id),
    "payload": {
        "secret_path": "/shared/db",
    },
}
output = await handler.execute(envelope)
# output.result["secret_keys"]  -- list[str], no values exposed
# output.result["count"]        -- int
```

### Batch Fetch

```python
envelope = {
    "operation": "infisical.get_secrets_batch",
    "correlation_id": str(correlation_id),
    "payload": {
        "secret_names": ["POSTGRES_PASSWORD", "KAFKA_SASL_PASSWORD"],
        "secret_path": "/shared",
    },
}
output = await handler.execute(envelope)
# output.result["secrets"]    -- dict[str, str]  (name -> plain value)
# output.result["errors"]     -- dict[str, str]  (name -> error message)
# output.result["from_cache"] -- int
# output.result["from_fetch"] -- int
```

### Synchronous Access

For callers that cannot use async (e.g., `SecretResolver._read_infisical_secret_sync`):

```python
from pydantic import SecretStr

value: SecretStr | None = handler.get_secret_sync(
    secret_name="POSTGRES_PASSWORD",
    project_id=None,        # uses handler default
    environment_slug=None,  # uses handler default
    secret_path=None,       # uses handler default
)
if value is not None:
    raw = value.get_secret_value()
```

Returns `None` when the handler is not initialized or the adapter is unavailable.

### Cache Invalidation

```python
# Invalidate a specific secret (matches on trailing "::secret_name")
count = handler.invalidate_cache("POSTGRES_PASSWORD")

# Invalidate all cached entries
count = handler.invalidate_cache()
```

### Introspection

```python
info = handler.describe()
# {
#   "handler_type": "INFRA_HANDLER",
#   "handler_category": "EFFECT",
#   "supported_operations": [...],
#   "cache_ttl_seconds": 300.0,
#   "initialized": True,
#   "cache_hits": 42,
#   "cache_misses": 7,
#   "total_fetches": 7,
#   "version": "0.9.0",
# }
# Note: credentials are NEVER exposed in describe() output.
```

### Shutdown

```python
await handler.shutdown()
# Clears cache, resets circuit breaker, releases adapter resources.
```

---

## SecretResolutionError

`SecretResolutionError` is raised when a required secret cannot be resolved—whether
from Infisical, env, or file.

```python
from omnibase_infra.errors import SecretResolutionError

try:
    output = await handler.execute(envelope)
except SecretResolutionError as e:
    # The error carries a ModelInfraErrorContext with correlation_id,
    # transport_type=INFISICAL, and operation name.
    # It does NOT include secret names, paths, or values.
    logger.error(
        "Secret resolution failed",
        extra={"correlation_id": str(e.context.correlation_id)},
    )
```

`SecretResolver` also raises `SecretResolutionError` for required secrets that are
missing from all configured sources:

```python
try:
    secret = resolver.get_secret("database.postgres.password")
except SecretResolutionError as e:
    print(e.model.context.get("logical_name"))  # the dotted name
    print(e.model.correlation_id)               # UUID for tracing
```

---

## Opt-In Infisical Behavior

Infisical integration is **opt-in** via the `INFISICAL_ADDR` environment variable.

| `INFISICAL_ADDR` set? | Behavior |
|----------------------|----------|
| No (local dev) | `HandlerInfisical` is not initialized. `SecretResolver` falls back to env/file sources only. Local development works without Infisical. |
| Yes (production) | `HandlerInfisical` initializes and prefetches secrets from Infisical at service startup. |

This means you can develop locally using just `.env` variables and enable centralized
secret management by adding `INFISICAL_ADDR` to the environment.

The runtime service kernel and config prefetcher respect this pattern:

```python
import os

infisical_addr = os.environ.get("INFISICAL_ADDR")
if infisical_addr:
    handler = HandlerInfisical(container)
    await handler.initialize({
        "host": infisical_addr,
        "client_id": os.environ["INFISICAL_CLIENT_ID"],
        "client_secret": os.environ["INFISICAL_CLIENT_SECRET"],
        "project_id": os.environ["INFISICAL_PROJECT_ID"],
    })
else:
    handler = None  # SecretResolver will use env fallback only
```

See [docs/guides/INFISICAL_SECRETS_GUIDE.md](../guides/INFISICAL_SECRETS_GUIDE.md)
for the full six-step bootstrap sequence.

---

## sanitize_secret_path

When constructing error messages or log entries that involve secret paths, use
`sanitize_secret_path` from `omnibase_infra.utils.util_error_sanitization`. It masks
sensitive path segments while preserving the mount point for debugging.

```python
from omnibase_infra.utils.util_error_sanitization import sanitize_secret_path

# Preserves first segment, masks the rest
sanitize_secret_path("secret/data/myapp/database/credentials")
# -> "secret/***/***"

sanitize_secret_path("kv/production/api-keys/stripe")
# -> "kv/***/***"

sanitize_secret_path("secret")   # single segment: returned as-is
# -> "secret"

sanitize_secret_path(None)        # None passthrough
# -> None

sanitize_secret_path("")          # empty: returned as-is
# -> ""
```

**What it protects against**:
- Exposure of application names and environments (`production`, `myapp`)
- Exposure of service or database names (`postgres`, `redis`)
- Exposure of credential types (`api-keys`, `certificates`)

For Infisical paths following the `/shared/<transport>/KEY` convention, the function
masks everything after the first segment:

```python
sanitize_secret_path("/shared/db/POSTGRES_PASSWORD")
# -> "/***/***/***"  (leading slash creates empty first segment)
```

Use `sanitize_secret_path` in any error context, log statement, or exception message
that references a secret path. Never include raw paths in errors or logs.

---

## Related Documentation

- [`docs/guides/INFISICAL_SECRETS_GUIDE.md`](../guides/INFISICAL_SECRETS_GUIDE.md) - Developer guide: fetching secrets, bootstrap sequence, local dev
- [`docs/patterns/security_patterns.md`](./security_patterns.md) - Secret management guidelines, Infisical integration
- [`docs/patterns/error_handling_patterns.md`](./error_handling_patterns.md) - Error context and sanitization
- [`src/omnibase_infra/handlers/handler_infisical.py`](../../src/omnibase_infra/handlers/handler_infisical.py) - HandlerInfisical implementation
- [`src/omnibase_infra/handlers/models/infisical/model_infisical_handler_config.py`](../../src/omnibase_infra/handlers/models/infisical/model_infisical_handler_config.py) - Configuration model
- [`src/omnibase_infra/errors/error_infra.py`](../../src/omnibase_infra/errors/error_infra.py) - `SecretResolutionError` definition
- [`src/omnibase_infra/utils/util_error_sanitization.py`](../../src/omnibase_infra/utils/util_error_sanitization.py) - `sanitize_secret_path` and related utilities

## API Reference

### SecretResolver

| Method | Description |
|--------|-------------|
| `get_secret(logical_name, required=True)` | Resolve secret synchronously |
| `get_secrets(logical_names, required=True)` | Resolve multiple secrets synchronously |
| `get_secret_async(logical_name, required=True)` | Resolve secret asynchronously |
| `get_secrets_async(logical_names, required=True)` | Resolve multiple secrets asynchronously |
| `refresh(logical_name)` | Invalidate cache for single secret |
| `refresh_all()` | Invalidate all cached secrets |
| `get_cache_stats()` | Get cache statistics |
| `list_configured_secrets()` | List configured logical names |
| `get_source_info(logical_name)` | Get masked source information |

### Configuration Models

| Model | Purpose |
|-------|---------|
| `ModelSecretResolverConfig` | Top-level resolver configuration |
| `ModelSecretMapping` | Maps logical name to source |
| `ModelSecretSourceSpec` | Defines source type and path |

### Cache Models

| Model | Purpose |
|-------|---------|
| `ModelCachedSecret` | Cached secret with TTL tracking |
| `ModelSecretCacheStats` | Cache statistics for observability |
| `ModelSecretSourceInfo` | Non-sensitive source info for introspection |
