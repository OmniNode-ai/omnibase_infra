# SecretResolver Pattern

## Overview

The `SecretResolver` provides centralized secret resolution for ONEX infrastructure. It abstracts secret retrieval from multiple sources (environment variables, file-based secrets, Vault) behind a unified interface with caching and convention-based fallback.

**Key Features**:
- Unified interface for multiple secret sources
- TTL-based caching with automatic expiration
- Thread-safe and async-safe operations
- Convention fallback when no explicit mapping exists
- Introspection methods that never expose secret values
- Pydantic `SecretStr` wrapping to prevent accidental logging

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
                source_type="vault",
                source_path="secret/data/kafka#sasl_password",
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
| **Bootstrap exceptions** | Vault token/addr always from env (two-phase initialization). |
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
| `vault.*` | `vault.token` | Vault bootstrap (env-only) |
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
| `.token` | Auth tokens | `vault.token` |
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
- Bootstrap secrets (Vault token/addr)

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

### Vault (`vault`)

HashiCorp Vault KV v2 secrets with optional field selection.

```python
# With field specifier
ModelSecretSourceSpec(
    source_type="vault",
    source_path="secret/data/database/postgres#password",  # path#field
)

# Without field (returns first field)
ModelSecretSourceSpec(
    source_type="vault",
    source_path="secret/data/api-keys",  # Returns first value
)
```

**Path format**: `path/to/secret#field_name`

**When to use**:
- Production secrets with rotation
- Centralized secret management
- Dynamic database credentials
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
| `vault` | 5 minutes (300s) | Supports secret rotation |

### Override TTL Per Secret

```python
ModelSecretMapping(
    logical_name="dynamic.api.key",
    source=ModelSecretSourceSpec(
        source_type="vault",
        source_path="secret/data/api-keys#rotating_key",
    ),
    ttl_seconds=60,  # 1 minute override
)
```

### Configure Default TTLs

```python
config = ModelSecretResolverConfig(
    default_ttl_env_seconds=86400,      # 24 hours
    default_ttl_file_seconds=86400,     # 24 hours
    default_ttl_vault_seconds=300,      # 5 minutes
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

Vault credentials must be available **before** the resolver can access Vault. This creates a two-phase initialization pattern.

### Phase 1: Bootstrap (env-only)

```python
import os

# These MUST come from environment, never from Vault
vault_token = os.environ.get("VAULT_TOKEN")
vault_addr = os.environ.get("VAULT_ADDR")
vault_ca_cert = os.environ.get("VAULT_CACERT")
```

### Phase 2: Initialize Vault Handler

```python
from omnibase_infra.handlers.handler_vault import HandlerVault

vault_handler = None
if vault_token and vault_addr:
    vault_handler = HandlerVault(
        vault_addr=vault_addr,
        vault_token=vault_token,
        ca_cert=vault_ca_cert,
    )
```

### Phase 3: Initialize Resolver

```python
resolver = SecretResolver(config=config, vault_handler=vault_handler)
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
        "vault.token",
        "vault.addr",
        "vault.ca_cert",
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

**Note**: For Vault secrets, async uses native async I/O. For env/file secrets, the sync call is wrapped in a thread executor.

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

Bootstrap secrets (needed to initialize Vault) are isolated from normal resolution:

```python
config = ModelSecretResolverConfig(
    bootstrap_secrets=["vault.token", "vault.addr", "vault.ca_cert"],
    # ... other settings
)
```

**Bootstrap secrets**:
- Always resolve from environment variables only
- Never routed to Vault (prevents circular dependency)
- Use convention-based naming: `vault.token` -> `VAULT_TOKEN`

### Vault Integration Security

Vault integration is currently a **stub implementation** that raises `NotImplementedError`:

```python
# When vault_handler is configured but secret is from Vault:
resolver.get_secret("vault.sourced.secret")
# Raises NotImplementedError with helpful message:
# "Vault secret resolution not yet implemented for logical name: vault.sourced.secret.
#  Configure this secret via 'env' or 'file' source until Vault integration is complete."
```

**When Vault handler is NOT configured**:
- Returns `None` (graceful degradation)
- Logs warning with logical name only (no path)

See [OMN-1374](https://linear.app/omninode/issue/OMN-1374) for the Vault integration implementation plan.

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
- Vault paths (which could reveal secret structure)

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
# - Vault paths
# - Secret values
# - Internal directory structure
```

### Best Practices

| Do | Do Not |
|----|--------|
| Use explicit mappings for sensitive secrets | Rely on convention fallback for critical secrets |
| Configure bootstrap secrets in `bootstrap_secrets` | Store Vault credentials in Vault |
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
#     source_type="vault",
#     source_path_masked="vault:secret/data/***",  # Path is masked
#     is_cached=True,
#     expires_at=datetime(2025, 1, 15, 12, 30, 0),
# )
```

### Masked Path Formats

| Source Type | Masked Format |
|-------------|---------------|
| `env` | `env:POSTGRES_PASSWORD` |
| `file` | `file:/run/secrets/***` |
| `vault` | `vault:secret/data/***` |

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
- Hard to migrate to Vault

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
- Easy migration to Vault
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

4. **Move to Vault** when ready (update mapping):
   ```python
   ModelSecretMapping(
       logical_name="database.postgres.password",
       source=ModelSecretSourceSpec(
           source_type="vault",
           source_path="secret/data/database/postgres#password",
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

from omnibase_infra.handlers.handler_vault import HandlerVault
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.models import (
    ModelSecretResolverConfig,
    ModelSecretMapping,
    ModelSecretSourceSpec,
)


def create_secret_resolver() -> SecretResolver:
    """Create and configure the secret resolver for production."""

    # Phase 1: Bootstrap Vault credentials (always from env)
    vault_token = os.environ.get("VAULT_TOKEN")
    vault_addr = os.environ.get("VAULT_ADDR", "https://vault.internal:8200")

    # Phase 2: Initialize Vault handler if available
    vault_handler = None
    if vault_token and vault_addr:
        vault_handler = HandlerVault(
            vault_addr=vault_addr,
            vault_token=vault_token,
        )

    # Phase 3: Configure resolver
    config = ModelSecretResolverConfig(
        mappings=[
            # Database credentials from Vault
            ModelSecretMapping(
                logical_name="database.postgres.password",
                source=ModelSecretSourceSpec(
                    source_type="vault",
                    source_path="secret/data/database/postgres#password",
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
        default_ttl_vault_seconds=300,   # 5 minutes
        default_ttl_env_seconds=86400,   # 24 hours
        default_ttl_file_seconds=3600,   # 1 hour (K8s may rotate)
    )

    return SecretResolver(config=config, vault_handler=vault_handler)


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

- Uses `Literal["env", "vault", "file"]` for `source_type` parameter
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

## Related Documentation

- [`docs/patterns/security_patterns.md`](./security_patterns.md) - Secret management guidelines, Vault integration
- [`docs/patterns/error_handling_patterns.md`](./error_handling_patterns.md) - Error context and sanitization
- [`src/omnibase_infra/handlers/handler_vault.py`](../../src/omnibase_infra/handlers/handler_vault.py) - Vault handler implementation
- [`src/omnibase_infra/errors/error_infra.py`](../../src/omnibase_infra/errors/error_infra.py) - `SecretResolutionError` definition

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
