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

`SecretResolutionError` includes structured context:

```python
except SecretResolutionError as e:
    print(e.context.transport_type)   # EnumInfraTransportType.RUNTIME
    print(e.context.operation)        # "get_secret"
    print(e.context.correlation_id)   # UUID for tracing
    print(e.logical_name)             # The requested logical name
```

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
# .secretresolver_allowlist
# Format: file_path:line_number:variable_name:ticket_reference

src/bootstrap.py:15:VAULT_TOKEN:OMN-764
src/legacy/adapter.py:42:LEGACY_API_KEY:OMN-999
```

**Note**: Allowlisted entries should include a ticket reference and be reviewed periodically.

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
