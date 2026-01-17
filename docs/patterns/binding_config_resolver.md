# Handler Config Resolver Pattern

## Overview

The `BindingConfigResolver` provides a unified interface for resolving handler configurations from multiple sources with proper priority ordering. It abstracts configuration loading from environment variables, files, and Vault secrets, applying environment overrides and caching resolved configurations.

**Key Features**:
- Multi-source configuration resolution with priority ordering
- TTL-based caching with automatic expiration
- Thread-safe and async-safe operations
- Environment variable overrides (highest priority)
- Vault secret integration via SecretResolver
- Path traversal protection for file-based configs
- File size limits to prevent memory exhaustion

**Core Principle**: Handlers should never directly parse their own configuration from multiple sources. Use the resolver to normalize and validate configurations.

## Quick Start

```python
from pathlib import Path
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models import (
    ModelBindingConfigResolverConfig,
    ModelBindingConfig,
)

# Configure resolver
config = ModelBindingConfigResolverConfig(
    config_dir=Path("/etc/onex/handlers"),
    cache_ttl_seconds=300.0,
    env_prefix="HANDLER",
)

# Initialize resolver
resolver = BindingConfigResolver(config=config)

# Resolve from inline config
binding = resolver.resolve(
    handler_type="db",
    inline_config={"pool_size": 10, "timeout_ms": 5000}
)

# Resolve from file reference
binding = resolver.resolve(
    handler_type="vault",
    config_ref="file:configs/vault.yaml"
)

# Access binding configuration
print(f"Handler: {binding.handler_type}")
print(f"Timeout: {binding.timeout_ms}ms")
```

## Design Philosophy

The BindingConfigResolver follows these core principles:

| Principle | Description |
|-----------|-------------|
| **Dumb and deterministic** | Resolves and caches only. Does not discover, mutate, or manage configs. |
| **Environment overrides win** | Environment variables always take precedence for operational flexibility. |
| **Defense in depth** | Path traversal blocked, file size limited, schemes validated. |
| **Never exposes raw values** | Error messages are sanitized to exclude configuration content. |
| **Thread-safe** | Sync operations use `threading.Lock`, async uses per-key `asyncio.Lock`. |

---

## Problem Statement

Handler configuration in distributed systems often comes from multiple sources:
- Inline YAML in contract files (development defaults)
- External files (Kubernetes ConfigMaps)
- Environment variables (runtime overrides)
- Vault secrets (production credentials)

Without centralized resolution:
- Each handler implements its own config loading logic
- Priority ordering is inconsistent across handlers
- Environment overrides require custom code in each handler
- No unified caching strategy leads to repeated file/network I/O
- Security controls (path traversal, size limits) are inconsistently applied

The BindingConfigResolver solves these problems with a single resolution interface.

---

## Solution Architecture

### Resolution Priority

| Priority | Source | Description |
|----------|--------|-------------|
| 1 (Highest) | Environment Variables | Pattern: `{ENV_PREFIX}_{HANDLER_TYPE}_{FIELD}` |
| 2 | Vault Secrets | Via SecretResolver for `vault:` references in config values |
| 3 | File-Based Config | YAML/JSON files referenced via `config_ref` |
| 4 (Lowest) | Inline Config | Dictionary passed to `resolve()` method |

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BindingConfigResolver                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        Cache Layer                               │    │
│  │  ┌───────────────────────────────────────────────────────────┐  │    │
│  │  │  Handler Type -> _CacheEntry(config, expires_at, source)  │  │    │
│  │  └───────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  │                                       │
│                          Cache Miss                                      │
│                                  ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Config Reference Parser                       │    │
│  │                                                                  │    │
│  │  config_ref="file:configs/db.yaml"                              │    │
│  │       │                                                          │    │
│  │       ▼                                                          │    │
│  │  ModelConfigRef(scheme=FILE, path="configs/db.yaml")            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐     │
│  │  File Loader  │   │  Env Loader   │   │  Vault Loader         │     │
│  │               │   │               │   │  (via SecretResolver) │     │
│  │ YAML/JSON     │   │ JSON/YAML     │   │                       │     │
│  │ size limit    │   │ from env var  │   │ Logical name lookup   │     │
│  │ path check    │   │               │   │                       │     │
│  └───────┬───────┘   └───────┬───────┘   └───────────┬───────────┘     │
│          │                   │                       │                   │
│          └───────────────────┼───────────────────────┘                   │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        Config Merger                             │    │
│  │                                                                  │    │
│  │  1. Start with config from config_ref (if provided)             │    │
│  │  2. Overlay inline_config (takes precedence)                    │    │
│  │  3. Apply environment variable overrides (highest priority)     │    │
│  │  4. Resolve vault:// references in values                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        Validator                                 │    │
│  │                                                                  │    │
│  │  - Construct ModelRetryPolicy if dict provided                  │    │
│  │  - Validate against ModelBindingConfig schema            │    │
│  │  - Filter unknown fields if strict_validation=False             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│                  ModelBindingConfig                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### ModelBindingConfigResolverConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config_dir` | `Path \| None` | `None` | Base directory for relative `file://` paths |
| `enable_caching` | `bool` | `True` | Whether to cache resolved configurations |
| `cache_ttl_seconds` | `float` | `300.0` | TTL for cached configs (0-86400) |
| `env_prefix` | `str` | `"HANDLER"` | Prefix for environment variable overrides |
| `secret_resolver` | `object \| None` | `None` | SecretResolver for `vault:` references |
| `strict_validation` | `bool` | `True` | Fail on unknown fields if True |
| `allowed_schemes` | `frozenset[str]` | `{"file", "env", "vault"}` | Allowed config_ref schemes |

**Example Configuration**:

```python
from pathlib import Path
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.models import (
    ModelBindingConfigResolverConfig,
    ModelSecretResolverConfig,
)

# Create secret resolver for vault: references
secret_config = ModelSecretResolverConfig(
    enable_convention_fallback=True,
    convention_env_prefix="ONEX_",
)
secret_resolver = SecretResolver(config=secret_config)

# Create handler config resolver
config = ModelBindingConfigResolverConfig(
    config_dir=Path("/etc/onex/handlers"),
    cache_ttl_seconds=600.0,          # 10 minutes
    env_prefix="ONEX_HANDLER",        # ONEX_HANDLER_DB_TIMEOUT_MS
    secret_resolver=secret_resolver,  # For vault: resolution
    strict_validation=True,           # Fail on unknown fields
    allowed_schemes=frozenset({"file", "env"}),  # Disable vault: scheme
)
```

### Config Reference Formats

| Format | Example | Description |
|--------|---------|-------------|
| File (absolute) | `file:///etc/onex/db.yaml` | Absolute file path |
| File (relative) | `file://configs/handler.yaml` | Relative to `config_dir` |
| File (shorthand) | `file:config.yaml` | Shorthand relative path |
| Environment | `env:DB_CONFIG` | JSON/YAML in environment variable |
| Vault | `vault:secret/data/handlers/db#config` | Via SecretResolver |

**File Reference Examples**:

```python
# Absolute path - works without config_dir
binding = resolver.resolve(
    handler_type="db",
    config_ref="file:///etc/onex/handlers/database.yaml",
)

# Relative path - requires config_dir
binding = resolver.resolve(
    handler_type="db",
    config_ref="file://database.yaml",  # Resolves to config_dir/database.yaml
)

# Shorthand relative path
binding = resolver.resolve(
    handler_type="db",
    config_ref="file:configs/db.yaml",  # Same as file://configs/db.yaml
)
```

**Environment Reference Example**:

```python
# Set environment variable with JSON config
# export DB_CONFIG='{"pool_size": 20, "timeout_ms": 10000}'

binding = resolver.resolve(
    handler_type="db",
    config_ref="env:DB_CONFIG",
)
# binding.timeout_ms == 10000
```

**Vault Reference Example**:

```python
# Vault contains: {"config": {"pool_size": 20, "timeout_ms": 10000}}
binding = resolver.resolve(
    handler_type="db",
    config_ref="vault:secret/data/handlers/db#config",
)
```

---

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models import ModelBindingConfigResolverConfig

# Create resolver with defaults
config = ModelBindingConfigResolverConfig()
resolver = BindingConfigResolver(config=config)

# Resolve from inline config only
binding = resolver.resolve(
    handler_type="vault",
    inline_config={
        "timeout_ms": 30000,
        "rate_limit_per_second": 100.0,
        "retry_policy": {
            "max_retries": 5,
            "backoff_strategy": "exponential",
        },
    },
)

print(f"Handler: {binding.handler_type}")
print(f"Timeout: {binding.timeout_ms}ms")
print(f"Rate limit: {binding.rate_limit_per_second}/sec")
print(f"Retries: {binding.retry_policy.max_retries}")
```

### File-Based Configuration

Create a YAML configuration file:

```yaml
# /etc/onex/handlers/database.yaml
handler_type: db
name: primary-postgres
priority: 10
timeout_ms: 5000
rate_limit_per_second: 500.0
retry_policy:
  max_retries: 3
  backoff_strategy: exponential
  base_delay_ms: 100
  max_delay_ms: 5000
```

Load and resolve:

```python
config = ModelBindingConfigResolverConfig(
    config_dir=Path("/etc/onex/handlers"),
)
resolver = BindingConfigResolver(config=config)

binding = resolver.resolve(
    handler_type="db",
    config_ref="file:database.yaml",
)
```

### Environment Overrides

Environment variables always override other sources:

```bash
# Set environment overrides
export HANDLER_DB_TIMEOUT_MS=60000
export HANDLER_DB_MAX_RETRIES=5
export HANDLER_DB_ENABLED=false
```

```python
# File config says timeout_ms=5000, but env says 60000
binding = resolver.resolve(
    handler_type="db",
    config_ref="file:database.yaml",
)

# Environment wins: timeout_ms == 60000
print(f"Timeout: {binding.timeout_ms}ms")  # 60000
print(f"Enabled: {binding.enabled}")        # False
```

### Merging Multiple Sources

When both `config_ref` and `inline_config` are provided:

```python
binding = resolver.resolve(
    handler_type="db",
    config_ref="file:database.yaml",      # Base configuration
    inline_config={
        "timeout_ms": 10000,              # Override specific values
        "priority": 20,
    },
)
# Result: file config + inline overrides + env overrides
```

### Async Resolution

For I/O-bound configurations (file or Vault):

```python
async def resolve_handlers():
    config = ModelBindingConfigResolverConfig(
        config_dir=Path("/etc/onex/handlers"),
    )
    resolver = BindingConfigResolver(config=config)

    # Resolve single config asynchronously
    binding = await resolver.resolve_async(
        handler_type="vault",
        config_ref="file:vault.yaml",
    )

    return binding
```

### Parallel Async Resolution

Resolve multiple configurations in parallel:

```python
async def resolve_all_handlers():
    configs = await resolver.resolve_many_async([
        {"handler_type": "vault", "config_ref": "file:vault.yaml"},
        {"handler_type": "db", "config_ref": "file:database.yaml"},
        {"handler_type": "consul", "config_ref": "file:consul.yaml"},
        {"handler_type": "kafka", "config": {"timeout_ms": 30000}},
    ])
    return configs  # List[ModelBindingConfig]
```

---

## Environment Variable Overrides

### Override Naming Convention

Pattern: `{ENV_PREFIX}_{HANDLER_TYPE}_{FIELD}`

| Override | Example | Description |
|----------|---------|-------------|
| Timeout | `HANDLER_VAULT_TIMEOUT_MS=60000` | Override timeout |
| Enabled | `HANDLER_DB_ENABLED=false` | Enable/disable handler |
| Priority | `HANDLER_CONSUL_PRIORITY=50` | Execution priority |
| Rate Limit | `HANDLER_HTTP_RATE_LIMIT_PER_SECOND=100.5` | Rate limiting |
| Name | `HANDLER_DB_NAME=primary-db` | Display name |
| Max Retries | `HANDLER_VAULT_MAX_RETRIES=5` | Retry policy |
| Backoff Strategy | `HANDLER_DB_BACKOFF_STRATEGY=fixed` | Retry backoff |
| Base Delay | `HANDLER_KAFKA_BASE_DELAY_MS=200` | Initial retry delay |
| Max Delay | `HANDLER_KAFKA_MAX_DELAY_MS=10000` | Max retry delay |

### Supported Override Fields

```python
_ENV_OVERRIDE_FIELDS = {
    "ENABLED": "enabled",
    "PRIORITY": "priority",
    "TIMEOUT_MS": "timeout_ms",
    "RATE_LIMIT_PER_SECOND": "rate_limit_per_second",
    "MAX_RETRIES": "max_retries",
    "BACKOFF_STRATEGY": "backoff_strategy",
    "BASE_DELAY_MS": "base_delay_ms",
    "MAX_DELAY_MS": "max_delay_ms",
    "NAME": "name",
}
```

### Type Coercion

| Field Type | Env Value | Parsed As |
|------------|-----------|-----------|
| `int` | `"30000"` | `30000` |
| `float` | `"1.5"` | `1.5` |
| `bool` | `"true"` / `"1"` / `"yes"` | `True` |
| `bool` | `"false"` / `"0"` / `"no"` | `False` |
| `str` | `"value"` | `"value"` |

**Example**:

```bash
export HANDLER_DB_TIMEOUT_MS=60000          # int
export HANDLER_DB_RATE_LIMIT_PER_SECOND=1.5  # float
export HANDLER_DB_ENABLED=false              # bool
export HANDLER_DB_NAME="backup-db"           # str
```

---

## Caching

### Cache Behavior

- Enabled by default (`enable_caching=True`)
- TTL-based expiration (`cache_ttl_seconds=300.0` default)
- Per-handler-type cache entries
- Thread-safe cache operations
- Expired entries evicted on access

### Cache Entry Structure

```python
@dataclass
class _CacheEntry:
    config: ModelBindingConfig  # Resolved configuration
    expires_at: datetime               # Expiration timestamp (UTC)
    source: str                        # Description for debugging
```

### Cache Statistics

```python
stats = resolver.get_cache_stats()

print(f"Cached entries: {stats.total_entries}")
print(f"Cache hits: {stats.hits}")
print(f"Cache misses: {stats.misses}")
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Expired evictions: {stats.expired_evictions}")
print(f"Manual refreshes: {stats.refreshes}")
print(f"File loads: {stats.file_loads}")
print(f"Env loads: {stats.env_loads}")
print(f"Vault loads: {stats.vault_loads}")
print(f"Total loads: {stats.total_loads}")
```

### Cache Invalidation

```python
# Invalidate specific handler type
resolver.refresh("vault")

# Invalidate all cached configurations
resolver.refresh_all()
```

### Disabling Cache

For development or frequently changing configurations:

```python
config = ModelBindingConfigResolverConfig(
    enable_caching=False,  # Disable caching entirely
)
```

---

## Thread Safety

The resolver is thread-safe for concurrent access from both sync and async contexts.

### Sync Operations

```python
# Protected by threading.Lock
# Lock held during entire resolution for atomicity
binding = resolver.resolve(handler_type="db", inline_config={...})
```

### Async Operations

```python
# Per-handler-type asyncio.Lock
# Allows parallel resolution of different handlers
# Serializes resolution for same handler type
binding = await resolver.resolve_async(handler_type="db", config_ref="file:db.yaml")
```

### Locking Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Thread Safety Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  threading.Lock (_lock)                                        │ │
│  │  - Protects cache reads/writes                                 │ │
│  │  - Protects stats updates                                      │ │
│  │  - Held briefly for in-memory operations                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Per-Key asyncio.Lock (_async_key_locks)                       │ │
│  │  - One lock per handler_type                                   │ │
│  │  - Prevents duplicate async fetches for SAME handler           │ │
│  │  - Allows parallel fetches for DIFFERENT handlers              │ │
│  │  - Double-check locking pattern for cache                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Async Resolution Flow:                                              │
│  1. Acquire threading.Lock -> check cache                           │
│  2. If miss: release threading.Lock                                 │
│  3. Acquire per-key asyncio.Lock                                    │
│  4. Double-check cache (another thread may have resolved)           │
│  5. Resolve from sources                                            │
│  6. Acquire threading.Lock -> cache result                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

### Error Types

| Error | When Raised |
|-------|-------------|
| `ProtocolConfigurationError` | Invalid config, missing file, parse errors, validation failures |

### Error Context

All errors include structured context via `ModelInfraErrorContext`:

```python
from omnibase_infra.errors import ProtocolConfigurationError

try:
    binding = resolver.resolve(
        handler_type="db",
        config_ref="file:missing.yaml",
    )
except ProtocolConfigurationError as e:
    # Access error context
    print(f"Operation: {e.context.operation}")       # "load_from_file"
    print(f"Target: {e.context.target_name}")        # "binding_config_resolver"
    print(f"Transport: {e.context.transport_type}") # EnumInfraTransportType.RUNTIME
    print(f"Correlation: {e.context.correlation_id}")
```

### Error Scenarios

| Scenario | Error Message |
|----------|---------------|
| Empty config_ref | `Config reference cannot be empty` |
| Invalid scheme | `Unknown scheme 'xyz'. Supported: file, env, vault` |
| File not found | `Configuration file not found` |
| Permission denied | `Permission denied reading configuration file` |
| Invalid YAML | `Invalid YAML in configuration file` |
| File too large | `Configuration file exceeds size limit` |
| Path traversal | `Configuration file path traversal not allowed` |
| Missing env var | `Environment variable not set: VAR_NAME` |
| No SecretResolver | `Vault scheme used but no SecretResolver configured` |
| Validation failure | `Invalid handler configuration: ...` |

### Error Sanitization

Error messages are sanitized to exclude sensitive information:

**Errors NEVER include**:
- Actual configuration values
- File contents
- Secret values
- Internal file paths (only scheme shown in source description)

**Errors MAY include**:
- Handler type
- Scheme name
- Environment variable names
- Correlation IDs

---

## Security Considerations

### Path Traversal Protection

File paths are validated to prevent traversal attacks:

```python
# BLOCKED: Path traversal attempt
result = ModelConfigRef.parse("file:../../../etc/passwd")
# result.success == False
# result.error_message == "Path traversal sequences ('..') are not allowed: ..."

# Also blocked when using relative paths with config_dir
binding = resolver.resolve(
    handler_type="db",
    config_ref="file:../../secret/config.yaml",  # Raises ProtocolConfigurationError
)
```

**How it works**:
- `ModelConfigRef.parse()` blocks `..` patterns in all paths
- `_load_from_file()` resolves paths and validates they stay within `config_dir`
- Both absolute and relative paths are checked

### File Size Limits

Files exceeding 1MB are rejected to prevent memory exhaustion:

```python
MAX_CONFIG_FILE_SIZE = 1024 * 1024  # 1MB

# If config file is larger than 1MB:
# Raises: ProtocolConfigurationError("Configuration file exceeds size limit")
```

### Scheme Allowlisting

Restrict which schemes can be used:

```python
config = ModelBindingConfigResolverConfig(
    allowed_schemes=frozenset({"file", "env"}),  # Disable vault: scheme
)

# Using vault: now raises an error
binding = resolver.resolve(
    handler_type="db",
    config_ref="vault:secret/db",  # Raises: Scheme 'vault' is not in allowed schemes
)
```

### Vault Integration Security

- Vault access is via SecretResolver only (not direct Vault API)
- Inherits SecretResolver's security controls
- Bootstrap secrets for Vault must come from environment (see SecretResolver docs)

### Best Practices

| Do | Do Not |
|----|--------|
| Use `config_dir` to constrain file paths | Allow arbitrary absolute paths |
| Restrict `allowed_schemes` in production | Enable all schemes without need |
| Validate configurations before deployment | Trust untrusted config sources |
| Use environment overrides for secrets | Put secrets in config files |
| Set appropriate cache TTLs | Cache forever with `cache_ttl_seconds=86400` |

---

## Integration Points

### With Orchestrators

```python
from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes import NodeOrchestrator
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models import ModelBindingConfigResolverConfig

class MyOrchestrator(NodeOrchestrator):
    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

        # Get secret resolver from container if available
        secret_resolver = container.get("secret_resolver", default=None)

        self._config_resolver = BindingConfigResolver(
            ModelBindingConfigResolverConfig(
                config_dir=Path("/etc/onex/handlers"),
                secret_resolver=secret_resolver,
                cache_ttl_seconds=300.0,
            )
        )

    async def initialize_handlers(self) -> None:
        """Initialize handlers from contract bindings."""
        for binding_spec in self.contract.handler_bindings:
            config = await self._config_resolver.resolve_async(
                handler_type=binding_spec.handler_type,
                config_ref=binding_spec.config_ref,
                inline_config=binding_spec.config,
            )

            if config.enabled:
                handler = self._create_handler(config)
                self._register_handler(config.handler_type, handler)
```

### With Handler Registry

```python
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.handlers.handler_vault import HandlerVault
from omnibase_infra.handlers.handler_db import HandlerDb

# Resolve configuration
config = resolver.resolve(
    handler_type="vault",
    config_ref="file:vault.yaml",
)

# Create handler with resolved config
handler = HandlerVault(
    timeout_ms=config.timeout_ms,
    retry_policy=config.retry_policy,
)

# Register handler
registry.register(config.handler_type, handler)
```

### With Contract YAML

Handler bindings in `contract.yaml`:

```yaml
# contract.yaml
handler_bindings:
  - handler_type: vault
    config_ref: "file:configs/vault.yaml"
    priority: 10

  - handler_type: db
    config:
      timeout_ms: 5000
      pool_size: 20
    retry_policy:
      max_retries: 3
      backoff_strategy: exponential

  - handler_type: consul
    config_ref: "env:CONSUL_HANDLER_CONFIG"
    enabled: true
```

Loading bindings:

```python
def load_handler_bindings(contract_path: Path) -> list[ModelBindingConfig]:
    """Load and resolve handler bindings from contract."""
    with open(contract_path) as f:
        contract = yaml.safe_load(f)

    bindings = contract.get("handler_bindings", [])
    return resolver.resolve_many(bindings)
```

---

## Output Models

### ModelBindingConfig

The resolved configuration model:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `handler_type` | `str` | (required) | Handler type identifier |
| `name` | `str \| None` | `None` | Display name (defaults to handler_type) |
| `enabled` | `bool` | `True` | Whether handler is active |
| `priority` | `int` | `0` | Execution priority (-100 to 100) |
| `config_ref` | `str \| None` | `None` | External config reference |
| `config` | `dict \| None` | `None` | Inline configuration |
| `retry_policy` | `ModelRetryPolicy \| None` | `None` | Retry configuration |
| `timeout_ms` | `int` | `30000` | Operation timeout (100-600000) |
| `rate_limit_per_second` | `float \| None` | `None` | Max ops/sec (0.1-10000) |

### ModelRetryPolicy

Retry configuration:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Max retry attempts (0-10) |
| `backoff_strategy` | `"fixed" \| "exponential"` | `"exponential"` | Backoff type |
| `base_delay_ms` | `int` | `100` | Initial delay (10-60000) |
| `max_delay_ms` | `int` | `5000` | Max delay cap (100-300000) |

---

## ONEX Compliance

The BindingConfigResolver follows ONEX infrastructure patterns:

### Error Handling

Uses the standard `omnibase_infra.errors` hierarchy:

```python
from omnibase_infra.errors import ProtocolConfigurationError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

context = ModelInfraErrorContext.with_correlation(
    correlation_id=correlation_id,
    transport_type=EnumInfraTransportType.RUNTIME,
    operation="resolve_config",
    target_name="binding_config_resolver",
)

raise ProtocolConfigurationError(
    "Configuration file not found",
    context=context,
)
```

### Type Safety

- Uses `Literal["fixed", "exponential"]` for backoff strategy
- All fields have proper Pydantic validation
- No `Any` types (uses `object` for generic config dicts)
- Frozen models prevent mutation

### Not a Node

BindingConfigResolver is a **runtime utility**, not an ONEX node. It does not require:
- `contract.yaml` (not a node)
- `ModelONEXContainer` injection (standalone utility)
- Handler routing (direct method calls)

This is intentional - the resolver is a low-level primitive used by orchestrators, not a workflow component.

---

## API Reference

### BindingConfigResolver

| Method | Description |
|--------|-------------|
| `resolve(handler_type, config_ref, inline_config, correlation_id)` | Resolve config synchronously |
| `resolve_many(bindings, correlation_id)` | Resolve multiple configs synchronously |
| `resolve_async(handler_type, config_ref, inline_config, correlation_id)` | Resolve config asynchronously |
| `resolve_many_async(bindings, correlation_id)` | Resolve multiple configs in parallel |
| `refresh(handler_type)` | Invalidate cache for handler type |
| `refresh_all()` | Invalidate all cached configs |
| `get_cache_stats()` | Get cache statistics |

### Configuration Models

| Model | Purpose |
|-------|---------|
| `ModelBindingConfigResolverConfig` | Resolver configuration |
| `ModelBindingConfig` | Resolved handler configuration |
| `ModelRetryPolicy` | Retry behavior configuration |
| `ModelBindingConfigCacheStats` | Cache statistics |
| `ModelConfigRef` | Parsed config reference |
| `ModelConfigRefParseResult` | Config reference parse result |
| `EnumConfigRefScheme` | Config reference schemes (FILE, ENV, VAULT) |

---

## Migration Guide

This section helps teams migrate from common configuration patterns to `BindingConfigResolver`.

### Migrating from Direct Environment Variables

**Before (manual os.getenv)**:
```python
import os

class MyHandler:
    def __init__(self):
        self.timeout_ms = int(os.getenv("HANDLER_DB_TIMEOUT_MS", "5000"))
        self.retry_attempts = int(os.getenv("HANDLER_DB_RETRY_ATTEMPTS", "3"))
        self.connection_string = os.getenv("DB_CONNECTION_STRING", "")
```

**After (BindingConfigResolver)**:
```python
from omnibase_core.container import ModelONEXContainer
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models import ModelBindingConfigResolverConfig

class MyHandler:
    def __init__(self, container: ModelONEXContainer):
        resolver = BindingConfigResolver(
            config=ModelBindingConfigResolverConfig(env_prefix="HANDLER")
        )
        config = resolver.resolve(handler_type="db", inline_config={})
        self.timeout_ms = config.timeout_ms
        self.retry_attempts = config.retry_policy.max_retries if config.retry_policy else 3
        # Connection strings should come from Vault, not env directly
```

### Migrating from Config Files

**Before (manual file loading)**:
```python
import yaml
from pathlib import Path

def load_config(handler_type: str) -> dict:
    config_path = Path(f"configs/{handler_type}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

class MyHandler:
    def __init__(self, handler_type: str):
        config = load_config(handler_type)
        self.timeout_ms = config.get("timeout_ms", 5000)
        self.pool_size = config.get("pool_size", 10)
```

**After (BindingConfigResolver)**:
```python
from pathlib import Path
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.models import ModelBindingConfigResolverConfig

class MyHandler:
    def __init__(self, handler_type: str):
        resolver = BindingConfigResolver(
            config=ModelBindingConfigResolverConfig(
                config_dir=Path("configs"),
            )
        )
        # Config file reference resolved automatically
        config = resolver.resolve(
            handler_type=handler_type,
            config_ref=f"file:{handler_type}.yaml",
        )
        self.timeout_ms = config.timeout_ms
        # pool_size would be in config.config dict if not a standard field
```

### Migrating from Vault Direct Access

**Before (direct vault client)**:
```python
import hvac

class MyHandler:
    def __init__(self):
        client = hvac.Client(url="https://vault.example.com")
        client.token = os.getenv("VAULT_TOKEN")
        secret = client.secrets.kv.v2.read_secret_version(path="handlers/db")
        self.db_password = secret["data"]["data"]["password"]
        self.db_host = secret["data"]["data"]["host"]
```

**After (BindingConfigResolver)**:
```python
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.models import (
    ModelBindingConfigResolverConfig,
    ModelSecretResolverConfig,
)

class MyHandler:
    def __init__(self):
        # Create secret resolver for vault: references
        secret_resolver = SecretResolver(
            config=ModelSecretResolverConfig(
                enable_convention_fallback=True,
            )
        )

        resolver = BindingConfigResolver(
            config=ModelBindingConfigResolverConfig(
                secret_resolver=secret_resolver,
            )
        )

        # Vault reference resolved automatically via config_ref
        config = resolver.resolve(
            handler_type="db",
            config_ref="vault:secret/data/handlers/db#config",
        )
        # Or use vault: references in inline config values
        # config = resolver.resolve(
        #     handler_type="db",
        #     inline_config={"password": "vault:secret/data/handlers/db#password"},
        # )
```

### Key Benefits of Migration

| Aspect | Before | After |
|--------|--------|-------|
| **Source Priority** | Manual implementation required | Built-in: Env > Vault > File > Inline |
| **Caching** | None or custom implementation | TTL-based with statistics |
| **Thread Safety** | Manual locking required | Built-in sync/async safety |
| **Error Handling** | Inconsistent across handlers | Structured with correlation IDs |
| **Path Security** | Manual validation | Built-in traversal protection |
| **File Size Limits** | Often missing | 1MB limit prevents memory exhaustion |
| **Testing** | Mock individual sources | Mock resolver or container |
| **Environment Overrides** | Manual per-field implementation | Automatic with naming convention |

### Migration Checklist

Use this checklist when migrating handlers to `BindingConfigResolver`:

**Identification Phase**:
- [ ] Identify all handlers using direct `os.getenv()` calls
- [ ] Identify all handlers loading config files manually (yaml.safe_load, json.load)
- [ ] Identify all direct Vault client usage (hvac, vault-cli)
- [ ] Document current environment variable naming conventions
- [ ] List all config file locations and formats

**Preparation Phase**:
- [ ] Create `ModelBindingConfigResolverConfig` with appropriate settings
- [ ] Register `BindingConfigResolver` in container (if using DI)
- [ ] Create `SecretResolver` if Vault integration is needed
- [ ] Define config file directory structure (`config_dir`)
- [ ] Establish environment variable prefix convention (`env_prefix`)

**Migration Phase**:
- [ ] Update handler constructors to accept `ModelONEXContainer` or resolver
- [ ] Replace direct `os.getenv()` with resolver calls
- [ ] Replace manual file loading with `config_ref="file:..."` patterns
- [ ] Replace direct Vault access with `config_ref="vault:..."` or SecretResolver
- [ ] Convert custom retry logic to use `ModelRetryPolicy`

**Testing Phase**:
- [ ] Update unit tests to mock resolver instead of environment
- [ ] Verify environment override behavior (`HANDLER_{TYPE}_{FIELD}`)
- [ ] Test cache invalidation with `refresh()` and `refresh_all()`
- [ ] Verify error messages include correlation IDs
- [ ] Test file not found and permission error scenarios

**Cleanup Phase**:
- [ ] Remove deprecated config loading code
- [ ] Remove direct `hvac` client imports (if fully migrated)
- [ ] Update documentation to reference new patterns
- [ ] Remove old config file parsing utilities

### Common Migration Pitfalls

| Pitfall | Solution |
|---------|----------|
| Forgetting to set `config_dir` for relative paths | Always configure `config_dir` when using `file:` references |
| Using `vault:` scheme without `SecretResolver` | Ensure `secret_resolver` is provided in config |
| Expecting immediate cache updates | Call `refresh()` after external config changes |
| Mixing old and new patterns | Fully migrate each handler before moving to the next |
| Ignoring environment override naming | Follow `{PREFIX}_{HANDLER_TYPE}_{FIELD}` convention exactly |

---

## Related Documentation

- [`docs/patterns/secret_resolver.md`](./secret_resolver.md) - Secret management for vault: references
- [`docs/patterns/error_handling_patterns.md`](./error_handling_patterns.md) - Error context and sanitization
- [`docs/patterns/security_patterns.md`](./security_patterns.md) - Security guidelines
- [`docs/patterns/handler_plugin_loader.md`](./handler_plugin_loader.md) - Handler loading from contracts
- [`src/omnibase_infra/runtime/binding_config_resolver.py`](../../src/omnibase_infra/runtime/binding_config_resolver.py) - Implementation
- [`src/omnibase_infra/runtime/models/`](../../src/omnibase_infra/runtime/models/) - Configuration models
