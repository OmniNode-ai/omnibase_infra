# ADR: Handler Plugin Loader Security Model

**Status**: Accepted
**Date**: 2026-01-10
**Related Tickets**: OMN-1132, PR #134

## Context

The `HandlerPluginLoader` is a core runtime component that dynamically loads Python handler classes from YAML contract specifications. This enables contract-driven handler discovery without hardcoding handler references, supporting the ONEX principle of declarative node configuration.

The loader uses Python's `importlib.import_module()` to dynamically import handler classes specified in contract files:

```yaml
# handler_contract.yaml
handler_name: "AuthHandler"
handler_class: "myapp.handlers.auth.AuthHandler"
handler_type: "effect"
```

This dynamic import mechanism introduces security considerations that must be understood and mitigated in production deployments.

## Decision

**YAML contracts are treated as executable code, not mere configuration.**

The `HandlerPluginLoader` implements the following security model:

### 1. Contract Files = Code

Any YAML contract file that specifies a `handler_class` path has the same security implications as a Python script. When the loader processes a contract:

1. It reads the `handler_class` fully-qualified path (e.g., `myapp.handlers.AuthHandler`)
2. It calls `importlib.import_module()` on the module path
3. The module's top-level code executes during import

This means:
- **Module-level side effects execute immediately** when a contract is loaded
- **Malicious contracts can execute arbitrary code** if an attacker can write to contract directories
- **Import-time vulnerabilities** in handler modules are triggered by contract discovery

### 2. Built-in Protections

The loader implements several protections:

| Protection | Implementation | Mitigates |
|------------|----------------|-----------|
| File size limits | `MAX_CONTRACT_SIZE = 10MB` | Memory exhaustion from oversized files |
| YAML safe loading | `yaml.safe_load()` | YAML deserialization attacks |
| Protocol validation | Duck typing checks for `ProtocolHandler` | Loading arbitrary non-handler classes |
| Error containment | Graceful failure for individual contracts | Single bad contract doesn't crash system |
| Correlation ID tracking | UUID4 for all operations | Audit trail for security events |

### 3. What the Loader Does NOT Protect Against

The loader explicitly does NOT provide:

- **Path sandboxing**: Any module on `sys.path` can be imported
- **Module signature verification**: No cryptographic validation of handler code
- **Allowlist enforcement**: No built-in restriction on allowed module paths
- **Import hook filtering**: No interception of `importlib` calls

## Rationale

### Why dynamic import is necessary

1. **Contract-driven architecture**: ONEX uses declarative contracts to define handler bindings without code changes
2. **Plugin extensibility**: Third-party handlers can be loaded without modifying core runtime
3. **Deployment flexibility**: Different environments can use different handlers via contract configuration
4. **Zero coupling**: Runtime has no compile-time dependency on handler implementations

### Why contracts are treated as code

Attempting to "sanitize" import paths creates a false sense of security:

1. **Any valid Python module can have side effects** at import time
2. **Attackers who can write contracts** already have significant system access
3. **Path validation is insufficient** when the module itself may be compromised
4. **Defense in depth** requires treating the entire contract source as trusted

### Why safe_load is not sufficient

`yaml.safe_load()` prevents YAML deserialization attacks (e.g., `!!python/object` tags) but does NOT prevent:

- Malicious `handler_class` values
- Module side effects during `import_module()`
- Post-import exploitation via handler instantiation

## Consequences

### Positive

- Clear security model: contracts = code
- Enables flexible, declarative handler configuration
- Existing protections prevent common attack vectors
- Audit trail via correlation IDs for incident response

### Negative

- Requires strict access control on contract directories
- Cannot safely load contracts from untrusted sources
- No runtime isolation between handlers
- Side effects during import may cause unexpected behavior

### Neutral

- Security posture matches other plugin systems (e.g., pytest plugins, Django apps)
- Responsibility for handler security shifts to deployment configuration

## Deployment Guidance

### Minimum Requirements

All production deployments MUST implement:

1. **File permissions**: Contract directories readable only by runtime user
   ```bash
   chmod 755 /app/contracts
   chmod 644 /app/contracts/**/*.yaml
   chown -R appuser:appgroup /app/contracts
   ```

2. **Write protection**: Contract directories should be read-only at runtime
   ```bash
   # Mount contract volume as read-only in Kubernetes
   volumeMounts:
     - name: handler-contracts
       mountPath: /app/contracts
       readOnly: true
   ```

3. **Source validation**: Contracts should only come from trusted sources
   - Version-controlled repositories
   - Signed container images
   - Verified artifact registries

### High-Security Environments

For elevated security requirements, consider:

#### Path Allowlisting

Implement an application-level wrapper that validates module paths before loading:

```python
ALLOWED_MODULE_PREFIXES = (
    "myapp.handlers.",
    "approved_plugins.",
)

def secure_load_from_contract(loader: HandlerPluginLoader, path: Path) -> ModelLoadedHandler:
    """Load with module path validation."""
    with open(path) as f:
        contract_data = yaml.safe_load(f)

    handler_class = contract_data.get("handler_class", "")
    if not any(handler_class.startswith(prefix) for prefix in ALLOWED_MODULE_PREFIXES):
        raise SecurityError(f"Module path not in allowlist: {handler_class}")

    return loader.load_from_contract(path)
```

#### Import Hook Monitoring

Use Python's import system to audit dynamic imports:

```python
import sys
from importlib.abc import MetaPathFinder

class SecurityAuditFinder(MetaPathFinder):
    def find_module(self, fullname, path=None):
        logger.security(f"Dynamic import: {fullname}")
        return None  # Allow import to proceed

sys.meta_path.insert(0, SecurityAuditFinder())
```

#### Isolated Execution

For complete isolation, run handlers in separate processes:

```python
from multiprocessing import Process

def run_handler_isolated(handler_class: str, request: dict) -> dict:
    """Execute handler in isolated subprocess."""
    # Handler code cannot affect parent process memory
    ...
```

### Monitoring Recommendations

1. **Log all contract loading operations** with correlation IDs
2. **Alert on unexpected handler_class values** (new module paths)
3. **Monitor file system changes** to contract directories
4. **Track import errors** which may indicate probing attacks

## Implementation

### Current file structure

```
src/omnibase_infra/runtime/
    handler_plugin_loader.py    # Main loader implementation
    protocol_handler_plugin_loader.py  # Protocol definition
```

### Security-relevant code sections

```python
# Safe YAML loading (line 229)
raw_data = yaml.safe_load(f)

# Dynamic import (line 751)
module = importlib.import_module(module_path)

# Protocol validation (lines 609-709)
def _validate_handler_protocol(self, handler_class: type) -> bool:
    # Duck typing checks for ProtocolHandler methods
    ...
```

### Error codes for security events

| Code | Enum | Meaning |
|------|------|---------|
| HANDLER_LOADER_010 | `MODULE_NOT_FOUND` | Attempted import of non-existent module |
| HANDLER_LOADER_011 | `CLASS_NOT_FOUND` | Module exists but class missing |
| HANDLER_LOADER_012 | `IMPORT_ERROR` | Import failed (may indicate syntax/dependency issues) |
| HANDLER_LOADER_006 | `PROTOCOL_NOT_IMPLEMENTED` | Class doesn't implement handler protocol |

## References

- `src/omnibase_infra/runtime/handler_plugin_loader.py` (lines 25-31 for inline security notes)
- `src/omnibase_infra/runtime/protocol_handler_plugin_loader.py`
- `src/omnibase_infra/models/runtime/model_loaded_handler.py`
- `docs/patterns/security_patterns.md` (for general ONEX security guidance)
- Python `importlib` documentation: https://docs.python.org/3/library/importlib.html
