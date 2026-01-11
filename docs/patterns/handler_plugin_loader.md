# Handler Plugin Loader Pattern

## Overview

The `HandlerPluginLoader` enables contract-driven handler discovery by dynamically loading Python handler classes from YAML contract specifications. This plugin-based approach decouples handler registration from orchestrator code, supporting the ONEX principle of declarative node configuration.

**Key Capabilities**:
- Single contract loading from a specific path
- Directory-based discovery with recursive scanning
- Glob pattern-based discovery for flexible matching
- Protocol validation via duck typing

## Why Plugin Pattern Over Hardcoded Registry

The plugin pattern offers significant advantages over hardcoded handler registries:

| Approach | Trade-offs |
|----------|------------|
| **Hardcoded Registry** | Compile-time safety, but tight coupling and requires code changes to add handlers |
| **Plugin Pattern** | Loose coupling, runtime discovery, but requires path validation |

### Benefits of Plugin Architecture

1. **Contract-driven**: Handler configuration lives in `contract.yaml` or `handler_contract.yaml`, not Python code
2. **Loose coupling**: Orchestrators don't import handler modules directly
3. **Runtime discovery**: New handlers can be added without modifying orchestrator code
4. **Testability**: Mock handlers can be injected via contract configuration
5. **Deployment flexibility**: Different environments can use different handlers via contract configuration

### Hardcoded Registry Drawbacks

```python
# AVOID: Hardcoded registry pattern
HANDLER_REGISTRY = {
    "auth": "myapp.handlers.AuthHandler",
    "validate": "myapp.handlers.ValidateHandler",
    # Adding a new handler requires code change AND deployment
}
```

With hardcoded registries:
- Adding handlers requires code changes
- Tight coupling between orchestrator and handler modules
- Testing requires mocking the registry
- Environment-specific handlers require conditional imports

---

## Why YAML Contracts Over Python Decorators

While Python decorators like `@register_handler("auth")` seem convenient, YAML contracts offer better tooling support and auditability:

| Aspect | YAML Contracts | Python Decorators |
|--------|---------------|-------------------|
| **Tooling** | Machine-readable, lintable, diffable | Requires AST parsing |
| **Auditability** | Changes visible in git, reviewable | Scattered across codebase |
| **Non-Python access** | CI/CD, dashboards can read contracts | Requires Python runtime |
| **Separation of concerns** | Configuration separate from logic | Mixes config with code |
| **Discovery** | Single scan of contract files | Must import all modules to find decorators |

### YAML Contract Advantages

1. **Machine-Readable**: CI/CD pipelines can validate contracts without Python
2. **Centralized Discovery**: Find all handlers by scanning YAML files
3. **Clean Diffs**: Contract changes are easy to review
4. **IDE Support**: YAML validation and completion available
5. **Documentation**: Contracts serve as handler documentation

---

## Contract-Based Handler Declaration

### Contract File Structure

The loader recognizes two contract file names:

| Filename | Purpose |
|----------|---------|
| `handler_contract.yaml` | Dedicated handler contract (preferred) |
| `contract.yaml` | General ONEX contract with handler fields |

### Required Fields

```yaml
# handler_contract.yaml
handler_name: "auth.validate"           # Unique handler identifier
handler_class: "myapp.handlers.auth.AuthHandler"  # Fully-qualified class path
handler_type: "effect"                  # Handler type: effect, compute, reducer, orchestrator
```

### Full Contract Example

```yaml
# handler_contract.yaml
handler_name: "db.query"
handler_class: "myapp.handlers.database.QueryHandler"
handler_type: "effect"
capability_tags:
  - database
  - async
  - pooled
```

### Contract in Node Directory

```
nodes/authentication/
    contract.yaml           # Node contract with handler routing
    node.py                 # Declarative node class
    handlers/
        handler_contract.yaml   # Handler-specific contract
        handler_authenticate.py # Handler implementation
```

---

## Contract File Precedence

> **FAIL-FAST: Ambiguous Contract Configuration Raises Error**
>
> When **both** `handler_contract.yaml` **and** `contract.yaml` exist in the **same directory**,
> the loader raises a `ProtocolConfigurationError` with error code `AMBIGUOUS_CONTRACT_CONFIGURATION`
> (HANDLER_LOADER_040). The loader does NOT load either file in this case.
>
> This fail-fast behavior prevents:
> - Duplicate handler registrations
> - Confusion about which contract is authoritative
> - Unexpected runtime behavior from conflicting configurations
>
> **Solution**: Use only **ONE** contract file per handler directory.

### Discovery Rules

When scanning a directory, the loader discovers contracts following these rules:

1. **Both filenames searched**: `handler_contract.yaml` and `contract.yaml` are both valid
2. **No precedence between files**: Both files in different directories are loaded
3. **FAIL-FAST if both in same directory**: Raises `ProtocolConfigurationError` immediately
4. **Deduplication by resolved path**: Same file via different paths is loaded once
5. **Sorted processing**: Discovered paths are sorted for deterministic order

### Same Directory Behavior (Ambiguous Configuration)

If both `handler_contract.yaml` and `contract.yaml` exist in the same directory,
the loader **fails fast** with an error:

```
ProtocolConfigurationError: Ambiguous contract configuration in 'auth':
Found both 'handler_contract.yaml' and 'contract.yaml'.
Use only ONE contract file per handler directory to avoid conflicts.
```

**Error Code**: `HANDLER_LOADER_040` (AMBIGUOUS_CONTRACT_CONFIGURATION)

```
# Directory structure (CAUSES ERROR)
handlers/
    handler_contract.yaml   # Found
    contract.yaml          # Also found = ERROR!

# Result: ProtocolConfigurationError raised - no handlers loaded
```

### Why Fail-Fast Instead of Precedence?

The loader intentionally raises an error instead of implementing precedence because:

1. **Explicit is better than implicit**: Silent precedence could mask configuration errors
2. **Fail-fast philosophy**: Errors are caught early at startup, not at runtime
3. **No assumptions**: The loader cannot know which file the user intends to be authoritative
4. **Clear resolution**: Error message tells exactly what to do

### Correct Configuration (Recommended)

Use **one contract file per handler directory** to avoid ambiguity:

```
# CORRECT: One contract per directory
nodes/auth/
    handler_contract.yaml   # Preferred: dedicated handler contract
    handler_auth.py

nodes/validate/
    handler_contract.yaml
    handler_validate.py
```

### Incorrect Configuration (Causes Error)

```
# INCORRECT: Both contract types in same directory
nodes/auth/
    handler_contract.yaml   # Defines "auth.validate" handler
    contract.yaml          # Also defines "auth.validate" handler (conflict!)
    handler_auth.py

# This will:
# 1. Raise ProtocolConfigurationError with AMBIGUOUS_CONTRACT_CONFIGURATION
# 2. Stop handler loading immediately (fail-fast)
# 3. Provide actionable error message explaining how to fix
```

### Resolving Ambiguous Configurations

If you encounter the `AMBIGUOUS_CONTRACT_CONFIGURATION` error:

1. **Identify the intended contract**: Decide which file should be authoritative
2. **Remove or rename the other**: Delete the unused contract or rename it (e.g., `contract.yaml.bak`)
3. **Consolidate if needed**: Merge handler definitions into a single contract file
4. **Verify after changes**: Re-run the loader and confirm no errors occur

---

## Protocol Validation

The loader validates that loaded classes implement `ProtocolHandler` using duck typing:

### Required Methods

| Method | Type | Purpose |
|--------|------|---------|
| `handler_type` | Property | Returns handler type identifier |
| `initialize(config)` | Async | Connection/pool setup |
| `shutdown(timeout_seconds)` | Async | Resource cleanup |
| `execute(request, operation_config)` | Async | Operation execution |
| `describe()` | Sync | Handler metadata/introspection |

### Validation Example

```python
class ValidHandler:
    """Handler implementing ProtocolHandler via duck typing."""

    @property
    def handler_type(self) -> str:
        return "effect"

    async def initialize(self, config: dict) -> None:
        """Initialize handler connections."""
        pass

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Release resources."""
        pass

    async def execute(self, request: object, config: object) -> object:
        """Execute operation."""
        return {}

    def describe(self) -> dict:
        """Return handler metadata."""
        return {"handler_id": "valid.handler", "version": "1.0.0"}
```

### Why Duck Typing

ONEX uses duck typing for protocol validation to:
1. Avoid tight coupling to specific base classes
2. Enable flexibility in handler implementation strategies
3. Support mixin-based handler composition
4. Allow testing with mock handlers that satisfy the protocol

---

## Security Considerations

**CRITICAL**: YAML contracts are treated as executable code, not mere configuration.

This section provides comprehensive security documentation for the handler plugin loader's dynamic import mechanism. Understanding these security implications is essential for safe production deployment.

---

### Dynamic Import Security Model

The `HandlerPluginLoader` uses Python's `importlib.import_module()` to dynamically load handler classes. This is a powerful but security-sensitive operation.

#### Why Dynamic Import Is Used

| Benefit | Explanation |
|---------|-------------|
| **Contract-driven architecture** | Handler bindings defined in YAML, not hardcoded |
| **Plugin extensibility** | Third-party handlers load without core runtime changes |
| **Deployment flexibility** | Different environments use different handlers via contracts |
| **Zero coupling** | Runtime has no compile-time dependency on handler implementations |

#### Why Contracts Equal Code

When the loader processes a contract:
1. It reads the `handler_class` fully-qualified path (e.g., `myapp.handlers.AuthHandler`)
2. It calls `importlib.import_module()` on the module path
3. **The module's top-level code executes during import**

This means:
- Module-level side effects execute immediately when a contract is loaded
- Malicious contracts can execute arbitrary code if an attacker can write to contract directories
- Import-time vulnerabilities in handler modules are triggered by contract discovery

---

### Threat Model

Understanding what the loader protects against (and what it doesn't) is critical for secure deployment.

#### Attack Vectors

| Attack Vector | Description | Risk Level |
|---------------|-------------|------------|
| **Malicious contract injection** | Attacker writes contract pointing to malicious module | CRITICAL |
| **Module path manipulation** | Contract specifies path to unintended module | HIGH |
| **YAML deserialization attack** | Malicious YAML constructs execute code | HIGH (mitigated) |
| **Memory exhaustion** | Oversized contract files consume memory | MEDIUM (mitigated) |
| **Import side effects** | Legitimate modules have unintended import-time behavior | MEDIUM |
| **Handler instantiation exploits** | Malicious code runs when handler is instantiated | HIGH |

#### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRUSTED ZONE                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ Contract Files  │───>│ HandlerPlugin   │───>│ Handler Modules │  │
│  │ (YAML)          │    │ Loader          │    │ (Python)        │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│         │                        │                      │           │
│         ▼                        ▼                      ▼           │
│  Filesystem access       importlib.import_module   Arbitrary code   │
│  controls                                          execution        │
└─────────────────────────────────────────────────────────────────────┘
         │
         │ TRUST BOUNDARY: Everything entering this zone must be trusted
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      UNTRUSTED ZONE                                  │
│  User input, external APIs, unverified sources                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principle**: Contract files and handler modules MUST come from trusted sources. The loader cannot distinguish between legitimate and malicious module paths.

---

### Built-in Security Controls

The loader implements several protections:

| Protection | Implementation | Mitigates | Location |
|------------|----------------|-----------|----------|
| **File size limits** | `MAX_CONTRACT_SIZE = 10MB` | Memory exhaustion, DoS | `_validate_file_size()` |
| **YAML safe loading** | `yaml.safe_load()` | YAML deserialization attacks (`!!python/object`) | `load_from_contract()` |
| **Protocol validation** | Duck typing checks for all 5 `ProtocolHandler` methods | Loading arbitrary non-handler classes | `_validate_handler_protocol()` |
| **Error containment** | Graceful failure per contract | Single bad contract doesn't crash system | `load_from_directory()` |
| **Correlation ID tracking** | UUID4 for all operations | Audit trail for security events | All public methods |
| **Class type verification** | `isinstance(handler_class, type)` check | Loading non-class objects | `_import_handler_class()` |

#### Protocol Validation Details

The loader verifies loaded classes implement `ProtocolHandler` via duck typing:

```python
# Required methods checked by _validate_handler_protocol():
required_methods = [
    "handler_type",   # Property: handler type identifier
    "initialize",     # Async: connection/pool setup
    "shutdown",       # Async: resource cleanup
    "execute",        # Async: operation execution
    "describe",       # Sync: handler metadata
]
```

A class **fails validation** if any of these 5 methods are missing, preventing loading of arbitrary classes that don't implement the handler contract.

---

### What the Loader Does NOT Protect Against

**CRITICAL**: The following protections are NOT built into the loader and must be implemented at the deployment level:

| Gap | Description | Recommended Mitigation |
|-----|-------------|------------------------|
| **Path sandboxing** | Any module on `sys.path` can be imported | Application-level allowlist (see below) |
| **Module signature verification** | No cryptographic validation of handler code | Use signed container images |
| **Namespace restriction** | No built-in restriction on allowed module paths | Implement allowlist wrapper |
| **Import hook filtering** | No interception of `importlib` calls | Add custom `MetaPathFinder` |
| **Runtime isolation** | Handlers run in same process as loader | Use subprocess isolation for untrusted code |
| **Mutation prevention** | Loaded modules can modify global state | Run in separate process or container |

#### Why `yaml.safe_load()` Is Not Sufficient

`yaml.safe_load()` prevents YAML deserialization attacks (e.g., `!!python/object` tags) but does NOT prevent:

- Malicious `handler_class` values that point to attacker-controlled modules
- Module side effects during `import_module()` execution
- Post-import exploitation via handler instantiation

---

### Secure Deployment Checklist

#### Minimum Requirements (All Deployments)

All production deployments MUST implement:

1. **File permissions**: Contract directories readable only by runtime user
   ```bash
   chmod 755 /app/contracts
   chmod 644 /app/contracts/**/*.yaml
   chown -R appuser:appgroup /app/contracts
   ```

2. **Write protection**: Contract directories read-only at runtime
   ```yaml
   # Kubernetes volume mount
   volumeMounts:
     - name: handler-contracts
       mountPath: /app/contracts
       readOnly: true
   ```

3. **Source validation**: Contracts from trusted sources only
   - Version-controlled repositories with code review
   - Signed container images
   - Verified artifact registries

4. **Logging enabled**: Ensure handler loading is logged
   ```python
   import logging
   logging.getLogger("omnibase_infra.runtime.handler_plugin_loader").setLevel(logging.INFO)
   ```

#### High-Security Environments

For elevated security requirements, implement additional controls:

##### Path Allowlisting (Recommended)

Implement an application-level wrapper that validates module paths before loading:

```python
from pathlib import Path
from typing import Final

import yaml

from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader
from omnibase_infra.models.runtime import ModelLoadedHandler

# Define trusted namespace prefixes
ALLOWED_MODULE_PREFIXES: Final[tuple[str, ...]] = (
    "omnibase_infra.handlers.",     # First-party handlers
    "myapp.handlers.",              # Application handlers
    "approved_plugins.",            # Vetted third-party
)


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


def secure_load_from_contract(
    loader: HandlerPluginLoader,
    path: Path,
    correlation_id: str | None = None,
) -> ModelLoadedHandler:
    """Load handler with module path validation.

    Args:
        loader: The handler plugin loader
        path: Path to contract file
        correlation_id: Optional correlation ID for tracing

    Returns:
        Loaded handler if path passes validation

    Raises:
        SecurityError: If module path is not in allowlist
    """
    # Pre-validate module path before loader processes it
    with open(path) as f:
        contract_data = yaml.safe_load(f)

    handler_class = contract_data.get("handler_class", "")

    # Check against allowlist
    if not any(handler_class.startswith(prefix) for prefix in ALLOWED_MODULE_PREFIXES):
        raise SecurityError(
            f"Handler module path '{handler_class}' not in allowlist. "
            f"Allowed prefixes: {ALLOWED_MODULE_PREFIXES}"
        )

    # Path validated, proceed with loading
    return loader.load_from_contract(path, correlation_id)
```

##### Import Hook Monitoring

Use Python's import system to audit dynamic imports:

```python
import logging
import sys
from importlib.abc import MetaPathFinder

logger = logging.getLogger("security.imports")


class SecurityAuditFinder(MetaPathFinder):
    """Meta path finder that logs all dynamic imports."""

    def find_module(self, fullname: str, path=None):
        """Log import attempt without blocking.

        Returns None to allow the import to proceed via normal mechanisms.
        """
        logger.info(
            "Dynamic import: %s",
            fullname,
            extra={"module": fullname, "path": path},
        )
        return None  # Allow import to proceed


# Install the audit hook at application startup
sys.meta_path.insert(0, SecurityAuditFinder())
```

##### Subprocess Isolation

For complete isolation, run handlers in separate processes:

```python
import multiprocessing
from typing import Any


def run_handler_isolated(
    handler_module: str,
    handler_class: str,
    method: str,
    args: tuple[Any, ...],
) -> Any:
    """Execute handler method in isolated subprocess.

    The subprocess has its own memory space, so malicious code
    cannot affect the parent process.
    """
    def worker(queue, module, cls, method, args):
        import importlib
        mod = importlib.import_module(module)
        handler = getattr(mod, cls)()
        result = getattr(handler, method)(*args)
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=worker,
        args=(queue, handler_module, handler_class, method, args),
    )
    process.start()
    process.join(timeout=30)

    if process.is_alive():
        process.terminate()
        raise TimeoutError("Handler execution timed out")

    return queue.get()
```

---

### Monitoring and Incident Response

#### Security Event Logging

The loader logs security-relevant events with correlation IDs:

| Event | Log Level | When |
|-------|-----------|------|
| Contract load success | INFO | Handler successfully loaded |
| Contract load failure | WARNING | Individual contract failed |
| Protocol validation failure | WARNING | Class missing required methods |
| Module import error | WARNING | `importlib.import_module()` failed |
| File size exceeded | WARNING | Contract exceeds 10MB limit |
| Ambiguous contract configuration | WARNING | Both contract types in same directory |

#### Alerting Recommendations

Configure alerts for:

1. **Unexpected handler paths**: New module paths not seen before
2. **High failure rates**: Many contracts failing to load
3. **Import errors from unknown modules**: May indicate probing attacks
4. **File system changes**: Modifications to contract directories

#### Incident Response

If a malicious contract is suspected:

1. **Isolate**: Stop loading new contracts immediately
2. **Identify**: Check logs for correlation IDs of suspicious loads
3. **Contain**: Remove compromised contract files
4. **Analyze**: Review what code was executed during import
5. **Remediate**: Rotate any secrets that may have been exposed
6. **Prevent**: Implement additional controls (allowlisting, monitoring)

---

### Security Decision Trade-offs

| Decision | Trade-off |
|----------|-----------|
| **Plugin architecture vs hardcoded registry** | Flexibility vs compile-time safety |
| **Dynamic import vs static binding** | Runtime extensibility vs predictable behavior |
| **YAML contracts vs Python decorators** | Machine-readable config vs AST safety |
| **Duck typing vs explicit inheritance** | Loose coupling vs type guarantees |

The loader prioritizes flexibility and loose coupling, placing security responsibility on deployment configuration rather than runtime enforcement.

---

**For complete security rationale, see**: [ADR: Handler Plugin Loader Security Model](../decisions/adr-handler-plugin-loader-security.md)

---

## Error Codes

| Code | Enum Value | Meaning |
|------|------------|---------|
| HANDLER_LOADER_001 | `FILE_NOT_FOUND` | Contract file not found |
| HANDLER_LOADER_002 | `INVALID_YAML_SYNTAX` | YAML parsing failed |
| HANDLER_LOADER_003 | `SCHEMA_VALIDATION_FAILED` | Pydantic validation failed |
| HANDLER_LOADER_004 | `MISSING_REQUIRED_FIELDS` | Required contract fields missing |
| HANDLER_LOADER_005 | `FILE_SIZE_EXCEEDED` | Contract exceeds 10MB limit |
| HANDLER_LOADER_006 | `PROTOCOL_NOT_IMPLEMENTED` | Class doesn't implement ProtocolHandler |
| HANDLER_LOADER_007 | `NOT_A_FILE` | Path exists but is not a file |
| HANDLER_LOADER_008 | `FILE_READ_ERROR` | I/O error reading file |
| HANDLER_LOADER_009 | `FILE_STAT_ERROR` | I/O error getting file info |
| HANDLER_LOADER_010 | `MODULE_NOT_FOUND` | Handler module not found |
| HANDLER_LOADER_011 | `CLASS_NOT_FOUND` | Class not found in module |
| HANDLER_LOADER_012 | `IMPORT_ERROR` | Module import failed |
| HANDLER_LOADER_020 | `DIRECTORY_NOT_FOUND` | Directory not found |
| HANDLER_LOADER_021 | `PERMISSION_DENIED` | Permission denied |
| HANDLER_LOADER_022 | `NOT_A_DIRECTORY` | Path is not a directory |
| HANDLER_LOADER_030 | `EMPTY_PATTERNS_LIST` | Glob patterns list empty |

---

## Usage Examples

### Single Contract Loading

```python
from pathlib import Path
from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

loader = HandlerPluginLoader()

# Load single handler
handler = loader.load_from_contract(
    Path("src/handlers/auth/handler_contract.yaml"),
    correlation_id="request-123",
)

print(f"Loaded: {handler.handler_name}")
print(f"Type: {handler.handler_type}")
print(f"Class: {handler.handler_class}")
```

### Directory Discovery

```python
# Load all handlers from a directory tree
handlers = loader.load_from_directory(
    Path("src/handlers"),
    correlation_id="discovery-456",
)

print(f"Discovered {len(handlers)} handlers")
for handler in handlers:
    print(f"  - {handler.handler_name} ({handler.handler_type})")
```

### Glob Pattern Discovery

```python
# Discover with specific patterns
handlers = loader.discover_and_load(
    patterns=[
        "src/**/handler_contract.yaml",
        "plugins/**/contract.yaml",
    ],
    correlation_id="glob-789",
    base_path=Path("/app/project"),  # Optional: explicit base for deterministic results
)
```

### Error Handling

```python
from omnibase_infra.errors import ProtocolConfigurationError, InfraConnectionError

try:
    handler = loader.load_from_contract(contract_path)
except ProtocolConfigurationError as e:
    # Contract validation failed
    print(f"Contract error: {e}")
    print(f"Error code: {e.model.loader_error}")
    print(f"Correlation ID: {e.model.correlation_id}")
except InfraConnectionError as e:
    # Handler import failed
    print(f"Import error: {e}")
    print(f"Module path: {e.model.module_path}")
```

---

## Testing Handlers

### Mock Handler for Tests

```python
class MockTestHandler:
    """Mock handler for testing that satisfies ProtocolHandler."""

    @property
    def handler_type(self) -> str:
        return "mock"

    async def initialize(self, config: dict) -> None:
        pass

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        pass

    async def execute(self, request: object, config: object) -> object:
        return {"mocked": True}

    def describe(self) -> dict:
        return {"handler_id": "mock.test"}
```

### Test Contract Fixture

```python
import pytest
from pathlib import Path

@pytest.fixture
def valid_contract(tmp_path: Path) -> Path:
    """Create a valid handler contract for testing."""
    contract = tmp_path / "handler_contract.yaml"
    contract.write_text("""
handler_name: "test.handler"
handler_class: "tests.fixtures.MockTestHandler"
handler_type: "compute"
capability_tags:
  - test
  - mock
""")
    return contract
```

---

## Related Patterns

- [Contract Dependency Injection](./container_dependency_injection.md) - Container-based DI
- [Security Patterns](./security_patterns.md) - General security guidance
- [Error Handling Patterns](./error_handling_patterns.md) - Error context and codes
- [Correlation ID Tracking](./correlation_id_tracking.md) - Request tracing

## See Also

- `src/omnibase_infra/runtime/handler_plugin_loader.py` - Implementation
- `src/omnibase_infra/runtime/protocol_handler_plugin_loader.py` - Protocol definition
- `src/omnibase_infra/models/runtime/model_loaded_handler.py` - Result model
- `src/omnibase_infra/models/runtime/model_handler_contract.py` - Contract model
- [ADR: Handler Plugin Loader Security Model](../decisions/adr-handler-plugin-loader-security.md) - Security decisions
