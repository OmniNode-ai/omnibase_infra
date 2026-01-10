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

> **WARNING: No Automatic Precedence Between Contract Types**
>
> When **both** `handler_contract.yaml` **and** `contract.yaml` exist in the **same directory**,
> the loader will load **BOTH** files as separate handlers. There is **no precedence** -
> both are treated as valid, independent handler contracts.
>
> This behavior can lead to:
> - Duplicate handler registrations if both files define similar handlers
> - Confusion about which contract is the "source of truth"
> - Unexpected runtime behavior if handlers conflict
>
> **Best Practice**: Use only **ONE** contract file per handler directory.

### Discovery Rules

When scanning a directory, the loader discovers contracts following these rules:

1. **Both filenames searched**: `handler_contract.yaml` and `contract.yaml` are both valid
2. **No precedence between files**: Both files in different directories are loaded
3. **BOTH loaded if in same directory**: No file takes priority over the other
4. **Deduplication by resolved path**: Same file via different paths is loaded once
5. **Sorted processing**: Discovered paths are sorted for deterministic order

### Same Directory Behavior (Ambiguous Configuration)

If both `handler_contract.yaml` and `contract.yaml` exist in the same directory,
**both are loaded** as separate contracts. A warning is logged to alert operators:

```
WARNING: AMBIGUOUS CONTRACT CONFIGURATION: Directory '/app/handlers/auth' contains
both handler_contract.yaml and contract.yaml. BOTH files will be loaded as separate
handlers. This may cause duplicate handler registrations or unexpected behavior.
```

```
# Directory structure (AVOID THIS)
handlers/
    handler_contract.yaml   # Loaded as handler #1
    contract.yaml          # Loaded as handler #2 (if valid)

# Result: 2 handlers loaded - potentially causing conflicts!
```

### Why No Automatic Precedence?

The loader intentionally does **not** implement automatic precedence because:

1. **Explicit is better than implicit**: Silent precedence could mask configuration errors
2. **Different use cases**: Some projects may legitimately use both file types
3. **Fail-fast philosophy**: The warning alerts operators to potential issues early
4. **No assumptions**: The loader cannot know which file the user intends to be authoritative

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

### Incorrect Configuration (Avoid)

```
# INCORRECT: Both contract types in same directory
nodes/auth/
    handler_contract.yaml   # Defines "auth.validate" handler
    contract.yaml          # Also defines "auth.validate" handler (conflict!)
    handler_auth.py

# This will:
# 1. Log a warning about ambiguous configuration
# 2. Load BOTH handlers (may cause duplicate registration errors)
# 3. Potentially cause runtime conflicts
```

### Resolving Ambiguous Configurations

If you see the "AMBIGUOUS CONTRACT CONFIGURATION" warning:

1. **Identify the intended contract**: Decide which file should be authoritative
2. **Remove or rename the other**: Delete the unused contract or rename it (e.g., `contract.yaml.bak`)
3. **Consolidate if needed**: Merge handler definitions into a single contract file
4. **Verify after changes**: Re-run the loader and confirm no warnings appear

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

### Why Contracts Equal Code

When the loader processes a contract:
1. It reads the `handler_class` fully-qualified path
2. It calls `importlib.import_module()` on the module path
3. The module's top-level code executes during import

This means:
- Module-level side effects execute immediately
- Malicious contracts can execute arbitrary code if an attacker can write to contract directories
- Import-time vulnerabilities in handler modules are triggered by contract discovery

### Built-in Protections

| Protection | Implementation | Mitigates |
|------------|----------------|-----------|
| File size limits | `MAX_CONTRACT_SIZE = 10MB` | Memory exhaustion |
| YAML safe loading | `yaml.safe_load()` | YAML deserialization attacks |
| Protocol validation | Duck typing checks | Loading arbitrary non-handler classes |
| Error containment | Graceful failure per contract | Single bad contract doesn't crash system |
| Correlation ID tracking | UUID4 for all operations | Audit trail for security events |

### What the Loader Does NOT Protect Against

The loader explicitly does NOT provide:
- Path sandboxing (any module on `sys.path` can be imported)
- Module signature verification
- Allowlist enforcement (no built-in restriction on module paths)
- Import hook filtering

### Secure Deployment Checklist

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
   - Version-controlled repositories
   - Signed container images
   - Verified artifact registries

4. **Path allowlisting** (high-security environments):
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

**For complete security guidance, see**: [ADR: Handler Plugin Loader Security Model](../decisions/adr-handler-plugin-loader-security.md)

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
