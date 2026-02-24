> **Navigation**: [Home](../index.md) > [Migration](README.md) > Wire Default Handlers

# Migration Guide: wire_default_handlers() to Contract-Driven Handler Discovery

**Status**: IMMEDIATE - No deprecation period per project policy
**Ticket**: OMN-1133 (Contract-based handler discovery)
**PR**: #143

---

## Executive Summary

The `wire_default_handlers()` function and its related helper `wire_handlers_from_contract()` (which loads handlers from a contract path) are replaced by contract-driven handler discovery via `HandlerPluginLoader` and `ContractHandlerDiscovery`. This migration eliminates hardcoded handler registrations in favor of YAML-based handler contracts.

**Project Policy Reminder**: Per CLAUDE.md, there is NO backwards compatibility. Breaking changes are acceptable and encouraged. Old patterns should be removed immediately.

---

## What Changed

### Before: Hardcoded Handler Wiring

```python
# OLD APPROACH - wiring.py
from omnibase_infra.runtime.wiring import wire_default_handlers

# Register all handlers from _KNOWN_HANDLERS dict
summary = wire_default_handlers()
# Result: {'handlers': ['consul', 'db', 'http', 'infisical'], 'event_buses': ['inmemory']}
```

The old approach:
- Maintained a hardcoded `_KNOWN_HANDLERS` dict in `wiring.py`
- Required code changes to add/remove handlers
- Tightly coupled runtime to specific handler implementations
- Made testing require import mocking

### After: Contract-Driven Discovery

```python
# NEW APPROACH - contract-driven discovery
from omnibase_infra.runtime import RuntimeHostProcess
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus

event_bus = InMemoryEventBus()
process = RuntimeHostProcess(
    event_bus=event_bus,
    input_topic="my.input.topic",
    contract_paths=["src/handlers"],  # Directories containing handler contracts
)

await process.start()  # Handlers discovered from contracts automatically
```

The new approach:
- Discovers handlers from YAML contract files
- No code changes needed to add/remove handlers
- Loose coupling via contracts
- Testable via contract fixtures

---

## Migration Steps

### Step 1: Create Handler Contracts

For each handler you want to register, create a `handler_contract.yaml` file:

```yaml
# src/handlers/http/handler_contract.yaml
handler_name: "http"
handler_class: "omnibase_infra.handlers.handler_http.HttpRestHandler"
handler_type: "effect"
capability_tags:
  - http
  - rest
```

**Required fields:**
- `handler_name`: Unique identifier for the handler
- `handler_class`: Fully-qualified Python class path
- `handler_type`: One of `effect`, `compute`, `reducer`, `orchestrator`

**Optional fields:**
- `capability_tags`: List of tags for handler categorization

### Step 2: Update RuntimeHostProcess Initialization

**Before:**
```python
from omnibase_infra.runtime.wiring import wire_default_handlers

# Called during startup
wire_default_handlers()

# RuntimeHostProcess created without contract_paths
process = RuntimeHostProcess(
    event_bus=event_bus,
    input_topic="my.topic",
)
```

**After:**
```python
# No wiring call needed - discovery is automatic
process = RuntimeHostProcess(
    event_bus=event_bus,
    input_topic="my.topic",
    contract_paths=["src/handlers"],  # Point to contract directories
)
```

### Step 3: Remove wire_default_handlers() Calls

Search for and remove all calls to:
- `wire_default_handlers()`
- `wire_handlers_from_contract()`
- `wire_custom_handler()`

These are replaced by handler contracts and `HandlerPluginLoader`.

### Step 4: Update Tests

**Before (Mock Wiring):**
```python
from unittest.mock import patch

@patch("omnibase_infra.runtime.wiring.wire_default_handlers")
def test_my_handler(mock_wire):
    mock_wire.return_value = {"handlers": ["http"], "event_buses": []}
    # ...
```

**After (Contract Fixtures):**
```python
import pytest
from pathlib import Path

@pytest.fixture
def handler_contract_dir(tmp_path: Path) -> Path:
    """Create handler contracts for testing."""
    handlers_dir = tmp_path / "handlers"
    handlers_dir.mkdir()

    http_dir = handlers_dir / "http"
    http_dir.mkdir()
    (http_dir / "handler_contract.yaml").write_text("""
handler_name: "http"
handler_class: "omnibase_infra.handlers.handler_http.HttpRestHandler"
handler_type: "effect"
capability_tags:
  - http
  - test
""")
    return handlers_dir

async def test_my_handler(handler_contract_dir: Path):
    process = RuntimeHostProcess(
        event_bus=InMemoryEventBus(),
        input_topic="test.input",
        contract_paths=[str(handler_contract_dir)],
    )
    await process.start()
    assert process.get_handler("http") is not None
    await process.stop()
```

---

## Contract File Structure

### Directory Layout

```
src/
|-- handlers/
|   |-- http/
|   |   |-- handler_contract.yaml    # Handler contract
|   |   |-- handler_http.py          # Handler implementation
|   |-- db/
|   |   |-- handler_contract.yaml
|   |   |-- handler_db.py
|   |-- consul/
|   |   |-- handler_contract.yaml
|   |   |-- handler_consul.py
```

### Contract Filenames

The loader recognizes two contract filenames:

| Filename | Purpose |
|----------|---------|
| `handler_contract.yaml` | Dedicated handler contract (preferred) |
| `contract.yaml` | General ONEX contract with handler fields |

**Important**: Do NOT use both filenames in the same directory - this raises `ProtocolConfigurationError` (HANDLER_LOADER_040).

---

## Fallback Behavior (Temporary)

During migration, `RuntimeHostProcess` supports fallback to legacy wiring:

```python
# When contract_paths is None or empty, falls back to wire_default_handlers()
process = RuntimeHostProcess(
    event_bus=event_bus,
    input_topic="my.topic",
    # contract_paths not provided = fallback to wire_default_handlers()
)
```

**This fallback is temporary** and will be removed in a future release. Migrate to contract-driven discovery immediately.

---

## HandlerPluginLoader Direct Usage

For advanced scenarios, use `HandlerPluginLoader` directly:

### Single Contract Loading

```python
from pathlib import Path
from uuid import uuid4
from omnibase_infra.runtime import HandlerPluginLoader

loader = HandlerPluginLoader()

# Load single handler
handler = loader.load_from_contract(
    Path("src/handlers/http/handler_contract.yaml"),
    correlation_id=uuid4(),
)

print(f"Loaded: {handler.handler_name}")
print(f"Class: {handler.handler_class}")
```

### Directory Discovery

```python
# Load all handlers from a directory tree
handlers = loader.load_from_directory(
    Path("src/handlers"),
    correlation_id=uuid4(),
)

print(f"Discovered {len(handlers)} handlers")
for handler in handlers:
    print(f"  - {handler.handler_name}")
```

### Glob Pattern Discovery

```python
# Discover with specific patterns
handlers = loader.discover_and_load(
    patterns=[
        "src/**/handler_contract.yaml",
        "plugins/**/contract.yaml",
    ],
    correlation_id=uuid4(),
)
```

---

## Security Considerations

Handler contracts are treated as **executable code** because they specify Python module paths that are dynamically imported. See [Handler Plugin Loader Security](../patterns/handler_plugin_loader.md#security-considerations) for:

- Namespace allowlisting
- File permission requirements
- Deployment security checklist

**Production recommendation**: Use namespace allowlisting:

```python
loader = HandlerPluginLoader(
    allowed_namespaces=["omnibase_infra.handlers.", "myapp.handlers."]
)
```

---

## Error Handling

The new approach provides structured error codes for troubleshooting:

| Code | Name | Meaning |
|------|------|---------|
| HANDLER_LOADER_001 | `FILE_NOT_FOUND` | Contract file not found |
| HANDLER_LOADER_002 | `INVALID_YAML_SYNTAX` | YAML parsing failed |
| HANDLER_LOADER_004 | `MISSING_REQUIRED_FIELDS` | Required fields missing |
| HANDLER_LOADER_010 | `MODULE_NOT_FOUND` | Handler module not found |
| HANDLER_LOADER_011 | `CLASS_NOT_FOUND` | Class not found in module |
| HANDLER_LOADER_040 | `AMBIGUOUS_CONTRACT_CONFIGURATION` | Both contract types in directory |

Access error codes via exception context:

```python
from omnibase_infra.errors import ProtocolConfigurationError

try:
    handler = loader.load_from_contract(contract_path)
except ProtocolConfigurationError as e:
    print(f"Error code: {e.model.context.get('loader_error')}")
    print(f"Correlation ID: {e.model.context.get('correlation_id')}")
```

---

## Comparison: Old vs New

| Aspect | wire_default_handlers() | Contract-Driven |
|--------|------------------------|-----------------|
| Handler registration | Code changes required | YAML contract changes |
| Adding handlers | Modify `_KNOWN_HANDLERS` dict | Create new contract file |
| Removing handlers | Modify code | Delete contract file |
| Testing | Import mocking | Contract fixtures |
| Handler discovery | Explicit function call | Automatic at startup |
| Configuration | None | Per-handler contracts |
| Security | Implicit | Namespace allowlisting |

---

## Checklist

- [ ] Create `handler_contract.yaml` for each handler
- [ ] Update `RuntimeHostProcess` initialization with `contract_paths`
- [ ] Remove `wire_default_handlers()` calls
- [ ] Remove `wire_handlers_from_contract()` calls
- [ ] Remove `wire_custom_handler()` calls
- [ ] Update tests to use contract fixtures
- [ ] Configure namespace allowlisting for production

---

## Related Documentation

- [Handler Plugin Loader Pattern](../patterns/handler_plugin_loader.md) - Complete pattern documentation
- [Handler Protocol-Driven Architecture](../architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md) - Architecture overview
- [ADR: Handler Plugin Loader Security](../decisions/adr-handler-plugin-loader-security.md) - Security decisions

## Files to Update

When migrating, search for and update these patterns:

```bash
# Find all wire_default_handlers() calls
grep -r "wire_default_handlers" src/ tests/

# Find all wire_handlers_from_contract() calls
grep -r "wire_handlers_from_contract" src/ tests/

# Find all wire_custom_handler() calls
grep -r "wire_custom_handler" src/ tests/
```

---

## Support

For migration questions:
1. Check the [Handler Plugin Loader Pattern](../patterns/handler_plugin_loader.md) documentation
2. Review test examples in `tests/integration/runtime/test_runtime_host_handler_discovery.py`
3. Open an issue with the `migration` label
