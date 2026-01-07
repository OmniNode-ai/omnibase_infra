# Utility Directory Structure

This document clarifies the distinction between `utils/` and `shared/utils/` directories in `omnibase_infra`.

## Overview

The infrastructure package maintains two separate utility directories to enforce clear ownership boundaries and prevent circular dependencies:

```
src/omnibase_infra/
├── utils/                    # Cross-cutting infrastructure utilities
│   ├── __init__.py
│   ├── correlation.py        # Correlation ID generation/propagation
│   ├── util_dsn_validation.py
│   ├── util_env_parsing.py
│   ├── util_error_sanitization.py
│   └── util_semver.py
│
└── shared/
    └── utils/                # Legacy/transitional utilities (deprecated)
        └── __init__.py       # Re-exports from utils/ for compatibility
```

## Directory Purposes

### `utils/` - Cross-Cutting Infrastructure Utilities

**Location**: `src/omnibase_infra/utils/`

**Purpose**: General-purpose utilities used across multiple infrastructure components.

**Characteristics**:
- No dependencies on specific nodes or handlers
- Stateless, pure functions preferred
- Used by multiple modules across the package
- Part of the public API (`omnibase_infra.utils`)

**Current Utilities**:

| Module | Purpose |
|--------|---------|
| `correlation.py` | Correlation ID generation and context propagation for distributed tracing |
| `util_dsn_validation.py` | PostgreSQL DSN validation and credential sanitization |
| `util_env_parsing.py` | Type-safe environment variable parsing with validation |
| `util_error_sanitization.py` | Error message sanitization for secure logging and DLQ |
| `util_semver.py` | Semantic versioning validation utilities |

**Usage**:
```python
from omnibase_infra.utils import (
    generate_correlation_id,
    sanitize_error_message,
    parse_env_int,
)
```

### `shared/utils/` - Legacy/Transitional (Deprecated)

**Location**: `src/omnibase_infra/shared/utils/`

**Purpose**: Historical location for shared utilities, now deprecated.

**Status**: **DEPRECATED** - Do not add new utilities here.

This directory exists for backwards compatibility during migration. All utilities have been moved to `utils/` with re-exports maintained for existing imports.

**Migration**: Update imports from `omnibase_infra.shared.utils` to `omnibase_infra.utils`.

## Node-Specific Utilities

Nodes may have their own utility modules for node-specific helper functions:

```
nodes/node_registry_effect/
├── contract.yaml
├── node.py
├── handlers/
├── models/
└── utils/                    # Node-specific utilities (if needed)
    └── util_service_id.py    # Example: Service ID generation for this node
```

**Principle**: If a utility is only used by a single node, it belongs in that node's `utils/` directory. If it's used by multiple nodes, it belongs in the top-level `utils/`.

## Decision Matrix

| Utility Type | Location | Example |
|--------------|----------|---------|
| Cross-cutting (many consumers) | `utils/` | Correlation IDs, error sanitization |
| Node-specific (one consumer) | `nodes/<name>/utils/` | Node-specific formatting |
| Legacy (deprecated) | `shared/utils/` | Do not add new code |

## Import Conventions

**Preferred**:
```python
# Direct import from utils/
from omnibase_infra.utils import sanitize_error_message

# Or import the module for namespacing
from omnibase_infra.utils import util_error_sanitization
```

**Deprecated**:
```python
# Do not use - legacy import path
from omnibase_infra.shared.utils import sanitize_error_message  # DEPRECATED
```

## Related

- [Error Sanitization Patterns](./error_sanitization_patterns.md) - Uses `util_error_sanitization`
- [Correlation ID Tracking](./correlation_id_tracking.md) - Uses `correlation.py`
- [Container Dependency Injection](./container_dependency_injection.md) - Service resolution patterns
