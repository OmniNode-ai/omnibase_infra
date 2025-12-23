# Node Registration Orchestrator Protocol Architecture

## Overview

The Node Registration Orchestrator uses a **domain-grouped protocol pattern** where multiple cohesive protocols are defined in a single `protocols.py` file. This is an intentional architectural pattern documented in CLAUDE.md, not a code smell.

**Location**: `src/omnibase_infra/nodes/node_registration_orchestrator/protocols.py`

**Protocols Defined**:
- `ProtocolReducer`: Pure function that computes intents from events
- `ProtocolEffect`: Side-effectful executor that performs infrastructure operations

## Architectural Pattern: Reducer-Effect Workflow

The registration orchestrator implements a **reducer-effect separation pattern**, a functional architecture that cleanly separates pure computation from side effects:

```
                 ┌─────────────────────────────────────────────────┐
                 │         Node Registration Orchestrator          │
                 │                                                 │
  Event ────────►│  ┌──────────┐   intents   ┌──────────┐        │
                 │  │ Reducer  │────────────►│  Effect  │────────►│── Side Effects
                 │  │  (Pure)  │             │  (I/O)   │        │   (Consul, DB)
                 │  └──────────┘             └──────────┘        │
                 │       │                                        │
                 │       └─── state ──►                          │
                 └─────────────────────────────────────────────────┘
```

### ProtocolReducer

**Responsibility**: Deterministic computation of intents from events

**Contract**:
- MUST be deterministic (same inputs produce same outputs)
- MUST NOT perform I/O operations
- MUST return valid intents that the effect node can execute
- MUST sanitize any data included in error messages
- MAY filter duplicate or invalid events

**Method Signature**:
```python
async def reduce(
    self,
    state: ModelReducerState,
    event: ModelNodeIntrospectionEvent,
) -> tuple[ModelReducerState, list[ModelRegistrationIntent]]:
    ...
```

### ProtocolEffect

**Responsibility**: Execute side-effectful infrastructure operations

**Contract**:
- MUST execute exactly the operation specified by the intent
- MUST propagate correlation_id for distributed tracing
- MUST return a result even on failure (with success=False)
- MUST sanitize error messages before storing in result
- MAY implement retry logic internally

**Method Signature**:
```python
async def execute_intent(
    self,
    intent: ModelRegistrationIntent,
    correlation_id: UUID,
) -> ModelIntentExecutionResult:
    ...
```

## Why Domain-Grouped Protocols?

Per CLAUDE.md "Protocol File Naming" section:

> "Domain-grouped protocols: Use `protocols.py` when multiple cohesive protocols belong to a specific domain or node module"

Domain grouping is preferred when:
1. **Protocols are tightly coupled** - `ProtocolReducer` produces intents that `ProtocolEffect` consumes
2. **Protocols define the complete interface** - Together they define the full registration workflow contract
3. **Protocols share common type dependencies** - Both use `ModelRegistrationIntent`, `ModelReducerState`, and related models

Separating these into individual files would:
- Create unnecessary file fragmentation
- Obscure the relationship between reducer and effect
- Add import complexity without benefit

## Thread Safety Requirements

Both protocols require thread-safe implementations for concurrent async operations.

### ProtocolReducer Thread Safety

```python
# Thread Safety Guidelines:
# - Same reducer instance may process multiple events concurrently
# - Treat ModelReducerState as immutable (return new instances)
# - Avoid instance-level caches that could cause race conditions
```

### ProtocolEffect Thread Safety

```python
# Thread Safety Guidelines:
# - Multiple async tasks may invoke execute_intent() simultaneously
# - Use asyncio.Lock for any shared mutable state
# - Ensure underlying clients (Consul, PostgreSQL) are async-safe
```

## Error Sanitization Guidelines

All implementations MUST follow ONEX error sanitization guidelines from CLAUDE.md.

### NEVER Include in Error Messages

- Passwords, API keys, tokens, secrets
- Full connection strings with credentials
- PII (names, emails, SSNs, phone numbers)
- Internal IP addresses (in production)
- Private keys or certificates
- Raw event payload content (may contain secrets)

### SAFE to Include in Error Messages

- Service names (e.g., "consul", "postgres")
- Operation names (e.g., "register", "upsert")
- Correlation IDs (always include for tracing)
- Error codes (e.g., EnumCoreErrorCode values)
- Sanitized hostnames (e.g., "db.example.com")
- Port numbers, retry counts, timeout values
- Field names that are invalid or missing
- node_id (UUID, not PII)

## Validation Exemption

The `protocols.py` file triggers a validation warning for having multiple protocols in one file. This is an intentional pattern with a documented exemption.

**Exemption Location**: `src/omnibase_infra/validation/validation_exemptions.yaml`

**Exemption Reason**:
- Domain-grouped protocols for registration orchestrator workflow
- Protocols define the complete interface for the reducer-effect pattern
- Per CLAUDE.md "Protocol File Naming" convention

**Ticket**: OMN-888

## Review Criteria

This exemption should be reviewed if:

1. **Protocol count increases significantly** - If more than 4-5 protocols are added, consider whether the file should be split
2. **Protocols become decoupled** - If reducer and effect evolve to have independent lifecycles
3. **New patterns emerge** - If the codebase develops a different convention for workflow protocols

## Related Documentation

- **Implementation**: `src/omnibase_infra/nodes/node_registration_orchestrator/protocols.py`
- **Models**: `src/omnibase_infra/nodes/node_registration_orchestrator/models/`
- **Orchestrator Node**: `src/omnibase_infra/nodes/node_registration_orchestrator/node.py`
- **CLAUDE.md**: "Protocol File Naming" section
- **Ticket**: OMN-888 (Node Registration Orchestrator Workflow)

## Implementation Status

| Component | Status | Location | Ticket |
|-----------|--------|----------|--------|
| Protocols | **Complete** | `protocols.py` | OMN-888 |
| Models | **Complete** | `models/` | OMN-888 |
| Orchestrator Node | **Complete** | `node.py` | OMN-888 |
| Reducer Implementation | **Pending** | N/A | OMN-889 |
| Effect Implementation | **Placeholder** | `nodes/node_registry_effect/` | OMN-890 |
| Intent Models (Core) | **Pending** | N/A | OMN-912 |
| Projection Reader | **Pending** | N/A | OMN-930 |
| Time Injection Wiring | **Pending** | N/A | OMN-973 |

### Effect Node Status (OMN-890)

The `node_registry_effect` module at `src/omnibase_infra/nodes/node_registry_effect/` is currently a **placeholder**:

- The `__init__.py` exists but exports nothing (`__all__: list[str] = []`)
- No `node.py` or `contract.yaml` implementation
- Cannot execute Consul or PostgreSQL registration intents
- Blocked by: OMN-889 (reducer must generate intents first)

This is documented in the module docstring and tracked by OMN-890.
