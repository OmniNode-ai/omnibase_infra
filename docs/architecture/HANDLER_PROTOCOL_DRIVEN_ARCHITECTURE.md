> **Navigation**: [Home](../index.md) > [Architecture](README.md) > Handler Protocol-Driven Architecture

# Handler Wiring -> Protocol-Driven Architecture

## Overview

This document defines the migration from hardcoded handler wiring to a protocol-driven,
contract-based handler loading system. The goal is to make handler registration explicit,
type-safe, and extensible while preserving MVP velocity.

**Implementation**: `src/omnibase_infra/runtime/` (wiring, handler_registry, runtime_host_process)
**Status**: Design Complete, Implementation Pending

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Protocol-First** | Runtime depends on `ProtocolHandlerSource`, never on concrete sources |
| **Explicit Bootstrap** | Hardcoded handlers are centralized and loudly logged as "compat mode" |
| **Structural Constraints** | "Handlers cannot publish" is enforced by capability deprivation, not policy |
| **Contract → Descriptor** | Contracts are serialization format; descriptors are runtime objects (one-way transform) |
| **No Backwards Compatibility** | Breaking changes acceptable; descriptor schema co-released with runtime |

## Architecture Diagram

```
+------------------------------------------------------------------+
|                    Handler Loading Architecture                    |
+------------------------------------------------------------------+
|                                                                    |
|   ProtocolHandlerSource (omnibase_spi)                            |
|        ^                                                           |
|        |                                                           |
|   +----+----+                                                      |
|   |         |                                                      |
|   |    HandlerBootstrapSource          HandlerContractSource       |
|   |    (MVP - hardcoded)               (Beta - YAML contracts)     |
|   |         |                                   |                  |
|   |         v                                   v                  |
|   |    _KNOWN_HANDLERS dict             **/handler_contract.yaml   |
|   |         |                                   |                  |
|   |         +---------------+-------------------+                  |
|   |                         |                                      |
|   |                         v                                      |
|   |              list[ModelHandlerDescriptor]                      |
|   |                         |                                      |
|   |                         v                                      |
|   |              RuntimeHostProcess                                |
|   |              (registers descriptors)                           |
|   |                         |                                      |
|   |                         v                                      |
|   |              Handler Instances                                 |
|   |              (no publish access)                               |
|   |                                                                |
+------------------------------------------------------------------+
```

## Two Handler Systems

The codebase has two distinct handler systems that require different treatment:

### Infrastructure Handlers

| Aspect | Description |
|--------|-------------|
| **Location** | `src/omnibase_infra/handlers/handler_*.py` |
| **Purpose** | Protocol/transport operations (HTTP, DB, Consul, Vault) |
| **Wiring** | Contract-driven via `HandlerPluginLoader` (preferred) or `wire_default_handlers()` (legacy fallback) |
| **State** | Contract-driven discovery implemented (PR #143). See [Migration Guide](../migration/MIGRATION_WIRE_DEFAULT_HANDLERS.md) |

**Current Handlers**:
- `HttpRestHandler` - HTTP/REST API operations
- `HandlerDb` - PostgreSQL database operations
- `HandlerConsul` - Service discovery
- `HandlerVault` - Secret management

### Node Handlers

| Aspect | Description |
|--------|-------------|
| **Location** | `src/omnibase_infra/nodes/*/handlers/` |
| **Purpose** | Event processing for nodes |
| **Wiring** | Contract YAML (`handler_routing` section) |
| **State** | Already contract-driven (ahead of plan) |

**Example** (from `node_registration_orchestrator/contract.yaml`):
```yaml
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler: "HandlerNodeIntrospected"
      output_events: ["ModelNodeRegistrationInitiated"]
```

## Enum Architecture

### Two Distinct Enums (Do Not Merge)

These represent **different axes** and must remain separate.

**EnumHandlerType and EnumHandlerTypeCategory are intentionally orthogonal and must both be specified on descriptors.**

#### EnumHandlerType (Architectural Role)

**Location**: `omnibase_core` (or rename existing in `omnibase_infra`)
**Purpose**: What is this handler in the architecture?

```python
class EnumHandlerType(str, Enum):
    """Handler architectural role - selects interface and lifecycle."""
    INFRA_HANDLER = "infra_handler"      # Protocol/transport handler
    NODE_HANDLER = "node_handler"         # Event processing handler
    PROJECTION_HANDLER = "projection_handler"  # Projection read/write
    COMPUTE_HANDLER = "compute_handler"   # Pure computation
```

**Drives**: Lifecycle, protocol selection, runtime invocation pattern

#### EnumHandlerTypeCategory (Behavioral Classification)

**Location**: `omnibase_core`
**Purpose**: How does this handler behave at runtime?

```python
class EnumHandlerTypeCategory(str, Enum):
    """Handler behavioral classification - selects policy envelope."""
    COMPUTE = "compute"                          # Pure, deterministic
    EFFECT = "effect"                            # Side-effecting I/O
    NONDETERMINISTIC_COMPUTE = "nondeterministic_compute"  # Pure but not deterministic
```

**Drives**: Security rules, determinism guarantees, replay safety, permissions

### Category Classification Guide

| Category | Deterministic? | Side Effects? | Examples |
|----------|----------------|---------------|----------|
| COMPUTE | Yes | No | Validation, transformation, mapping |
| EFFECT | N/A | Yes | DB, HTTP, Consul, Vault, Kafka, LLM calls |
| NONDETERMINISTIC_COMPUTE | No | No | UUID generation, `datetime.now()`, `random.choice()` |

**Note**: LLM API calls are **EFFECT** (external I/O), not NONDETERMINISTIC_COMPUTE.

### ADAPTER Tag (Policy Modifier, Not Category)

ADAPTER is **not** a behavioral category—it's a **policy tag** applied to EFFECT handlers that are platform plumbing.

```python
class ModelHandlerDescriptor(BaseModel):
    # ... other fields ...
    handler_type_category: EnumHandlerTypeCategory  # Always EFFECT for adapters
    is_adapter: bool = False                        # Policy modifier tag
```

**Why not a category?**
- Adapters perform I/O, so behaviorally they are EFFECT
- Replay semantics: I/O is I/O, regardless of whether it's "domain" or "transport"
- ADAPTER describes **policy intent** (stricter defaults), not **runtime behavior**

**When to use ADAPTER tag:**
- Kafka ingress/egress
- HTTP ingress gateway
- Webhook receiver
- CLI bridge

**When NOT to use ADAPTER tag:**
- DB handlers (domain persistence)
- Vault handlers (secret access)
- Consul handlers (service discovery)
- Outbound HTTP client (business API calls)

**ADAPTER tag enforces stricter defaults:**
- No secrets by default
- Narrower network permissions
- Tighter allowed outputs
- Enhanced observability requirements

### Current Handler Classifications

All existing infrastructure handlers are **EFFECT** (none are tagged ADAPTER):

| Handler | Category | ADAPTER Tag | Rationale |
|---------|----------|-------------|-----------|
| `handler_db` | EFFECT | No | Reads/writes domain persistence |
| `handler_http` | EFFECT | No | External HTTP calls with business semantics |
| `handler_consul` | EFFECT | No | Service discovery with external state |
| `handler_vault` | EFFECT | No | Secret access (definitely not plumbing) |

## Contract Architecture

### Distinct Contract Types (Do Not Unify)

#### ModelHandlerContract (Infrastructure Handlers)

**Location**: `omnibase_spi`
**Purpose**: Declare callable units invoked by nodes

```yaml
# handler_contract.yaml
contract_version: "1.0.0"
handler_identity:
  name: "http-rest-handler"
  version: "1.0.0"
handler_type: INFRA_HANDLER
handler_type_category: EFFECT
capabilities:
  - HTTP_GET
  - HTTP_POST
security:
  allowed_domains: ["api.example.com"]
  secret_scopes: []
# Explicitly forbidden:
# - execution_graph
# - dispatch_rules
# - publish_declarations
```

#### ModelNodeHandlerContract (Node Handlers)

**Location**: `omnibase_spi`
**Purpose**: Declare routing bindings owned by a node

```yaml
# In node contract.yaml
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model: "ModelNodeIntrospectionEvent"
      handler: "HandlerNodeIntrospected"
      output_events: ["ModelNodeRegistrationInitiated"]
```

### Shared Contract Components

Both contract types share embedded models:

- `ModelContractIdentity` - name, version, description
- `ModelContractPackaging` - import path, artifact ref
- `ModelContractSecurity` - allowed domains, secret scopes, classification

## Handler Descriptor Model

The canonical runtime representation of a handler:

```python
class ModelHandlerDescriptor(BaseModel):
    """Runtime handler descriptor - canonical representation."""

    # Identity
    handler_name: ModelIdentifier
    handler_version: ModelSemVer

    # Classification
    handler_type: EnumHandlerType
    handler_type_category: EnumHandlerTypeCategory

    # Policy Tags
    is_adapter: bool = False  # Platform plumbing - triggers stricter defaults

    # Surface
    capabilities: list[EnumHandlerCapability]
    commands_accepted: list[EnumHandlerCommandType]

    # Instantiation
    import_path: str | None = None
    artifact_ref: ModelArtifactRef | None = None

    # Metadata
    security_metadata_ref: ModelSecurityMetadataRef | None = None
    packaging_metadata_ref: ModelPackagingMetadataRef | None = None
```

**Key Principle**: Contract → Descriptor is a one-way transformation. Runtime never consumes contracts directly.

**ADAPTER Validation**: If `is_adapter=True`, runtime validates:
- `handler_type_category` must be EFFECT (adapters do I/O)
- Secret scopes rejected unless explicitly overridden
- Domain allowlist required (empty = no outbound allowed)

## Publish Enforcement

### Structural Constraint (Not Policy)

Handlers cannot publish by **construction**, not by policy:

1. Handler invocation context does NOT expose bus publisher
2. Handlers receive only:
   - Input payload
   - Narrow `HandlerInvocationContext`
   - Allowed IO clients (if EFFECT)
3. Any publish attempt must route through runtime-owned APIs that handlers cannot access

### Current State (Already Correct)

| Handler Type | Direct Bus Access | Can Emit Events |
|--------------|-------------------|-----------------|
| Infrastructure | None | Returns `ModelHandlerOutput` |
| Node | None | Returns `list[BaseModel]` (orchestrator publishes) |

## Error Experience

### Standardized Validation Error Model

All handler-related failures emit structured errors:

```python
class ModelHandlerValidationError(BaseModel):
    """Structured error for handler validation failures."""

    # Required
    error_type: EnumHandlerErrorType  # CONTRACT_PARSE_ERROR, SECURITY_VIOLATION, etc.
    rule_id: str                       # e.g., "SEC-ALLOWLIST-DOMAIN"
    handler_identity: ModelIdentifier
    source_type: EnumHandlerSourceType  # BOOTSTRAP | CONTRACT
    message: str
    remediation_hint: str

    # Optional
    file_path: str | None = None       # Required for contract-based failures
    details: dict[str, Any] | None = None
    caused_by: ModelHandlerValidationError | None = None
```

### Error Visibility Rules

| Timing | Failures | Behavior |
|--------|----------|----------|
| Startup | Parse errors, validation failures, architecture violations, expired bootstrap | Refuse to start, emit structured logs, aggregate summary |
| Invocation | Security violations dependent on runtime inputs | Abort invocation, emit structured error with correlation ID |

## Hybrid Mode Semantics

When `handler_source_mode = HYBRID`:

1. Load contract handlers into registry keyed by `handler_identity`
2. For each bootstrap handler:
   - If contract handler with same identity exists → skip bootstrap
   - Else → register bootstrap handler
3. Log fallback occurrences with structured fields

**Resolution is per-handler identity**, not whole-source failover.

## Production Hardening

### Bootstrap Emergency Override

```python
@dataclass
class HandlerSourceConfig:
    handler_source_mode: EnumHandlerSourceType = EnumHandlerSourceType.CONTRACT
    allow_bootstrap_override: bool = False
    bootstrap_expires_at: datetime | None = None
```

**Enforcement at startup**:
- If `mode = BOOTSTRAP` and `expires_at` is set and `now > expires_at`:
  - Refuse to start (or force CONTRACT mode)
- Always emit structured log with expiry status

---

## Ticket Summary

### Phase: MVP (8 tickets)

| # | Title | Repo |
|---|-------|------|
| 1 | ProtocolHandlerSource + EnumHandlerSourceType | omnibase_spi |
| 2 | Handler enums (EnumHandlerTypeCategory, capabilities, commands) | omnibase_core |
| 3 | ModelHandlerDescriptor (canonical runtime object) | omnibase_spi |
| 4 | HandlerBootstrapSource (descriptor-based) | omnibase_infra |
| 5 | Rename wiring + structured logs | omnibase_infra |
| 6 | Runtime uses ProtocolHandlerSource + no-publish enforcement | omnibase_infra |
| 7 | Categorize bootstrap handlers | omnibase_infra |
| 8 | Structured validation & error reporting | omnibase_infra |

### Phase: Beta (6 tickets)

| # | Title | Repo |
|---|-------|------|
| 9 | Source mode flag + hybrid semantics | omnibase_infra |
| 10 | ModelHandlerContract + YAML templates | omnibase_spi |
| 11 | HandlerContractSource + filesystem discovery | omnibase_infra |
| 12 | Security validation (registration + invocation) | omnibase_infra |
| 13 | Architecture validator (FSM rule clarified) | omnibase_infra |
| 14 | Placeholder: Registry-based discovery | omnibase_infra |

### Phase: Production (1 ticket)

| # | Title | Repo |
|---|-------|------|
| 15 | Contract-first + bootstrap expiry override | omnibase_infra |

### Integration Tickets

| # | Title | Repo | Phase |
|---|-------|------|-------|
| I1 | Reconcile EnumHandlerType (existing) with new enum architecture | omnibase_infra/core | MVP |
| I2 | Migrate infrastructure handler metadata to descriptor model | omnibase_infra | MVP |
| I3 | Test coverage: existing no-publish constraint | omnibase_infra | MVP |

## TDD Requirements

High-value tickets requiring test-first development:

| Ticket | TDD Requirement |
|--------|-----------------|
| 6 | Test handler invocation context cannot access publish API |
| 8 | Tests for structured error shape on each failure path |
| 12 | Tests for each security violation type |
| 13 | Tests for each architecture rule |

## Implementation Order

```
Phase 1 (Must merge first):
├── omnibase_core: Ticket 2 (enums)
└── omnibase_spi: Ticket 1, 3 (protocol, descriptor)

Phase 2 (Parallel with mocks):
├── Ticket 4: HandlerBootstrapSource (mock descriptors)
├── Ticket 6: Runtime + no-publish (mock ProtocolHandlerSource)
├── Ticket 11: HandlerContractSource (mock filesystem)
├── Ticket 12: Security validation (mock policy)
└── Ticket 13: Architecture validator (mock descriptors)

Phase 3 (Integration):
├── Ticket 9: Hybrid mode (real bootstrap + contract sources)
└── Ticket 15: Production hardening
```

## Non-Negotiable Guardrails

1. **No new stringly-typed identifiers** if typed equivalent exists
2. **Contract → Descriptor is one-way** - runtime never consumes contracts directly
3. **Handlers cannot publish by construction** - not by policy
4. **Hybrid mode resolves per-handler identity** - not whole-source failover
5. **All validation failures emit structured errors** with rule ID and remediation hint

## Related Documentation

- [MESSAGE_DISPATCH_ENGINE.md](MESSAGE_DISPATCH_ENGINE.md) - Dispatcher architecture
- [CURRENT_NODE_ARCHITECTURE.md](CURRENT_NODE_ARCHITECTURE.md) - Node patterns
- [container_dependency_injection.md](../patterns/container_dependency_injection.md) - DI patterns
- [error_handling_patterns.md](../patterns/error_handling_patterns.md) - Error patterns

## Changelog

- **2025-12-28**: ADAPTER redesigned as policy tag
  - ADAPTER removed from EnumHandlerTypeCategory enum
  - ADAPTER now a boolean tag (`is_adapter: bool`) on ModelHandlerDescriptor
  - Preserves correct replay semantics: adapters are EFFECT (they do I/O)
  - Stricter defaults enforced via tag validation, not category membership
- **2025-12-28**: Initial design document created
  - Consolidated from multi-iteration design review
  - Locked decisions on enum architecture, contract separation, publish enforcement
  - Defined 15 implementation tickets + 3 integration tickets
