> **Navigation**: [Home](../index.md) > [Design](README.md) > Handler Protocol Ticket Plan

# Handler Protocol-Driven Architecture: Ticket Plan

## Purpose

This document contains the complete, agent-consumable ticket specifications for migrating
from hardcoded handler wiring to protocol-driven architecture.

**Parent Document**: [HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md](../architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md)
**Status**: Ready for Linear import

---

## Legend

- **Phase**:
  - MVP: Required immediately, structural seams without behavior change
  - Beta: Enables extensibility with enforcement
  - Production: Hardening and lock-down
- **Repo**:
  - omnibase_core
  - omnibase_infra
  - omnibase_spi

---

## MVP Tickets

### Ticket 1 — Define ProtocolHandlerSource + EnumHandlerSourceType

- **Phase**: MVP
- **Repo**: omnibase_spi
- **Scope**:
  - Add `ProtocolHandlerSource`:
    - `discover_handlers() -> list[ModelHandlerDescriptor]`
    - `source_type: EnumHandlerSourceType`
  - Add `EnumHandlerSourceType`:
    - BOOTSTRAP
    - CONTRACT
    - HYBRID
- **Notes**:
  - Protocol only, no runtime logic
  - Runtime must not branch on concrete source type
- **TDD**: N/A (structural)

---

### Ticket 2 — Add Handler Enums (omnibase_core)

- **Phase**: MVP
- **Repo**: omnibase_core
- **Scope**:
  - Add `EnumHandlerTypeCategory`:
    - COMPUTE - Pure, deterministic
    - EFFECT - Side-effecting I/O
    - NONDETERMINISTIC_COMPUTE - Pure but not deterministic
  - Add `EnumHandlerCapability` (initial controlled set) or `ModelIdentifier` + namespace enum
  - Add `EnumHandlerCommandType` (if no existing typed command identifiers exist)
  - Document vocabulary ownership and extension rules
- **Notes**:
  - Reuse existing typed command models if they already exist
  - Strings acceptable only as serialized forms, not runtime fields
  - ADAPTER is NOT a category—it's a policy tag (see Ticket 3)
- **TDD**: N/A (structural)

---

### Ticket 3 — Create ModelHandlerDescriptor (Canonical Runtime Object)

- **Phase**: MVP
- **Repo**: omnibase_spi
- **Scope**:
  - Add `ModelHandlerDescriptor` with **fully defined fields**:

  **Identity**
  - `handler_name: ModelIdentifier | EnumHandlerName`
  - `handler_version: ModelSemVer`

  **Classification**
  - `handler_type: EnumHandlerType`
  - `handler_type_category: EnumHandlerTypeCategory`

  **Policy Tags**
  - `is_adapter: bool = False` - Platform plumbing flag, triggers stricter defaults

  **Surface**
  - `capabilities: list[EnumHandlerCapability] | list[ModelIdentifier]`
  - `commands_accepted: list[EnumHandlerCommandType] | list[ModelIdentifier]`

  **Instantiation**
  - `import_path: str` OR `artifact_ref: ModelArtifactRef`

  **Optional Metadata**
  - `security_metadata_ref: ModelSecurityMetadataRef | None`
  - `packaging_metadata_ref: ModelPackagingMetadataRef | None`

  **Documentation**
  - Explicitly document: Descriptor is canonical runtime representation
  - Contracts are serialization format transformed into descriptors
  - ADAPTER is a policy tag, not a category:
    - Behaviorally, adapters are EFFECT (they do I/O)
    - `is_adapter=True` triggers stricter defaults (no secrets, narrow permissions)
    - Use for: Kafka ingress/egress, HTTP gateway, webhook, CLI bridge
    - Do NOT use for: DB, Vault, Consul, outbound HTTP client
- **TDD**: N/A (structural)

---

### Ticket 4 — Implement HandlerBootstrapSource (Descriptor-Based)

- **Phase**: MVP
- **Repo**: omnibase_infra
- **Scope**:
  - Implement `HandlerBootstrapSource : ProtocolHandlerSource`
  - Centralize all hardcoded handlers here
  - Each handler registered as a `ModelHandlerDescriptor`
  - No other module may instantiate handlers directly
- **Blocked By**: Tickets 1, 2, 3
- **TDD**: N/A (wiring)

---

### Ticket 5 — Rename Wiring + Structured Startup Logs

- **Phase**: MVP
- **Repo**: omnibase_infra
- **Scope**:
  - Rename `wire_default_handlers()` → `wire_bootstrap_handlers()`
  - Emit structured log at startup:
    ```json
    {
      "handler_source_mode": "BOOTSTRAP",
      "handler_count": 4,
      "compat_mode": true
    }
    ```
- **Files**: `src/omnibase_infra/runtime/wiring.py`
- **TDD**: N/A (cosmetic)

---

### Ticket 6 — Runtime Uses ProtocolHandlerSource + Structural No-Publish Enforcement

- **Phase**: MVP
- **Repo**: omnibase_infra
- **Scope**:
  - `RuntimeHostProcess` accepts `handler_source: ProtocolHandlerSource`
  - Default source: `HandlerBootstrapSource`
  - Runtime registers descriptors returned by `discover_handlers()`
  - Handler invocation context:
    - Does NOT expose any bus publisher
    - Cannot access runtime publish APIs
  - Add test proving handler invocation cannot publish
- **Blocked By**: Tickets 1, 4
- **TDD Requirements**:
  - [ ] RED: Test that `HandlerInvocationContext` has no `publish` method or attribute
  - [ ] RED: Test that calling any bus API from handler call stack raises `ArchitectureViolationError`
  - [ ] GREEN: Implement invocation context without publisher access
  - [ ] GREEN: Add runtime guard if publish attempted

---

### Ticket 7 — Categorize Bootstrap Handlers

- **Phase**: MVP
- **Repo**: omnibase_infra
- **Scope**:
  - Ensure every bootstrap handler descriptor sets:
    - `handler_type_category`
  - All current IO handlers must be `EFFECT`:
    - `handler_db` → EFFECT
    - `handler_http` → EFFECT
    - `handler_consul` → EFFECT
    - `handler_vault` → EFFECT
- **Blocked By**: Ticket 2, 4
- **TDD**: N/A (annotation)

---

### Ticket 8 — Structured Validation & Error Reporting for Handlers

- **Phase**: MVP
- **Repo**: omnibase_infra
- **Scope**:
  - Define canonical handler validation error model:
    ```python
    class ModelHandlerValidationError(BaseModel):
        error_type: EnumHandlerErrorType
        rule_id: str
        handler_identity: ModelIdentifier
        source_type: EnumHandlerSourceType
        message: str
        remediation_hint: str
        file_path: str | None = None
        details: dict[str, Any] | None = None
        caused_by: ModelHandlerValidationError | None = None
    ```
  - Ensure all handler-related validation paths emit structured errors:
    - contract parsing
    - descriptor validation
    - security validation
    - architecture validation
  - Include remediation hints in all errors
  - Ensure startup aggregates and prints validation failures clearly
- **Non-Goals**: No fancy UI, no retries, no suppression
- **TDD Requirements**:
  - [ ] RED: Test that contract parse error produces structured error with rule_id
  - [ ] RED: Test that security violation produces structured error with remediation_hint
  - [ ] GREEN: Implement error model and emission

---

## Beta Tickets

### Ticket 9 — Handler Source Mode Feature Flag (bootstrap | contract | hybrid)

- **Phase**: Beta
- **Repo**: omnibase_infra
- **Scope**:
  - Config: `handler_source_mode: EnumHandlerSourceType`
  - HYBRID semantics are **per-handler identity**:
    - Contract handler overrides bootstrap if identity matches
    - Bootstrap used only when contract missing
  - Structured logs include:
    - `contract_handler_count`
    - `bootstrap_handler_count`
    - `fallback_handler_count`
- **TDD**:
  - [ ] RED: Test hybrid resolution: contract wins over bootstrap by identity
  - [ ] RED: Test fallback when contract missing
  - [ ] GREEN: Implement per-handler resolution

---

### Ticket 10 — Define ModelHandlerContract + YAML Templates

- **Phase**: Beta
- **Repo**: omnibase_spi
- **Scope**:
  - Add `ModelHandlerContract` (declaration-only)
  - YAML templates:
    - `handler_contract.yaml`
    - default compute template
    - default effect template
    - default effect+adapter template (EFFECT with `is_adapter: true`)
  - Explicitly forbid:
    - execution graphs
    - dispatch rules
    - publish declarations
  - Document: Contract → Descriptor is one-way
  - When `is_adapter: true` in contract, validation enforces:
    - `handler_type_category` must be EFFECT
    - `secret_scopes: []` (no secrets by default, unless explicit override)
    - `allowed_domains` required (empty = no outbound allowed)
    - `observability_level: "enhanced"`
- **TDD**: N/A (schema definition)

---

### Ticket 11 — Implement HandlerContractSource + Filesystem Discovery

- **Phase**: Beta
- **Repo**: omnibase_infra
- **Scope**:
  - Implement `HandlerContractSource : ProtocolHandlerSource`
  - Accept `contract_paths: list[Path]`
  - Recursive scan for `**/handler_contract.yaml`
  - Parse → validate → transform into `ModelHandlerDescriptor`
  - Structured logs:
    - `discovered_contract_count`
    - `validation_failure_count`
- **TDD**:
  - [ ] RED: Test discovery finds nested contracts
  - [ ] RED: Test discovery ignores malformed files (with structured error)
  - [ ] GREEN: Implement filesystem scan and parsing

---

### Ticket 12 — Security Validation (Registration-Time + Invocation-Time)

- **Phase**: Beta
- **Repo**: omnibase_infra
- **Scope**:
  - Registration-time validation:
    - secret scopes declared and permitted by environment policy
    - allowed domains declared
    - data classification vs environment policy
  - Invocation-time enforcement:
    - outbound domain allowlist
    - secret scope access
    - classification constraints
  - ADAPTER tag validation (when `is_adapter=True`):
    - Validate `handler_type_category` is EFFECT (adapters do I/O)
    - Reject secret scopes unless explicitly overridden
    - Require explicit domain allowlist (empty = no outbound allowed)
    - Enforce enhanced observability configuration
- **TDD Requirements**:
  - [ ] RED: Test secret scope violation at registration
  - [ ] RED: Test domain violation at invocation
  - [ ] RED: Test classification constraint enforcement
  - [ ] RED: Test `is_adapter=True` handler rejected when requesting secrets
  - [ ] RED: Test `is_adapter=True` with non-EFFECT category raises validation error
  - [ ] GREEN: Implement two-layer validation with adapter tag rules

---

### Ticket 13 — Architecture Validator (FSM Rule Clarified)

- **Phase**: Beta
- **Repo**: omnibase_infra
- **Scope**:
  - Validate:
    - no direct handler dispatch bypassing runtime
    - no handler publishing
    - no workflow FSM logic in orchestrators that duplicates reducer-governed transitions
  - Explicitly allow:
    - reducer-driven aggregate state machines
- **TDD Requirements**:
  - [ ] RED: Test that direct handler dispatch raises ArchitectureViolationError
  - [ ] RED: Test that handler publishing raises ArchitectureViolationError
  - [ ] RED: Test that workflow FSM in orchestrator raises ArchitectureViolationError
  - [ ] GREEN: Implement validator with rule IDs

---

### Ticket 14 — Placeholder: Registry-Based Handler Contract Discovery

- **Phase**: Beta (Phase 2)
- **Repo**: omnibase_infra
- **Scope**:
  - Optional alternative discovery mechanism:
    - fetch contract refs from registry
    - resolve artifacts
    - load via `HandlerContractSource`
  - Kept separate to avoid scope creep
- **Notes**: Deferred, no implementation in initial Beta

---

## Production Tickets

### Ticket 15 — Production Hardening: Contract-First + Bootstrap Emergency Override

- **Phase**: Production
- **Repo**: omnibase_infra
- **Scope**:
  - Default: `handler_source_mode = CONTRACT`
  - Allow emergency BOOTSTRAP override only if:
    - `allow_bootstrap_override = true`
    - optional `bootstrap_expires_at` provided
  - Enforce expiry at startup:
    - if expired, refuse to start in bootstrap mode (or force CONTRACT mode)
  - Structured logs include:
    - `bootstrap_expires_at`
    - `bootstrap_expired`
- **TDD**:
  - [ ] RED: Test expired bootstrap override refuses to start
  - [ ] GREEN: Implement startup expiry gate

---

## Integration Tickets

### Ticket I1 — Reconcile EnumHandlerType with New Enum Architecture

- **Phase**: MVP (blocks Ticket 2)
- **Repo**: omnibase_infra → omnibase_core
- **Scope**:
  - Existing `EnumHandlerType` has: EFFECT, COMPUTE, REDUCER, ORCHESTRATOR
  - This conflates architectural role with behavioral category
  - Decision: Split into two enums:
    - `EnumHandlerType` (architectural): INFRA_HANDLER, NODE_HANDLER, PROJECTION_HANDLER, COMPUTE_HANDLER
    - `EnumHandlerTypeCategory` (behavioral): COMPUTE, EFFECT, NONDETERMINISTIC_COMPUTE
  - ADAPTER is a policy tag (`is_adapter: bool`), NOT a category
  - Migrate existing enum or rename to avoid confusion
  - Both enums are orthogonal and must both be specified on descriptors
- **Notes**:
  - May require coordinated changes across repos
  - ADAPTER tag triggers stricter defaults but is behaviorally EFFECT

---

### Ticket I2 — Migrate Infrastructure Handler Metadata to Descriptor Model

- **Phase**: MVP (after Ticket 3)
- **Repo**: omnibase_infra
- **Scope**:
  - Current handlers have: `handler_type` property, `describe()` method
  - Map these to `ModelHandlerDescriptor` fields
  - Update `_KNOWN_HANDLERS` dict to produce descriptors
  - Ensure `HandlerBootstrapSource.discover_handlers()` returns proper descriptors
- **Files**:
  - `src/omnibase_infra/handlers/handler_http.py`
  - `src/omnibase_infra/handlers/handler_db.py`
  - `src/omnibase_infra/handlers/handler_consul.py`
  - `src/omnibase_infra/handlers/handler_vault.py`
  - `src/omnibase_infra/runtime/wiring.py`

---

### Ticket I3 — Test Coverage: Existing No-Publish Constraint

- **Phase**: MVP
- **Repo**: omnibase_infra
- **Scope**:
  - The constraint already exists (handlers return results, don't publish)
  - Add explicit tests proving:
    - `HttpRestHandler` cannot access bus
    - `HandlerNodeIntrospected` cannot access bus (only returns events)
  - This validates Ticket 6's requirement is already met
- **TDD**: Yes (tests prove existing behavior)

---

## Dependency Graph

```
omnibase_core:
├── Ticket 2 (enums) ─────────────────────────────────────┐
└── Ticket I1 (reconcile existing enum) ──────────────────┤
                                                          │
omnibase_spi:                                             │
├── Ticket 1 (ProtocolHandlerSource) ◄────────────────────┤
├── Ticket 3 (ModelHandlerDescriptor) ◄───────────────────┘
└── Ticket 10 (ModelHandlerContract) ◄── Beta

omnibase_infra:
├── Ticket 4 (HandlerBootstrapSource) ◄── Tickets 1, 2, 3
├── Ticket 5 (Rename wiring) ◄── Ticket 4
├── Ticket 6 (Runtime + no-publish) ◄── Tickets 1, 4
├── Ticket 7 (Categorize handlers) ◄── Tickets 2, 4
├── Ticket 8 (Error reporting) ◄── Ticket 3
├── Ticket I2 (Migrate metadata) ◄── Ticket 3
├── Ticket I3 (Test coverage) ◄── none
│
├── Ticket 9 (Source mode flag) ◄── Tickets 4, 11 (Beta)
├── Ticket 11 (HandlerContractSource) ◄── Ticket 10 (Beta)
├── Ticket 12 (Security validation) ◄── Ticket 11 (Beta)
├── Ticket 13 (Architecture validator) ◄── Ticket 8 (Beta)
├── Ticket 14 (Registry discovery) ◄── Ticket 11 (Beta, deferred)
│
└── Ticket 15 (Production hardening) ◄── Ticket 9 (Production)
```

---

## Agent Notes (Non-Negotiable)

1. **No new stringly-typed identifiers** if typed equivalent exists
2. **Contract → Descriptor is one-way** - runtime never consumes contracts directly
3. **Handlers cannot publish by construction** - not by policy
4. **Hybrid mode resolves per-handler identity** - not whole-source failover
5. **All validation failures emit structured errors** with rule_id and remediation_hint
6. **Do not emit plain-string errors** for handler validation paths
7. **Every failure must identify**: what rule failed, which handler, what file (if applicable), how to fix it
8. **ADAPTER is a policy tag, not a category** - handlers with `is_adapter=True` are behaviorally EFFECT with stricter defaults
9. **EnumHandlerType and EnumHandlerTypeCategory are orthogonal** - both must be specified on every descriptor

---

## Related Documentation

- **[Migration Guide: wire_default_handlers()](../migration/MIGRATION_WIRE_DEFAULT_HANDLERS.md)** - Step-by-step migration from legacy wiring
- **[Handler Plugin Loader Pattern](../patterns/handler_plugin_loader.md)** - Contract-driven handler discovery pattern

---

## Changelog

- **2026-01-13**: Contract-based handler discovery implemented (PR #143)
  - `HandlerPluginLoader` and `ContractHandlerDiscovery` implemented
  - `RuntimeHostProcess` accepts `contract_paths` parameter
  - Migration guide created at `docs/migration/MIGRATION_WIRE_DEFAULT_HANDLERS.md`
  - Legacy `wire_default_handlers()` available as fallback (to be removed)
- **2025-12-28**: ADAPTER redesigned as policy tag
  - ADAPTER removed from EnumHandlerTypeCategory
  - ADAPTER now a boolean tag (`is_adapter: bool`) on descriptor
  - Preserves correct replay semantics: adapters are EFFECT (they do I/O)
  - Stricter defaults enforced via tag validation, not category
  - Updated Tickets 2, 3, 10, 12, I1 to reflect tag-based design
- **2025-12-28**: Initial ticket plan created
  - 15 implementation tickets + 3 integration tickets
  - TDD requirements specified for high-value tickets
  - Dependency graph documented
