# Session Handoff: ONEX Infrastructure Migration

**Date**: December 3, 2025
**Session Type**: Extended architectural planning and repository setup
**Working Directory**: `/Users/jonah/Code/omnibase_infra.bak` (with operations on new repo)
**Key Tools Used**: Linear MCP, parallel-solve, polymorphic agents

---

## Executive Summary

This session accomplished significant foundational work for the ONEX infrastructure migration. The primary outcome is a **major architectural pivot** from a 1-container-per-node model to a **Runtime Host model** that will dramatically improve resource efficiency and deployment simplicity.

Key accomplishments:
1. Fresh repository setup with proper directory structure
2. Linear ticket cleanup (11 tickets canceled due to migration)
3. Comprehensive architecture documentation
4. Critical architectural decision: Runtime Host model adoption

---

## Work Completed This Session

### 1. Repository Setup

Created fresh `/Users/jonah/Code/omnibase_infra/` directory (sibling to `omnibase_infra.bak`).

**Seeded with planning documents from backup**:
- `docs/MVP_EXECUTION_PLAN.md`
- `docs/HANDOFF_OMNIBASE_INFRA_MVP.md`
- `docs/DECLARATIVE_EFFECT_NODES_PLAN.md`

**Copied configuration files**:
- `CLAUDE.md` - Claude Code instructions and ONEX patterns
- `.claude/settings.local.json` - Local Claude settings
- `pyproject.toml` - Reference template for dependencies

**Created initial directory structure**:
```
src/omnibase_infra/
├── __init__.py
├── clients/
│   └── __init__.py
├── enums/
│   └── __init__.py
├── infrastructure/
│   └── __init__.py
├── models/
│   └── __init__.py
├── nodes/
│   └── __init__.py
├── shared/
│   └── __init__.py
└── utils/
    └── __init__.py
```

**Created tests/ structure**:
```
tests/
├── __init__.py
├── conftest.py
├── integration/
│   └── __init__.py
├── nodes/
│   └── __init__.py
└── unit/
    └── __init__.py
```

**Created documentation**:
- `README.md` - Repository overview

**Architectural Note**: Removed `adapters/` directory - adapters ARE effect nodes, they belong in `nodes/`. This enforces the ONEX principle that all external integrations are effect nodes with proper contracts.

---

### 2. Linear Ticket Cleanup

Canceled 11 tickets related to omnibase_infra, omniarchon, and omninode_bridge due to repository migration:

| Ticket | Title | Status |
|--------|-------|--------|
| OMN-145 | Fix Pydantic v2 PrivateAttr initialization | Canceled |
| OMN-132 | Promote declarative transformation models | Canceled |
| OMN-133 | Enforce node_ prefix naming convention | Canceled |
| OMN-131 | ONEX 4-Node Ingestion Pipeline | Canceled |
| OMN-119 | Agent Observability Dashboard Integration | Canceled |
| OMN-69 | Fix omnibase_core import structure | Canceled |
| OMN-53 | Phase 5: Intelligence Effect Adapter | Canceled |
| OMN-44 | Document Event Architecture | Canceled |
| OMN-38 | Add DLQ Routing to Publishers | Canceled |
| OMN-41 | Create DLQ Monitoring Service | Canceled |
| OMN-39 | Implement Secret Sanitization | Canceled |

**Cancellation Reason**: "Canceled due to repository migration and consolidation. This work is being superseded by the fresh ONEX-compliant repository rebuild."

---

### 3. Architecture Documentation

Created `docs/CURRENT_NODE_ARCHITECTURE.md` (927 lines) documenting:

- **Current 1-container-per-node architecture** - How nodes work today
- **Full file trees** for Vault Adapter and Consul Projector nodes
- **Contract YAML structure** - All required fields and patterns
- **Node base classes** - `NodeEffectService`, `NodeComputeService`, etc.
- **Import patterns** - From `omnibase_core` dependencies
- **Deployment model limitations** - Why the current model doesn't scale

This document serves as the authoritative reference for the pre-migration architecture.

---

### 4. Major Architectural Decision: Runtime Host Model

**THIS IS THE KEY CHANGE FROM THIS SESSION**

#### Problem Statement

The current architecture has significant limitations:
- **Resource waste**: Each node = 1 Docker container (~150MB memory each)
- **Connection explosion**: Each container maintains its own Kafka connections
- **Deployment complexity**: N nodes = N containers to deploy and manage
- **Startup overhead**: Each container goes through full initialization

#### Old Architecture (1-Container-Per-Node)

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Container 1   │  │   Container 2   │  │   Container 3   │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ vault     │  │  │  │ consul    │  │  │  │ postgres  │  │
│  │ adapter   │  │  │  │ projector │  │  │  │ adapter   │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
│  [Kafka conn]   │  │  [Kafka conn]   │  │  [Kafka conn]   │
│  [Entry point]  │  │  [Entry point]  │  │  [Entry point]  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

#### New Architecture (Runtime Host)

```
┌───────────────────────────────────────────────────────────┐
│                    NodeRuntime Host                       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Shared Kafka Connection                 │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ NodeInstance│  │ NodeInstance│  │ NodeInstance│       │
│  │   vault     │  │   consul    │  │   postgres  │       │
│  │   adapter   │  │   projector │  │   adapter   │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Handlers: [local] [http] [db] [llm] [kafka]         │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

#### Key Concepts

| Concept | Description |
|---------|-------------|
| `NodeRuntime` | Single process that hosts multiple node instances |
| `NodeInstance` | Lightweight wrapper around a node's business logic |
| `Handler` | Shared infrastructure (http, db, llm, kafka) |
| `FileRegistry` | Loads contracts from filesystem |
| `OnexEnvelope` | Unified message format for all node communication |

#### Benefits

1. **Resource efficiency**: Single Kafka connection shared by all nodes
2. **Memory reduction**: ~150MB total vs ~150MB per node
3. **Simpler deployment**: 1 container instead of N containers
4. **Faster startup**: Nodes are just instances, not full processes
5. **Easier testing**: Spin up runtime with multiple nodes in-process

---

## Repository Responsibilities

The Runtime Host model requires changes across multiple repositories:

| Component | Repository | Notes |
|-----------|------------|-------|
| `NodeRuntime` class | `omnibase_core` | Core runtime host implementation |
| `NodeInstance` class | `omnibase_core` | Node instance wrapper |
| `RuntimeHostContract` | `omnibase_core` or `omnibase_spi` | Contract schema for hosts |
| `OnexEnvelope` | `omnibase_core` | Unified message envelope |
| `NodeKind.RUNTIME_HOST` | `omnibase_core` | New enum value |
| `FileRegistry` | `omnibase_core` | Contract file loading |
| CLI entry point | `omnibase_core` | `omninode-runtime-host` command |
| Handler: `local` | `omnibase_core` | Echo/test handler (no external deps) |
| Handler: `http` | `omnibase_infra` | httpx-based HTTP handler |
| Handler: `db` | `omnibase_infra` | PostgreSQL handler |
| Handler: `llm` | `omnibase_infra` | LLM API handler |
| Handler contracts | `omnibase_infra` | Per-handler contracts |
| Infrastructure config | `omnibase_infra` | Kafka, Postgres config |

**Dependency Flow**: `core <- spi <- infra` (infra depends on spi, spi depends on core, core depends on nothing)

---

## Core Design Invariants

These invariants MUST be maintained throughout the ONEX architecture:

1. **All behavior is contract-driven** - No implicit behavior, everything defined in YAML contracts
2. **NodeRuntime is the only executable event loop** - Nodes don't run their own loops
3. **Node logic is pure: no I/O, no mixins, no inheritance** - Pure functions only
4. **Core never depends on SPI or infra** - Dependency flows one way: infra -> spi -> core
5. **SPI only defines protocols, never implementations** - Pure interfaces only
6. **Infra owns all I/O and real system integrations** - All external interactions in infra

**Critical Invariant**:
```
No code in omnibase_core may initiate network I/O, database I/O,
file I/O, or external process execution.
```

---

## Runtime Host Wiring

The runtime host consists of components from both core and infra:

```
RuntimeHostProcess (omnibase_infra)
    |-- NodeRuntime (omnibase_core)
          |-- NodeInstance(vault_adapter)
          |-- NodeInstance(user_query)
          |-- NodeInstance(echo)
          |-- handlers:
                 vault_handler: VaultHandler (infra)
                 db_handler: PostgresHandler (infra)
                 http_handler: HttpRestHandler (infra)
```

**Key Separation**:
- `NodeRuntime` -> Core logic (what the runtime **is**)
- `RuntimeHostProcess` -> Infra-level process container (where the runtime **runs**)
- Handler registry -> Infra's injection layer (how handlers are **wired**)

---

## Topic Naming Schema

Standardized Kafka topic naming for the Runtime Host model:

```
# Command topics (inbound to nodes)
onex.app.local.global.cmd.node.<node_slug>.v1

# Event topics (outbound from nodes)
onex.app.local.global.evt.node.<node_slug>.v1

# System log topics (runtime logs)
onex.sys.local.global.log.runtime.<runtime_slug>.v1

# Examples:
onex.app.local.global.cmd.node.vault-adapter.v1
onex.app.local.global.evt.node.vault-adapter.v1
onex.sys.local.global.log.runtime.infra-host-01.v1
```

---

## Phased Implementation Plan

### Phase 0: Core Types (omnibase_core)
- Add `NodeKind.RUNTIME_HOST` to enums
- Implement `OnexEnvelope` model
- Implement `RuntimeHostContract` model
- Define handler protocol interface

### Phase 1: Minimal Local Runtime
- Implement `NodeRuntime` class
- Implement `NodeInstance` wrapper
- Implement `FileRegistry` for contract loading
- Create `local_handler` (echo/test)
- Create `http_handler` (httpx-based)
- Add CLI: `omninode-runtime-host`
- **Validation**: End-to-end echo flow through Kafka

### Phase 2: First Real Integration
- Implement ONE of: `db_handler` OR `llm_handler`
- Full integration test with real external service
- Performance benchmarking vs old model

### Phase 3: Cloud Data Plane
- Local runtime connects to AWS infrastructure
- S3, DynamoDB, or other cloud handlers
- Production deployment patterns

### Phase 4: Multi-Host Scaling
- Multiple runtime hosts
- Load balancing across hosts
- Node affinity and routing

---

## Current State of New Repository

```
/Users/jonah/Code/omnibase_infra/
├── CLAUDE.md                              # Claude Code instructions
├── README.md                              # Repository overview
├── pyproject.toml                         # Python project config
├── .claude/
│   └── settings.local.json                # Local Claude settings
├── docs/
│   ├── CURRENT_NODE_ARCHITECTURE.md       # NEW - Pre-migration reference
│   ├── DECLARATIVE_EFFECT_NODES_PLAN.md   # Contract-driven nodes
│   ├── HANDOFF_OMNIBASE_INFRA_MVP.md      # Previous MVP handoff
│   ├── HANDOFF_SESSION_2025_12_03.md      # THIS DOCUMENT
│   └── MVP_EXECUTION_PLAN.md              # Original MVP plan
├── src/
│   ├── __init__.py
│   └── omnibase_infra/
│       ├── __init__.py
│       ├── clients/                       # External service clients
│       │   └── __init__.py
│       ├── enums/                         # Infrastructure enums
│       │   └── __init__.py
│       ├── infrastructure/                # Core infrastructure
│       │   └── __init__.py
│       ├── models/                        # Pydantic models
│       │   └── __init__.py
│       ├── nodes/                         # ONEX nodes (including adapters)
│       │   └── __init__.py
│       ├── shared/                        # Shared utilities
│       │   └── __init__.py
│       └── utils/                         # Helper utilities
│           └── __init__.py
└── tests/
    ├── __init__.py
    ├── conftest.py                        # Pytest configuration
    ├── integration/                       # Integration tests
    │   └── __init__.py
    ├── nodes/                             # Node-specific tests
    │   └── __init__.py
    └── unit/                              # Unit tests
        └── __init__.py
```

---

## Next Steps (Immediate Priority)

### 1. In omnibase_core (MUST BE DONE FIRST)

```python
# New enum value
class NodeKind(str, Enum):
    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"
    RUNTIME_HOST = "runtime_host"  # NEW

# OnexEnvelope model
class OnexEnvelope(BaseModel):
    envelope_id: UUID
    correlation_id: UUID
    node_slug: str
    handler_type: str  # "local", "http", "db", "llm"
    payload: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: datetime

# RuntimeHostContract model
class RuntimeHostContract(BaseModel):
    name: str
    version: str
    nodes: list[str]  # List of node contracts to load
    handlers: list[str]  # Available handlers

# NodeRuntime class
class NodeRuntime:
    def __init__(self, config: RuntimeHostConfig): ...
    def register_handler(self, handler: Handler): ...
    def load_nodes(self, contracts_dir: Path): ...
    async def start(self): ...
    async def stop(self): ...

# NodeInstance class
class NodeInstance:
    def __init__(self, contract: NodeContract, runtime: NodeRuntime): ...
    async def handle(self, envelope: OnexEnvelope) -> OnexEnvelope: ...

# FileRegistry class
class FileRegistry:
    def __init__(self, contracts_dir: Path): ...
    def load_all(self) -> list[NodeContract]: ...
    def get(self, slug: str) -> NodeContract: ...
```

### 2. In omnibase_infra

```
src/omnibase_infra/
├── handlers/                    # NEW - Handler implementations
│   ├── __init__.py
│   ├── local_handler.py         # Echo/test handler
│   ├── http_handler.py          # httpx-based HTTP handler
│   ├── db_handler.py            # PostgreSQL handler (Phase 2)
│   └── llm_handler.py           # LLM API handler (Phase 2)
├── contracts/                   # NEW - Example contracts
│   ├── runtime_host.yaml        # Host contract example
│   ├── vault_adapter.yaml       # Vault node contract
│   └── consul_projector.yaml    # Consul node contract
└── ... (existing structure)
```

### 3. Validation Checklist

- [ ] Echo flow: CLI -> Kafka -> Runtime -> local_handler -> Kafka -> response
- [ ] HTTP flow: CLI -> Kafka -> Runtime -> http_handler -> external API -> response
- [ ] Multi-node: Single runtime hosting 3+ node instances
- [ ] Performance: Memory usage < 200MB for 10 nodes
- [ ] Contract loading: FileRegistry loads all contracts from directory

---

## Open Questions

### 1. RuntimeHostContract Location
**Question**: Where does `RuntimeHostContract` live - `omnibase_core` or `omnibase_spi`?
**Recommendation**: Start in `omnibase_core` since it's tightly coupled to `NodeRuntime`. Move to SPI later if needed.

### 2. Local Handler Location
**Decision**: `local_handler` (echo) belongs in `omnibase_core`.
**Rationale**:
- Essential for testing and development workflows
- Has zero external dependencies
- Core is the right place for pure testing utilities
- Aligns with the invariant that core contains no I/O (echo is in-memory only)

### 3. Handler Registration
**Question**: How do handlers register themselves with the runtime?
**Options**:
- **Explicit registration**: Runtime config lists handlers to load
- **Auto-discovery**: Scan package for Handler implementations
- **Plugin system**: Entry points in pyproject.toml

**Recommendation**: Start with explicit registration for simplicity. Add auto-discovery later.

### 4. Envelope Versioning
**Question**: How do we version `OnexEnvelope` for backwards compatibility?
**Recommendation**: Add `envelope_version` field. Runtime rejects unknown versions.

---

## Reference Documents

| Document | Path | Description |
|----------|------|-------------|
| MVP Execution Plan | `/Users/jonah/Code/omnibase_infra/docs/MVP_EXECUTION_PLAN.md` | Original MVP plan (needs update for Runtime Host) |
| Current Architecture | `/Users/jonah/Code/omnibase_infra/docs/CURRENT_NODE_ARCHITECTURE.md` | Pre-migration reference |
| Declarative Effects | `/Users/jonah/Code/omnibase_infra/docs/DECLARATIVE_EFFECT_NODES_PLAN.md` | Contract-driven effect nodes |
| Previous Handoff | `/Users/jonah/Code/omnibase_infra/docs/HANDOFF_OMNIBASE_INFRA_MVP.md` | Prior session handoff |
| Backup Repository | `/Users/jonah/Code/omnibase_infra.bak/` | Old repository for reference |

---

## Session Metadata

| Field | Value |
|-------|-------|
| Date | December 3, 2025 |
| Duration | Extended session |
| Primary Working Directory | `/Users/jonah/Code/omnibase_infra.bak` |
| Target Repository | `/Users/jonah/Code/omnibase_infra` |
| Linear Tickets Modified | 11 canceled |
| Documents Created | 2 (CURRENT_NODE_ARCHITECTURE.md, this handoff) |
| Architectural Decisions | 1 major (Runtime Host model) |

---

## Glossary

| Term | Definition |
|------|------------|
| **NodeRuntime** | Single process that hosts multiple NodeInstances |
| **NodeInstance** | Lightweight wrapper around node business logic |
| **Handler** | Shared infrastructure component (http, db, llm, kafka) |
| **OnexEnvelope** | Unified message format for all node communication |
| **FileRegistry** | Loads node contracts from filesystem |
| **Effect Node** | Node that performs external I/O (adapters are effect nodes) |
| **Contract** | YAML specification defining a node's interface and behavior |

---

## Appendix: Why Runtime Host?

### Before (10 nodes)
- 10 Docker containers
- 10 Kafka connections
- ~1.5GB memory
- 10 separate deployments
- 10 health checks
- 10 log streams

### After (10 nodes)
- 1 Docker container
- 1 Kafka connection
- ~200MB memory
- 1 deployment
- 1 health check (with per-node status)
- 1 log stream (with node context)

**Bottom line**: The Runtime Host model is essential for scaling ONEX to production workloads with hundreds of nodes.

---

*End of Handoff Document*
