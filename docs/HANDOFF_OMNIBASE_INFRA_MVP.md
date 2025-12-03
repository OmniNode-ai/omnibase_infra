# omnibase_infra MVP Handoff Document

**Date**: December 2, 2025
**Status**: Planning Complete, Ready for Execution
**Estimated MVP Timeline**: 2-3 weeks (realistic)
**Session ID**: da3112e3-8682-428f-81ae-39616e41bd57

---

## Executive Summary

This document captures the complete analysis and planning for rebuilding `omnibase_infra` as a fresh, ONEX-compliant repository using `omnibase_core 0.3.5` and `omnibase_spi 0.2.0` (PyPI releases).

### Key Decision

**Approach**: Fresh start with reference repositories

1. Move current `omnibase_infra` to `omnibase_infra_bak`
2. Create new `omnibase_infra` repository
3. Use backup AND `omninode_bridge` as references
4. Extract only valuable business logic, contracts, and models
5. Follow `omniintelligence` patterns exactly

### Why Fresh Start vs Refactor

| Factor | Refactor Existing | Fresh Start |
|--------|-------------------|-------------|
| Import Path Fixes | 300+ files, complex | Built correct from start |
| Scaffolding Rewrite | 100% needed anyway | New scaffolding |
| Technical Debt | Carried forward | Clean slate |
| Time Estimate | 3-4 weeks | 2-3 weeks |
| Risk | High (unknown breaks) | Low (known patterns) |

---

## Current State Analysis

### omnibase_infra (Backup)

| Metric | Value |
|--------|-------|
| Total Python Files | 304 |
| Total YAML Contracts | 23 |
| Active Nodes | 11 (1 stub-only) |
| Total Node Code Lines | ~11,610 |
| Shared Models | 196 files, ~17,336 LOC |
| Infrastructure Utils | ~2,743 LOC |
| Active Tests | 0 (68K archived) |
| ONEX Compliance | 0% (wrong import paths) |

**Problem**: All nodes use import paths from `omnibase_core v0.1.0` which don't exist in v0.3.5:
```python
# WRONG (v0.1.0 paths - don't exist)
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.base.node_compute_service import NodeComputeService

# CORRECT (v0.3.5 paths)
from omnibase_core.nodes import NodeEffect, NodeCompute, NodeReducer, NodeOrchestrator
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.enums.enum_onex_error_code import EnumOnexErrorCode
```

### omninode_bridge (Reference)

| Metric | Value |
|--------|-------|
| Total LOC | ~260,000 |
| MVP Status | 85% complete |
| Infra Components | ~95,000 LOC designated |
| SQL Migrations | 49 files (40+ for infra) |
| Status | Being fully decomposed (will be deleted) |

**Valuable for omnibase_infra**:
- SQL migrations (40+ files, 25+ tables)
- Security validators (SQL injection protection)
- Metadata stamping service (21K LOC) - if needed

### omniintelligence (Pattern Reference)

| Metric | Value |
|--------|-------|
| Total LOC | 211,146 |
| Nodes | 21 (7 effect, 7 compute, 1 reducer, 1 orchestrator) |
| omnibase-core Version | 0.3.4 (PyPI) |
| Test Files | 12 |
| Pattern Compliance | 100% |

**Follow this repository's patterns exactly.**

---

## Target Architecture

### Repository Structure

```
omnibase_infra/                    # NEW - Fresh repository
├── pyproject.toml                 # PyPI deps: core ^0.3.5, spi ^0.2.0
├── src/omnibase_infra/
│   ├── __init__.py
│   ├── adapters/                  # Thin external service wrappers
│   ├── clients/                   # Service clients
│   ├── contracts/                 # Top-level contracts
│   ├── enums/                     # Centralized enums
│   │   ├── __init__.py
│   │   ├── enum_infra_fsm_type.py
│   │   ├── enum_operation_type.py
│   │   └── enum_service_status.py
│   ├── events/                    # Event infrastructure
│   │   ├── publisher/
│   │   └── models/
│   ├── models/                    # Centralized Pydantic models (CRITICAL)
│   │   ├── __init__.py            # Comprehensive exports
│   │   ├── model_postgres_*.py
│   │   ├── model_vault_*.py
│   │   ├── model_consul_*.py
│   │   ├── model_kafka_*.py
│   │   └── model_infrastructure_*.py
│   ├── nodes/                     # ONEX nodes
│   │   ├── node_postgres_adapter_effect/v1_0_0/
│   │   │   ├── contracts/
│   │   │   │   └── effect_contract.yaml
│   │   │   ├── effect.py
│   │   │   ├── __init__.py
│   │   │   └── __main__.py
│   │   ├── node_vault_adapter_effect/v1_0_0/
│   │   ├── node_consul_adapter_effect/v1_0_0/
│   │   ├── node_keycloak_adapter_effect/v1_0_0/
│   │   ├── node_webhook_effect/v1_0_0/
│   │   ├── node_circuit_breaker_compute/v1_0_0/
│   │   ├── node_tracing_compute/v1_0_0/
│   │   ├── infra_reducer/v1_0_0/
│   │   │   ├── contracts/
│   │   │   │   └── fsm_infrastructure_state.yaml
│   │   │   ├── reducer.py         # Uses MixinFSMExecution
│   │   │   └── __init__.py
│   │   └── infra_orchestrator/v1_0_0/
│   │       ├── contracts/
│   │       │   └── workflow_deployment.yaml
│   │       ├── orchestrator.py    # Inherits NodeOrchestratorDeclarative
│   │       └── __init__.py
│   ├── infrastructure/            # Utilities (from backup)
│   │   └── postgres_connection_manager.py
│   ├── shared/                    # Shared utilities
│   └── utils/
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── nodes/
└── docs/
```

### Node Inventory (Target)

| Node | Type | Suffix | Base Class | Priority |
|------|------|--------|------------|----------|
| `node_postgres_adapter_effect` | EFFECT | `_effect` | `NodeEffect` | P0 |
| `node_vault_adapter_effect` | EFFECT | `_effect` | `NodeEffect` | P1 |
| `node_consul_adapter_effect` | EFFECT | `_effect` | `NodeEffect` | P1 |
| `node_keycloak_adapter_effect` | EFFECT | `_effect` | `NodeEffect` | P2 |
| `node_webhook_effect` | EFFECT | `_effect` | `NodeEffect` | P2 |
| `node_circuit_breaker_compute` | COMPUTE | `_compute` | `NodeCompute` | P2 |
| `node_tracing_compute` | COMPUTE | `_compute` | `NodeCompute` | P3 |
| `infra_reducer` | REDUCER | (none) | `MixinFSMExecution` | P1 |
| `infra_orchestrator` | ORCHESTRATOR | (none) | `NodeOrchestratorDeclarative` | P1 |

### Nodes NOT to Include

| Node | Reason |
|------|--------|
| `kafka_adapter` | Use event bus mixins from core |
| `consul_projector` | Stub implementation, projectors go in reducer |
| `omni_infra_reducer` | Replace with declarative version |
| `omni_infra_orchestrator` | Replace with declarative version |

---

## Dependencies

### pyproject.toml Configuration

```toml
[tool.poetry.dependencies]
python = "^3.12"

# ONEX dependencies - PyPI releases (CRITICAL)
omnibase-core = "^0.3.5"
omnibase-spi = "^0.2.0"

# Core dependencies
pydantic = "^2.11.7"
fastapi = "^0.115.0"
uvicorn = "^0.32.0"

# Database
asyncpg = "^0.29.0"
psycopg2-binary = "^2.9.10"

# Service integration
python-consul = "^1.1.0"
hvac = "^2.1.0"  # Vault

# Observability
structlog = "^23.2.0"
prometheus-client = "^0.19.0"
opentelemetry-api = "^1.27.0"
opentelemetry-sdk = "^1.27.0"

# Resilience
tenacity = "^9.0.0"
circuitbreaker = "^2.0.0"
```

**REMOVED** (not needed):
- `aiokafka` - Use event bus mixins
- `confluent-kafka` - Use event bus mixins
- `redis` - Not needed for infra

---

## Declarative Node Patterns

### Reducer Pattern (FSM-Driven)

```python
# infra_reducer/v1_0_0/reducer.py
from omnibase_core.mixins.mixin_fsm_execution import MixinFSMExecution
from omnibase_core.models.contracts.subcontracts.model_fsm_subcontract import ModelFSMSubcontract
from omnibase_core.utils.util_safe_yaml_loader import load_and_validate_yaml_model

class InfraReducer(MixinFSMExecution):
    """Infrastructure reducer using declarative FSM pattern."""

    def __init__(self, config: ModelReducerConfig) -> None:
        super().__init__()
        self._fsm_contracts = self._load_fsm_contracts()

    def _load_fsm_contracts(self) -> dict[EnumInfraFSMType, ModelFSMSubcontract]:
        return {
            EnumInfraFSMType.INFRASTRUCTURE_STATE: load_and_validate_yaml_model(
                _CONTRACTS_DIR / "fsm_infrastructure_state.yaml",
                ModelFSMSubcontract,
            ),
        }
```

### Orchestrator Pattern (Workflow-Driven)

```python
# infra_orchestrator/v1_0_0/orchestrator.py
from omnibase_core.nodes.node_orchestrator_declarative import NodeOrchestratorDeclarative
from omnibase_core.models.container.model_onex_container import ModelONEXContainer

class InfraOrchestrator(NodeOrchestratorDeclarative):
    """Infrastructure orchestrator using declarative workflow pattern."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
        self._workflow_cache: dict[EnumOperationType, ModelWorkflowDefinition] = {}
```

---

## Reusable Assets from Backup

### High Value (Copy with Minor Fixes)

| Asset | Location | LOC | Changes Needed |
|-------|----------|-----|----------------|
| PostgresConnectionManager | `infrastructure/postgres/` | 634 | Import fixes |
| Shared Models (Pydantic) | `models/*/` | 17,336 | Consolidate to central dir |
| Contracts (YAML) | `nodes/*/contract.yaml` | 5,663 | Restructure format |
| SQL Sanitizer | `nodes/node_distributed_tracing_compute/utils/` | ~200 | Copy as utility |
| CircuitBreakerFactory | `infrastructure/resilience/` | 550 | Import fixes |

### Business Logic to Extract

| Node | Logic Location | Approx LOC | Extract |
|------|----------------|------------|---------|
| postgres_adapter | `nodes/postgres_adapter/v1_0_0/node.py` | 1,689 | SQL validation, circuit breaker integration |
| vault_adapter | `nodes/node_vault_adapter_effect/v1_0_0/node.py` | 705 | Secret lifecycle, mock client |
| consul_adapter | `nodes/consul_adapter/v1_0_0/node.py` | 983 | Service discovery, KV operations |
| keycloak_adapter | `nodes/node_keycloak_adapter_effect/v1_0_0/node.py` | 799 | JWT handling, role management |
| hook_node | `nodes/hook_node/v1_0_0/node.py` | 1,450 | Webhook delivery, retry logic |
| circuit_breaker | `nodes/node_event_bus_circuit_breaker_compute/v1_0_0/node.py` | 524 | State machine logic |

### DO NOT Copy

| Asset | Reason |
|-------|--------|
| Import statements | All wrong |
| Base class inheritance | Different in 0.3.5 |
| Node scaffolding | Rebuild with correct patterns |
| Archived tests | Recreate fresh |
| kafka_adapter | Use event bus mixins |

---

## Development Velocity Reference

### Historical Data (from omninode_bridge)

| Period | Commits | Rate |
|--------|---------|------|
| Initial Sprint (Sep 21-30) | 117 | 11.7/day |
| Stabilization (Oct) | 12 | 0.4/day |
| Feature Dev (Nov) | 19 | 0.6/day |
| **Sustainable Average** | - | **1-2/day** |

### Planning Velocity

| Scenario | Commits/Day | PRs/Week |
|----------|-------------|----------|
| Optimistic | 4-5 | 5-6 |
| Realistic | 1-2 | 3-4 |
| Conservative | 0.5 | 1-2 |

---

## MVP Timeline

### Week 1: Foundation (Days 1-5)

#### Day 1-2: Repository Setup
- [ ] Backup current repo: `mv omnibase_infra omnibase_infra_bak`
- [ ] Create new repo with correct structure
- [ ] Initialize pyproject.toml with PyPI dependencies
- [ ] Create directory structure following omniintelligence

#### Day 3-4: Core Infrastructure
- [ ] Create centralized `models/` directory
- [ ] Migrate Pydantic models from backup (fix imports)
- [ ] Create centralized `enums/` directory
- [ ] Set up `events/` infrastructure skeleton
- [ ] Copy PostgresConnectionManager (fix imports)

#### Day 5: Validation
- [ ] Verify all imports resolve with `python -c "import omnibase_infra"`
- [ ] Run mypy on entire package
- [ ] Run ruff linting
- [ ] Basic smoke tests

### Week 2: Node Implementation (Days 6-10)

#### Day 6-7: Priority Effect Nodes
- [ ] `node_postgres_adapter_effect` (P0)
  - New scaffolding with NodeEffect
  - Extract business logic from backup
  - Create contract YAML
- [ ] `node_vault_adapter_effect` (P1)
- [ ] `node_consul_adapter_effect` (P1)

#### Day 8: Additional Effect Nodes
- [ ] `node_keycloak_adapter_effect` (P2)
- [ ] `node_webhook_effect` (P2)

#### Day 9-10: Declarative Nodes
- [ ] `infra_reducer` with FSM contracts
  - Follow omniintelligence IntelligenceReducer pattern
  - Create FSM YAML contracts
- [ ] `infra_orchestrator` with workflow contracts
  - Inherit from NodeOrchestratorDeclarative
  - Create workflow YAML contracts

### Week 3: Testing & Polish (Days 11-15)

#### Day 11-12: Compute Nodes
- [ ] `node_circuit_breaker_compute`
- [ ] `node_tracing_compute` (if needed)

#### Day 13-14: Test Suite
- [ ] Create test infrastructure (conftest.py, fixtures)
- [ ] Unit tests for each node
- [ ] Integration tests for critical paths
- [ ] Migrate useful patterns from archived tests

#### Day 15: Final Validation
- [ ] Full test run with coverage report
- [ ] Target: >80% coverage
- [ ] Documentation review
- [ ] MVP sign-off

---

## Success Criteria

### MVP Must-Haves

| Requirement | Target |
|-------------|--------|
| PyPI dependencies (core 0.3.5, spi 0.2.0) | Working |
| 5 Effect nodes operational | All passing tests |
| 1 Declarative Reducer | FSM-driven, YAML contracts |
| 1 Declarative Orchestrator | Workflow-driven, YAML contracts |
| Test coverage | >80% |
| Type checking | mypy --strict passes |
| Linting | ruff clean |

### MVP Nice-to-Haves

| Requirement | Priority |
|-------------|----------|
| 2 Compute nodes | P2 |
| Full integration test suite | P2 |
| Performance benchmarks | P3 |
| CI/CD pipeline | P3 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| omnibase_core 0.3.5 incompatible | Validate on Day 1 with simple import test |
| Business logic extraction breaks | Keep backup, extract incrementally |
| Declarative patterns unfamiliar | Follow omniintelligence exactly |
| Single developer bottleneck | Focus on P0/P1, defer P2/P3 |
| Test recreation takes too long | Start with critical path tests only |

---

## Reference Documents

### In omninode_bridge
- `/docs/migration/OMNIBASE_INFRA_MAP.md` - Complete component inventory
- `/docs/migration/SQL_CATEGORIZATION.md` - SQL migration distribution
- `/docs/planning/MVP_REGISTRY_SELF_REGISTRATION.md` - MVP Phase 1a
- `/docs/planning/IMPLEMENTATION_ROADMAP.md` - Complete roadmap

### In omniintelligence
- Pattern reference for all node types
- Declarative reducer: `nodes/intelligence_reducer/v1_0_0/reducer.py`
- Declarative orchestrator: `nodes/intelligence_orchestrator/v1_0_0/orchestrator.py`

### In omnibase_core
- `nodes/node_reducer_declarative.py` - Base declarative reducer
- `nodes/node_orchestrator_declarative.py` - Base declarative orchestrator
- `mixins/mixin_fsm_execution.py` - FSM execution mixin
- `mixins/mixin_workflow_execution.py` - Workflow execution mixin

---

## Quick Start Commands

```bash
# 1. Backup current repo
cd /Users/jonah/Code
mv omnibase_infra omnibase_infra_bak

# 2. Create new repo
mkdir omnibase_infra
cd omnibase_infra
git init

# 3. Initialize with poetry
poetry init --name omnibase_infra --python "^3.12"
poetry add omnibase-core@^0.3.5 omnibase-spi@^0.2.0

# 4. Create structure
mkdir -p src/omnibase_infra/{adapters,clients,contracts,enums,events,models,nodes,shared,utils}
mkdir -p tests/{unit,integration,nodes}

# 5. Validate dependencies
python -c "from omnibase_core.nodes import NodeEffect; print('Core OK')"
python -c "from omnibase_spi import ProtocolEventBus; print('SPI OK')"
```

---

## Open Questions

1. **SQL Migrations**: Do we need to copy the 40+ SQL migrations from omninode_bridge, or will infrastructure tables be managed elsewhere?

2. **Metadata Stamping**: Is the metadata stamping service (21K LOC) needed in omnibase_infra, or is it going to a different repo?

3. **Event Infrastructure**: How much of the event publisher infrastructure do we need if we're using core event bus mixins?

4. **Keycloak Priority**: Is Keycloak adapter P1 or P2? Can it be deferred past MVP?

5. **Test Coverage Target**: Is 80% sufficient for MVP, or do we need higher?

---

## Next Steps

1. **Review this document** and answer open questions
2. **Validate omnibase_core 0.3.5** - ensure it's published to PyPI
3. **Create the backup** and new repository
4. **Begin Week 1 execution**

---

*This handoff document was generated during session da3112e3-8682-428f-81ae-39616e41bd57 on December 2, 2025.*
