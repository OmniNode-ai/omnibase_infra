# omnibase_infra MVP Execution Plan

**Created**: December 2, 2025
**Status**: Ready for Execution
**Target**: Fresh ONEX-compliant repository

---

## Executive Summary

Fresh start rebuild of `omnibase_infra` using:
- `omnibase-core` ^0.3.5 (PyPI)
- `omnibase-spi` ^0.2.0 (PyPI)
- Pattern reference: `omniintelligence`

### Architectural Decisions

| Service | Destination | Rationale |
|---------|-------------|-----------|
| Stamping Service | `omnibase_infra` | Deterministic crypto ops, foundational |
| Tree Service | `omniintelligence` | Semantic reasoning, pattern detection |
| Integration | Event Bus | Loose coupling via events |

---

## Core Design Invariants

These invariants MUST be maintained throughout the ONEX architecture:

1. **All behavior is contract-driven** - No implicit behavior
2. **NodeRuntime is the only executable event loop** - Nodes don't run their own loops
3. **Node logic is pure: no I/O, no mixins, no inheritance**
4. **Core never depends on SPI or infra** - Dependency: infra -> spi -> core
5. **SPI only defines protocols, never implementations**
6. **Infra owns all I/O and real system integrations**

**Critical Invariant**:
```
No code in omnibase_core may initiate network I/O, database I/O,
file I/O, or external process execution.
```

---

## Package Responsibilities

| Package | Contains | Depends On |
|---------|----------|------------|
| `omnibase_core` | Core models, ONEX contract models, NodeRuntime, NodeInstance, enums, primitives | pydantic, stdlib |
| `omnibase_spi` | Pure Protocol interfaces for handlers, event bus, system services | omnibase_core |
| `omnibase_infra` | Concrete handler implementations, runtime-host entrypoints, wire-up code | omnibase_core, omnibase_spi |

**Dependency Flow**:
```
omnibase_core   (no external deps)
       ^
omnibase_spi    (depends on core)
       ^
omnibase_infra  (depends on core + spi)
```

---

> **Note**: This MVP plan was created before the Runtime Host architectural decision.
> The Runtime Host model (documented in HANDOFF_SESSION_2025_12_03.md) supersedes
> the 1-container-per-node deployment approach described in some sections.

---

## Phase 0: Pre-flight

### 0.1 Dependency Validation

```bash
# Verify PyPI packages exist
pip index versions omnibase-core  # Should show 0.3.5
pip index versions omnibase-spi   # Should show 0.2.0

# Test imports work
python -c "from omnibase_core.nodes import NodeEffect; print('Core OK')"
python -c "from omnibase_spi.protocols import ProtocolEventBus; print('SPI OK')"
```

### 0.2 Backup Current Repository

```bash
cd /Users/jonah/Code
mv omnibase_infra omnibase_infra_bak
```

### 0.3 Initialize New Repository

```bash
mkdir omnibase_infra
cd omnibase_infra
git init
```

### Success Criteria
- [ ] omnibase-core 0.3.5 available on PyPI
- [ ] omnibase-spi 0.2.0 available on PyPI
- [ ] Current repo backed up
- [ ] Fresh repo initialized

---

## Phase 1: Foundation

### 1.1 Directory Structure

```
omnibase_infra/
├── pyproject.toml
├── README.md
├── src/omnibase_infra/
│   ├── __init__.py
│   ├── handlers/                  # Protocol handler implementations
│   │   ├── __init__.py
│   │   ├── http/
│   │   │   ├── http_rest_handler.py
│   │   │   └── http_circuit_breaker.py
│   │   ├── db/
│   │   │   ├── postgres_handler.py
│   │   │   └── connection_pool.py
│   │   └── event/
│   │       └── kafka_handler.py
│   ├── resilience/                # Resilience patterns
│   │   ├── retry_policy.py
│   │   ├── circuit_breaker.py
│   │   └── rate_limiter.py
│   ├── clients/                   # Service clients
│   │   └── __init__.py
│   ├── enums/                     # Centralized enums
│   │   ├── __init__.py            # Export all enums
│   │   ├── enum_infra_fsm_type.py
│   │   ├── enum_operation_type.py
│   │   └── enum_service_status.py
│   ├── models/                    # Centralized Pydantic models
│   │   ├── __init__.py            # Comprehensive exports
│   │   ├── model_postgres_config.py
│   │   ├── model_postgres_query.py
│   │   ├── model_vault_secret.py
│   │   ├── model_consul_service.py
│   │   ├── model_stamping_request.py
│   │   └── model_stamping_result.py
│   ├── nodes/                     # ONEX nodes (includes effect nodes for adapters)
│   │   └── __init__.py
│   ├── infrastructure/            # Utilities
│   │   ├── __init__.py
│   │   └── postgres_connection_manager.py
│   ├── shared/                    # Shared utilities
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   └── __init__.py
│   ├── integration/
│   │   └── __init__.py
│   └── nodes/
│       └── __init__.py
└── docs/
```

### 1.2 pyproject.toml

```toml
[tool.poetry]
name = "omnibase-infra"
version = "0.1.0"
description = "ONEX Infrastructure Services"
authors = ["OmniNode AI"]
readme = "README.md"
packages = [{include = "omnibase_infra", from = "src"}]

[tool.poetry.dependencies]
python = "^3.12"

# ONEX dependencies - PyPI releases
omnibase-core = "^0.3.5"
omnibase-spi = "^0.2.0"

# Core
pydantic = "^2.11.7"
fastapi = "^0.115.0"
uvicorn = "^0.32.0"

# Database
asyncpg = "^0.29.0"
psycopg2-binary = "^2.9.10"

# Service integration
python-consul = "^1.1.0"
hvac = "^2.1.0"

# Cryptography (for stamping)
blake3 = "^0.4.1"

# Observability
structlog = "^23.2.0"
prometheus-client = "^0.19.0"
opentelemetry-api = "^1.27.0"
opentelemetry-sdk = "^1.27.0"

# Resilience
tenacity = "^9.0.0"
circuitbreaker = "^2.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^4.1.0"
mypy = "^1.8.0"
ruff = "^0.2.0"

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "ANN", "B", "C4", "SIM"]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 1.3 Core Import Patterns

**CORRECT imports for omnibase_core 0.3.5:**

```python
# Node base classes
from omnibase_core.nodes import NodeEffect, NodeCompute, NodeReducer, NodeOrchestrator

# Input/Output models
from omnibase_core.nodes import ModelEffectInput, ModelEffectOutput
from omnibase_core.nodes import ModelReducerInput, ModelReducerOutput

# Mixins for declarative patterns
from omnibase_core.mixins.mixin_fsm_execution import MixinFSMExecution
from omnibase_core.mixins.mixin_workflow_execution import MixinWorkflowExecution

# FSM subcontracts
from omnibase_core.models.contracts.subcontracts.model_fsm_subcontract import ModelFSMSubcontract
from omnibase_core.utils.util_safe_yaml_loader import load_and_validate_yaml_model

# Container
from omnibase_core.models.container.model_onex_container import ModelONEXContainer

# Errors
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
```

### 1.4 Centralized Enums (src/omnibase_infra/enums/)

**enum_infra_fsm_type.py:**
```python
from enum import Enum

class EnumInfraFSMType(str, Enum):
    """FSM types for infrastructure state management."""
    INFRASTRUCTURE_STATE = "infrastructure_state"
    SERVICE_LIFECYCLE = "service_lifecycle"
    CONNECTION_POOL = "connection_pool"
```

**enum_operation_type.py:**
```python
from enum import Enum

class EnumOperationType(str, Enum):
    """Infrastructure operation types."""
    QUERY = "query"
    TRANSACTION = "transaction"
    HEALTH_CHECK = "health_check"
    STAMP = "stamp"
    VALIDATE = "validate"
```

**__init__.py:**
```python
from omnibase_infra.enums.enum_infra_fsm_type import EnumInfraFSMType
from omnibase_infra.enums.enum_operation_type import EnumOperationType
from omnibase_infra.enums.enum_service_status import EnumServiceStatus

__all__ = [
    "EnumInfraFSMType",
    "EnumOperationType",
    "EnumServiceStatus",
]
```

### 1.5 PostgresConnectionManager

Copy from backup with import fixes:
- Source: `omnibase_infra_bak/src/omnibase_infra/infrastructure/postgres/postgres_connection_manager.py`
- Target: `src/omnibase_infra/infrastructure/postgres_connection_manager.py`

**Key changes:**
```python
# OLD (wrong)
from omnibase_core.core.errors.onex_error import OnexError, CoreErrorCode

# NEW (correct)
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
```

### Success Criteria
- [ ] Directory structure created
- [ ] pyproject.toml with correct dependencies
- [ ] `poetry install` succeeds
- [ ] `python -c "import omnibase_infra"` works
- [ ] Centralized enums in place
- [ ] PostgresConnectionManager migrated

---

## Phase 2: Effect Nodes

### Node Structure Pattern

Each effect node follows this structure:
```
nodes/vault_adapter/
    contract.yaml            # Declarative node contract
    models.py                # Input/output Pydantic models
    logic.py                 # Pure domain logic (NOT effect.py)
```

**Important Post-Migration Structure**:
- No inheritance
- No mixins
- No event loop
- No service wiring

All lifecycle work is handled by the runtime host. Nodes contain only pure domain logic
that operates on their input models and produces output models. The runtime host manages
initialization, shutdown, event loop execution, and dependency injection.

### 2.1 node_postgres_adapter_effect (P0)

**Priority**: P0 - Foundation for all database operations

**effect.py:**
```python
"""PostgreSQL adapter effect node."""
from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes import NodeEffect, ModelEffectInput, ModelEffectOutput
from tenacity import retry, stop_after_attempt, wait_exponential

from omnibase_infra.infrastructure.postgres_connection_manager import PostgresConnectionManager
from omnibase_infra.models import ModelPostgresQuery, ModelPostgresResult

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodePostgresAdapterEffect(NodeEffect):
    """Effect node for PostgreSQL database operations."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
        self._connection_manager: PostgresConnectionManager | None = None

    async def initialize(self) -> None:
        """Initialize connection pool."""
        config = self._container.get_config("postgres")
        self._connection_manager = PostgresConnectionManager(config)
        await self._connection_manager.initialize()

    async def shutdown(self) -> None:
        """Cleanup connection pool."""
        if self._connection_manager:
            await self._connection_manager.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def execute_effect(
        self,
        input_data: ModelEffectInput,
    ) -> ModelEffectOutput:
        """Execute database operation."""
        query = ModelPostgresQuery.model_validate(input_data.payload)

        # Validate SQL (prevent injection)
        self._validate_sql(query.sql)

        result = await self._connection_manager.execute(
            query.sql,
            query.parameters,
        )

        return ModelEffectOutput(
            success=True,
            payload=ModelPostgresResult(rows=result).model_dump(),
        )

    def _validate_sql(self, sql: str) -> None:
        """Validate SQL against injection attacks."""
        # SQL validation logic from backup
        pass
```

**contracts/effect_contract.yaml:**
```yaml
contract_version: "1.0.0"
node_version: "1.0.0"
name: "node_postgres_adapter_effect"
node_type: "EFFECT"
description: "PostgreSQL database adapter for ONEX infrastructure"

input_model:
  name: "ModelPostgresQuery"
  module: "omnibase_infra.models.model_postgres_query"

output_model:
  name: "ModelPostgresResult"
  module: "omnibase_infra.models.model_postgres_result"

io_operations:
  - type: "database"
    operations: ["read", "write", "transaction"]
    target: "postgresql"

dependencies:
  - name: "postgres_connection_manager"
    type: "utility"
    module: "omnibase_infra.infrastructure.postgres_connection_manager"
```

### 2.2 node_vault_adapter_effect (P1)

**Source reference**: `omnibase_infra_bak/src/omnibase_infra/nodes/node_vault_adapter_effect/v1_0_0/node.py`

**Key functionality to extract:**
- Secret lifecycle management
- Mock client for testing
- Token refresh logic
- KV operations

### 2.3 node_consul_adapter_effect (P1)

**Source reference**: `omnibase_infra_bak/src/omnibase_infra/nodes/consul_adapter/v1_0_0/node.py`

**Key functionality to extract:**
- Service registration/deregistration
- Health check management
- KV store operations
- Service discovery queries

### 2.4 node_keycloak_adapter_effect (P2)

**Source reference**: `omnibase_infra_bak/src/omnibase_infra/nodes/node_keycloak_adapter_effect/v1_0_0/node.py`

**Key functionality to extract:**
- JWT token handling
- Role management
- User authentication flows

### 2.5 node_webhook_effect (P2)

**Source reference**: `omnibase_infra_bak/src/omnibase_infra/nodes/hook_node/v1_0_0/node.py`

**Key functionality to extract:**
- Webhook delivery with retry
- HMAC signing
- Delivery status tracking

### Success Criteria
- [ ] node_postgres_adapter_effect operational
- [ ] node_vault_adapter_effect operational
- [ ] node_consul_adapter_effect operational
- [ ] node_keycloak_adapter_effect operational
- [ ] node_webhook_effect operational
- [ ] All effect nodes have contracts
- [ ] Unit tests for each node

---

## Phase 3: Stamping Service

### 3.1 Source Location

```
omninode_bridge/src/omninode_bridge/services/metadata_stamping/
├── service.py                    # Main orchestrator
├── engine/
│   ├── stamping_engine.py        # Core stamping logic
│   └── hash_generator.py         # BLAKE3 hashing
├── database/
│   └── client.py                 # PostgreSQL persistence
├── events/
│   └── publisher.py              # Event publishing
└── api/
    └── router.py                 # FastAPI endpoints
```

### 3.2 Target Structure

```
src/omnibase_infra/nodes/node_stamping_effect/v1_0_0/
├── __init__.py
├── __main__.py
├── effect.py                     # Main effect implementation
├── contracts/
│   └── effect_contract.yaml
└── engine/
    ├── __init__.py
    ├── stamping_engine.py        # Adapted from source
    └── hash_generator.py         # Adapted from source
```

### 3.3 Key Adaptations

**effect.py:**
```python
"""Metadata stamping effect node."""
from __future__ import annotations

from omnibase_core.nodes import NodeEffect, ModelEffectInput, ModelEffectOutput

from omnibase_infra.models import ModelStampingRequest, ModelStampingResult
from omnibase_infra.nodes.node_stamping_effect.v1_0_0.engine import StampingEngine


class NodeStampingEffect(NodeEffect):
    """Effect node for metadata stamping operations."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
        self._engine: StampingEngine | None = None

    async def initialize(self) -> None:
        """Initialize stamping engine."""
        self._engine = StampingEngine()

    async def execute_effect(
        self,
        input_data: ModelEffectInput,
    ) -> ModelEffectOutput:
        """Execute stamping operation."""
        request = ModelStampingRequest.model_validate(input_data.payload)

        result = await self._engine.stamp_content(
            content=request.content,
            format=request.format,
        )

        return ModelEffectOutput(
            success=True,
            payload=ModelStampingResult(
                uid=result.uid,
                hash=result.hash,
                timestamp=result.timestamp,
                stamped_content=result.stamped_content,
            ).model_dump(),
        )
```

### 3.4 Models for Stamping

**model_stamping_request.py:**
```python
from pydantic import BaseModel, Field

class ModelStampingRequest(BaseModel):
    """Request model for stamping operations."""
    content: str = Field(..., description="Content to stamp")
    format: str = Field(default="lightweight", description="Stamp format")
    metadata: dict[str, str] = Field(default_factory=dict)
```

**model_stamping_result.py:**
```python
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

class ModelStampingResult(BaseModel):
    """Result model for stamping operations."""
    uid: UUID = Field(..., description="Unique stamp identifier")
    hash: str = Field(..., description="BLAKE3 hash")
    timestamp: datetime = Field(..., description="Stamp timestamp")
    stamped_content: str = Field(..., description="Content with stamp")
```

### Success Criteria
- [ ] Stamping engine extracted and adapted
- [ ] Hash generator working with BLAKE3
- [ ] NodeStampingEffect operational
- [ ] <10ms stamping performance maintained
- [ ] Unit tests passing

---

## Phase 4: Declarative Nodes

### 4.1 infra_reducer (FSM-Driven)

**Pattern reference**: `omniintelligence/src/omniintelligence/nodes/intelligence_reducer/v1_0_0/reducer.py`

**Structure:**
```
src/omnibase_infra/nodes/infra_reducer/v1_0_0/
├── __init__.py
├── __main__.py
├── reducer.py
└── contracts/
    ├── fsm_infrastructure_state.yaml
    └── fsm_service_lifecycle.yaml
```

**reducer.py:**
```python
"""Infrastructure reducer using declarative FSM pattern."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.mixins.mixin_fsm_execution import MixinFSMExecution
from omnibase_core.models.contracts.subcontracts.model_fsm_subcontract import ModelFSMSubcontract
from omnibase_core.utils.util_safe_yaml_loader import load_and_validate_yaml_model

from omnibase_infra.enums import EnumInfraFSMType

if TYPE_CHECKING:
    from omnibase_infra.models import ModelReducerConfig

_CONTRACTS_DIR = Path(__file__).parent / "contracts"


class InfraReducer(MixinFSMExecution):
    """Infrastructure reducer with declarative FSM contracts."""

    def __init__(self, config: ModelReducerConfig) -> None:
        super().__init__()
        self._config = config
        self._fsm_contracts = self._load_fsm_contracts()

    def _load_fsm_contracts(self) -> dict[EnumInfraFSMType, ModelFSMSubcontract]:
        """Load all FSM contracts from YAML."""
        return {
            EnumInfraFSMType.INFRASTRUCTURE_STATE: load_and_validate_yaml_model(
                _CONTRACTS_DIR / "fsm_infrastructure_state.yaml",
                ModelFSMSubcontract,
            ),
            EnumInfraFSMType.SERVICE_LIFECYCLE: load_and_validate_yaml_model(
                _CONTRACTS_DIR / "fsm_service_lifecycle.yaml",
                ModelFSMSubcontract,
            ),
        }

    def get_fsm_contract(self, fsm_type: EnumInfraFSMType) -> ModelFSMSubcontract:
        """Get FSM contract by type."""
        return self._fsm_contracts[fsm_type]
```

**contracts/fsm_infrastructure_state.yaml:**
```yaml
name: "infrastructure_state"
version: "1.0.0"
description: "FSM for infrastructure state management"

states:
  - name: "initializing"
    type: "initial"
  - name: "healthy"
    type: "normal"
  - name: "degraded"
    type: "normal"
  - name: "unhealthy"
    type: "normal"
  - name: "shutdown"
    type: "final"

transitions:
  - from: "initializing"
    to: "healthy"
    event: "initialization_complete"
  - from: "healthy"
    to: "degraded"
    event: "service_degraded"
  - from: "degraded"
    to: "healthy"
    event: "service_recovered"
  - from: "degraded"
    to: "unhealthy"
    event: "service_failed"
  - from: "unhealthy"
    to: "degraded"
    event: "partial_recovery"
  - from: "*"
    to: "shutdown"
    event: "shutdown_requested"
```

### 4.2 infra_orchestrator (Workflow-Driven)

**Pattern reference**: `omniintelligence/src/omniintelligence/nodes/intelligence_orchestrator/v1_0_0/orchestrator.py`

**Structure:**
```
src/omnibase_infra/nodes/infra_orchestrator/v1_0_0/
├── __init__.py
├── __main__.py
├── orchestrator.py
└── contracts/
    └── workflows/
        ├── deployment.yaml
        └── health_check.yaml
```

**orchestrator.py:**
```python
"""Infrastructure orchestrator using declarative workflow pattern."""
from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator_declarative import NodeOrchestratorDeclarative

from omnibase_infra.enums import EnumOperationType

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.contracts.model_workflow_definition import ModelWorkflowDefinition


class InfraOrchestrator(NodeOrchestratorDeclarative):
    """Infrastructure orchestrator with declarative workflows."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
        self._workflow_cache: dict[EnumOperationType, ModelWorkflowDefinition] = {}

    async def execute_deployment(self, deployment_config: dict) -> dict:
        """Execute deployment workflow."""
        workflow = self._load_workflow("deployment")
        return await self.execute_workflow(workflow, deployment_config)

    async def execute_health_check(self) -> dict:
        """Execute health check workflow."""
        workflow = self._load_workflow("health_check")
        return await self.execute_workflow(workflow, {})
```

### Success Criteria
- [ ] InfraReducer with FSM contracts operational
- [ ] InfraOrchestrator with workflow contracts operational
- [ ] FSM state transitions working
- [ ] Workflow execution working
- [ ] Unit tests passing

---

## Phase 5: Compute Nodes

### 5.1 node_circuit_breaker_compute

**Source reference**: `omnibase_infra_bak/src/omnibase_infra/nodes/node_event_bus_circuit_breaker_compute/v1_0_0/node.py`

**Key functionality:**
- Circuit state machine (closed/open/half-open)
- Failure threshold tracking
- Recovery timing
- Metrics emission

### 5.2 node_tracing_compute (Optional)

**Source reference**: `omnibase_infra_bak/src/omnibase_infra/nodes/node_distributed_tracing_compute/v1_0_0/node.py`

**Key functionality:**
- Trace context propagation
- Span management
- OpenTelemetry integration

### Success Criteria
- [ ] node_circuit_breaker_compute operational
- [ ] node_tracing_compute operational (if needed)
- [ ] Unit tests passing

---

## Phase 6: Testing & Validation

### 6.1 Test Infrastructure

**tests/conftest.py:**
```python
"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from omnibase_core.models.container.model_onex_container import ModelONEXContainer


@pytest.fixture
def mock_container() -> ModelONEXContainer:
    """Create mock ONEX container for testing."""
    container = MagicMock(spec=ModelONEXContainer)
    container.get_config.return_value = {}
    return container


@pytest.fixture
def mock_postgres_pool() -> AsyncMock:
    """Create mock PostgreSQL connection pool."""
    pool = AsyncMock()
    pool.execute.return_value = []
    return pool
```

### 6.2 Unit Test Pattern

**tests/nodes/test_node_postgres_adapter_effect.py:**
```python
"""Tests for PostgreSQL adapter effect node."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.effect import (
    NodePostgresAdapterEffect,
)


class TestNodePostgresAdapterEffect:
    """Test suite for PostgreSQL adapter."""

    @pytest.fixture
    def node(self, mock_container):
        """Create node instance."""
        return NodePostgresAdapterEffect(mock_container)

    @pytest.mark.asyncio
    async def test_execute_query(self, node, mock_postgres_pool):
        """Test basic query execution."""
        with patch.object(node, "_connection_manager", mock_postgres_pool):
            mock_postgres_pool.execute.return_value = [{"id": 1}]

            result = await node.execute_effect(
                ModelEffectInput(payload={"sql": "SELECT 1", "parameters": []})
            )

            assert result.success is True
            assert result.payload["rows"] == [{"id": 1}]

    @pytest.mark.asyncio
    async def test_sql_injection_blocked(self, node):
        """Test SQL injection prevention."""
        with pytest.raises(ValueError, match="SQL injection"):
            node._validate_sql("SELECT * FROM users; DROP TABLE users;--")
```

### 6.3 Quality Gates

```bash
# Type checking
mypy src/omnibase_infra --strict

# Linting
ruff check src/omnibase_infra
ruff format src/omnibase_infra --check

# Tests with coverage
pytest tests/ -v --cov=omnibase_infra --cov-report=term-missing --cov-fail-under=80

# Full validation
make validate  # Or: poetry run validate
```

### 6.4 Validation Checklist

```bash
# Import validation
python -c "import omnibase_infra; print('Package OK')"
python -c "from omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0 import NodePostgresAdapterEffect; print('Postgres OK')"
python -c "from omnibase_infra.nodes.node_stamping_effect.v1_0_0 import NodeStampingEffect; print('Stamping OK')"
python -c "from omnibase_infra.nodes.infra_reducer.v1_0_0 import InfraReducer; print('Reducer OK')"
python -c "from omnibase_infra.nodes.infra_orchestrator.v1_0_0 import InfraOrchestrator; print('Orchestrator OK')"
```

### Success Criteria
- [ ] All imports resolve correctly
- [ ] mypy --strict passes
- [ ] ruff clean (no lint errors)
- [ ] pytest coverage >80%
- [ ] All nodes have unit tests
- [ ] Integration tests for critical paths

---

## Final Checklist

### MVP Must-Haves

| Requirement | Status |
|-------------|--------|
| PyPI dependencies (core 0.3.5, spi 0.2.0) | [ ] |
| node_postgres_adapter_effect | [ ] |
| node_vault_adapter_effect | [ ] |
| node_consul_adapter_effect | [ ] |
| node_stamping_effect | [ ] |
| infra_reducer (FSM-driven) | [ ] |
| infra_orchestrator (workflow-driven) | [ ] |
| Test coverage >80% | [ ] |
| mypy --strict passes | [ ] |
| ruff clean | [ ] |

### MVP Nice-to-Haves

| Requirement | Priority | Status |
|-------------|----------|--------|
| node_keycloak_adapter_effect | P2 | [ ] |
| node_webhook_effect | P2 | [ ] |
| node_circuit_breaker_compute | P2 | [ ] |
| node_tracing_compute | P3 | [ ] |
| Full integration tests | P2 | [ ] |
| CI/CD pipeline | P3 | [ ] |

---

## Quick Reference

### Create Directory Structure
```bash
mkdir -p src/omnibase_infra/{handlers/http,handlers/db,handlers/event,resilience,clients,enums,models,nodes,infrastructure,shared,utils}
mkdir -p tests/{unit,integration,nodes}
mkdir -p docs
touch src/omnibase_infra/__init__.py
touch src/omnibase_infra/{handlers,resilience,clients,enums,models,nodes,infrastructure,shared,utils}/__init__.py
```

### Initialize Poetry
```bash
poetry init --name omnibase-infra --python "^3.12"
poetry add omnibase-core@^0.3.5 omnibase-spi@^0.2.0
poetry add pydantic@^2.11.7 fastapi@^0.115.0 asyncpg@^0.29.0 blake3@^0.4.1
poetry add --group dev pytest@^8.0.0 pytest-asyncio@^0.23.0 mypy@^1.8.0 ruff@^0.2.0
```

### Validate Installation
```bash
poetry install
poetry run python -c "from omnibase_core.nodes import NodeEffect; print('Ready')"
```

---

*This execution plan was created on December 2, 2025.*
