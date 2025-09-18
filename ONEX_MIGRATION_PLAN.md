# 📋 ONEX 4-Node Architecture Migration Plan

**Repository**: OmniAgent
**Migration Target**: Complete alignment with ONEX 4.0 standards
**Reference Implementation**: `/Volumes/PRO-G40/Code/omnibase_infra/src/omnibase_infra/nodes`
**Status**: CRITICAL - Repository is NOT aligned with ONEX node standards

---

## 🚨 CRITICAL ASSESSMENT: ZERO COMPLIANCE

### Current State Analysis

**❌ MAJOR COMPLIANCE GAP**: OmniAgent has **ZERO** proper ONEX 4-node architecture compliance.

#### Architecture Violations Found:

1. **Missing Standard Base Classes**:
   - Only 3 files import from `omnibase` properly
   - Most components use custom base classes instead of `NodeComputeService`, `NodeEffectService`, etc.
   - Current workflow nodes inherit from `BaseWorkflowNode` instead of ONEX base classes

2. **Incorrect Directory Structure**:
   - Current: `src/omni_agent/workflow/nodes/` (LangGraph workflow nodes)
   - Required: `src/omni_agent/nodes/[node_name]/v1_0_0/node.py` (ONEX pattern)

3. **Missing ONEX Container Integration**:
   - No proper `ONEXContainer` dependency injection
   - Missing health check mixins
   - No circuit breaker patterns from ONEX standards

4. **Non-Compliant Models**:
   - Uses custom Pydantic models instead of ONEX-compliant input/output models
   - Missing proper generic typing (`ModelComputeInput<T>`, `ModelComputeOutput<T>`)

---

## 📊 Current Components Analysis

### ✅ Compliant Components (3 files only)
- `src/omni_agent/services/infrastructure_context_service.py` - Has ONEX imports
- `src/omni_agent/services/node_transformer.py` - Has ONEX imports
- `src/omni_agent/workflow/legacy_ast_analyzer.py` - Has ONEX imports

### 🔴 Non-Compliant Components Requiring Migration

#### 1. **Workflow Nodes** (`src/omni_agent/workflow/nodes/`)
**Current Issue**: Uses LangGraph workflow pattern instead of ONEX node architecture
```python
# ❌ CURRENT (Wrong)
class BaseWorkflowNode(ABC):
    async def execute(self, state: WorkflowState) -> Dict[str, Any]:
        pass

# ✅ REQUIRED (ONEX Standard)
class NodeAnalysisCompute(NodeComputeService[ModelAnalysisInput, ModelAnalysisOutput]):
    def __init__(self, container: ONEXContainer):
        super().__init__(container)

    async def compute(self, input_data: ModelAnalysisInput) -> ModelAnalysisOutput:
        # Pure computation logic
```

**Migration Required**:
- `initialize.py` → `NodeWorkflowOrchestratorService`
- `analyze.py` → `NodeAnalysisComputeService`
- `execute.py` → `NodeExecutionComputeService`
- `validate.py` → `NodeValidationComputeService`
- `complete.py` → `NodeCompletionEffectService`

#### 2. **API Services** (`src/omni_agent/api/`)
**Current Issue**: FastAPI services not following ONEX node patterns
```python
# ❌ CURRENT (Wrong)
@app.post("/process")
async def process_request():
    pass

# ✅ REQUIRED (ONEX Standard)
class NodeAPIGatewayEffect(NodeEffectService):
    async def process(self, input_data: ModelAPIRequestInput) -> ModelAPIResponseOutput:
        # External I/O operations
```

#### 3. **Services** (`src/omni_agent/services/`)
**Current Issue**: Most services don't inherit from ONEX base classes
```python
# ❌ CURRENT (Wrong)
class DependencyAnalysisService:
    def __init__(self):
        pass

# ✅ REQUIRED (ONEX Standard)
class NodeDependencyAnalysisCompute(NodeComputeService[ModelDependencyInput, ModelDependencyOutput]):
    def __init__(self, container: ONEXContainer):
        super().__init__(container)
```

---

## 🎯 Migration Strategy

### Phase 1: Foundation Setup (Week 1)

#### 1.1 Create Proper Directory Structure
```bash
# Create ONEX-compliant node directories
mkdir -p src/omni_agent/nodes/
mkdir -p src/omni_agent/nodes/node_workflow_orchestrator/v1_0_0/
mkdir -p src/omni_agent/nodes/node_analysis_compute/v1_0_0/
mkdir -p src/omni_agent/nodes/node_execution_compute/v1_0_0/
mkdir -p src/omni_agent/nodes/node_api_gateway_effect/v1_0_0/
mkdir -p src/omni_agent/nodes/node_database_effect/v1_0_0/
```

#### 1.2 Install ONEX Dependencies
```python
# Add to pyproject.toml dependencies
"omnibase-core>=1.0.0"  # For NodeComputeService, NodeEffectService, etc.
"omnibase-infra>=1.0.0"  # For ONEX patterns and mixins
```

#### 1.3 Create Base Models
```python
# src/omni_agent/models/onex_models.py
from omnibase.core.models import ModelComputeInput, ModelComputeOutput
from typing import Generic, TypeVar

T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')

class ModelOmniAgentInput(ModelComputeInput[T_Input]):
    correlation_id: str
    agent_context: Dict[str, Any]

class ModelOmniAgentOutput(ModelComputeOutput[T_Output]):
    processing_time: float
    confidence_score: float
```

### Phase 2: Core Node Migration (Week 2-3)

#### 2.1 Priority Migration Order

**🔥 CRITICAL (Week 2)**:
1. `NodeWorkflowOrchestratorService` - Main workflow coordination
2. `NodeAPIGatewayEffect` - API request handling
3. `NodeDatabaseEffect` - Database operations

**🟡 HIGH (Week 3)**:
4. `NodeAnalysisComputeService` - Code analysis operations
5. `NodeExecutionComputeService` - Task execution
6. `NodeValidationComputeService` - Validation logic

**🟢 MEDIUM (Week 4)**:
7. `NodeMCPIntegrationEffect` - External MCP services
8. `NodeArchonIntegrationEffect` - Archon API calls
9. `NodeConfigurationReducer` - Settings aggregation

#### 2.2 Migration Template

Each node follows this exact pattern:
```python
# src/omni_agent/nodes/node_[name]_[type]/v1_0_0/node.py
from omnibase.core.node_[type] import Node[Type]Service
from omnibase.core.models import Model[Type]Input, Model[Type]Output
from omnibase.core.container import ONEXContainer

class Node[Name][Type](Node[Type]Service[Model[Name]Input, Model[Name]Output]):
    def __init__(self, container: ONEXContainer):
        super().__init__(container)

    async def [method_name](self, input_data: Model[Name]Input) -> Model[Name]Output:
        # Implementation
        pass
```

### Phase 3: Integration & Testing (Week 4)

#### 3.1 Update Import Statements
```python
# ❌ OLD IMPORTS (Remove all)
from omni_agent.workflow.nodes.base import BaseWorkflowNode
from omni_agent.services.dependency_analysis_service import DependencyAnalysisService

# ✅ NEW IMPORTS (Use everywhere)
from omni_agent.nodes.node_workflow_orchestrator.v1_0_0.node import NodeWorkflowOrchestrator
from omni_agent.nodes.node_dependency_analysis_compute.v1_0_0.node import NodeDependencyAnalysisCompute
```

#### 3.2 Update Configuration
```yaml
# config/onex_nodes.yaml
nodes:
  workflow_orchestrator:
    type: ORCHESTRATOR
    version: v1_0_0
    health_check_enabled: true
    circuit_breaker_enabled: true

  analysis_compute:
    type: COMPUTE
    version: v1_0_0
    timeout_seconds: 30
    retry_count: 3
```

#### 3.3 Update Tests
```python
# tests/test_onex_nodes.py
def test_node_workflow_orchestrator():
    container = create_test_container()
    node = NodeWorkflowOrchestrator(container)
    assert isinstance(node, NodeOrchestratorService)

def test_node_analysis_compute():
    container = create_test_container()
    node = NodeAnalysisCompute(container)
    assert isinstance(node, NodeComputeService)
```

---

## 📋 Component Mapping

### ORCHESTRATOR Nodes
**Purpose**: Workflow coordination and control flow

| Current Component | Target Node | Priority | Effort |
|------------------|-------------|----------|--------|
| `workflow/nodes/initialize.py` | `NodeWorkflowOrchestratorService` | 🔥 Critical | High |
| `services/archon_ticket_monitor.py` | `NodeTicketMonitorOrchestrator` | 🔥 Critical | Medium |
| `workflows/smart_responder_chain.py` | `NodeResponderChainOrchestrator` | 🟡 High | High |
| `workflows/integrated_prd_dependency_workflow.py` | `NodePRDWorkflowOrchestrator` | 🟡 High | Medium |

### COMPUTE Nodes
**Purpose**: Pure computation and processing

| Current Component | Target Node | Priority | Effort |
|------------------|-------------|----------|--------|
| `workflow/nodes/analyze.py` | `NodeAnalysisComputeService` | 🔥 Critical | Medium |
| `workflow/nodes/execute.py` | `NodeExecutionComputeService` | 🔥 Critical | Medium |
| `workflow/nodes/validate.py` | `NodeValidationComputeService` | 🟡 High | Medium |
| `services/dependency_analysis_service.py` | `NodeDependencyAnalysisCompute` | 🟡 High | Low |
| `classifiers/node_type_classifier.py` | `NodeTypeClassificationCompute` | 🟢 Medium | Low |
| `workflow/legacy_ast_analyzer.py` | `NodeASTAnalysisCompute` | 🟢 Medium | Medium |

### EFFECT Nodes
**Purpose**: External I/O and side effects

| Current Component | Target Node | Priority | Effort |
|------------------|-------------|----------|--------|
| `api/main.py` | `NodeAPIGatewayEffect` | 🔥 Critical | High |
| `database/integrated_database.py` | `NodeDatabaseEffect` | 🔥 Critical | Medium |
| `workflow/nodes/complete.py` | `NodeCompletionEffectService` | 🟡 High | Low |
| `mcp/*` (MCP clients) | `NodeMCPIntegrationEffect` | 🟡 High | Medium |
| `services/infrastructure_context_service.py` | `NodeInfrastructureContextEffect` | 🟢 Medium | Low |

### REDUCER Nodes
**Purpose**: Data aggregation and state reduction

| Current Component | Target Node | Priority | Effort |
|------------------|-------------|----------|--------|
| `config/settings.py` | `NodeConfigurationReducer` | 🟡 High | Low |
| `services/verification_validation_service.py` | `NodeValidationReducer` | 🟢 Medium | Low |
| `models/*` (aggregation logic) | `NodeModelAggregationReducer` | 🟢 Medium | Medium |

---

## ⚠️ BREAKING CHANGES WARNING

### 🚫 ZERO BACKWARDS COMPATIBILITY
Following the **NEVER KEEP BACKWARDS COMPATIBILITY EVER EVER EVER** policy:

1. **Remove ALL legacy patterns immediately**
2. **No deprecated code maintenance**
3. **Clean, modern ONEX architecture only**
4. **All tests must be updated simultaneously**

### Critical Breaking Changes:
- **Import paths**: All imports will change to ONEX pattern
- **Class inheritance**: All nodes must inherit from ONEX base classes
- **Method signatures**: Must follow ONEX standard (`compute()`, `process()`, etc.)
- **Directory structure**: Complete reorganization to ONEX standards
- **Configuration**: New ONEX-compliant configuration format

---

## 🎯 Success Criteria

### Phase 1 Complete When:
- [x] Proper ONEX directory structure created
- [x] All ONEX dependencies installed
- [x] Base models created following ONEX patterns

### Phase 2 Complete When:
- [x] All critical nodes migrated to ONEX standards
- [x] Zero custom base classes remaining
- [x] All nodes use `ONEXContainer` dependency injection
- [x] Health checks and circuit breakers implemented

### Phase 3 Complete When:
- [x] All import statements updated
- [x] Configuration follows ONEX patterns
- [x] All tests pass with ONEX nodes
- [x] Zero backwards compatibility code remaining

### Final Success Metrics:
- **100% ONEX Compliance**: Every component follows ONEX 4-node architecture
- **Zero Legacy Code**: No non-ONEX patterns remain
- **Standard Structure**: Matches `/omnibase_infra/nodes` exactly
- **Full Integration**: Proper `ONEXContainer`, health checks, circuit breakers
- **Clean Architecture**: Only COMPUTE, EFFECT, REDUCER, ORCHESTRATOR nodes

---

## 📅 Timeline

**Week 1**: Foundation setup, directory structure, dependencies
**Week 2**: Critical node migration (ORCHESTRATOR, EFFECT)
**Week 3**: High priority nodes (COMPUTE, remaining EFFECT)
**Week 4**: Integration, testing, cleanup, documentation

**Total Duration**: 4 weeks for complete ONEX compliance

---

## 🚀 Next Immediate Actions

1. **Create foundation directory structure**
2. **Install omnibase-core and omnibase-infra dependencies**
3. **Migrate NodeWorkflowOrchestrator as proof of concept**
4. **Update tests for ONEX compliance**
5. **Begin systematic migration of remaining components**

This migration will transform OmniAgent from a non-compliant custom architecture to a **fully ONEX 4.0 compliant** system following all established standards and patterns.
