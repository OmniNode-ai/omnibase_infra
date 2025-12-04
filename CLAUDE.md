# Claude Code Rules for ONEX Infrastructure

## 🚨 MANDATORY: Agent-Driven Development

**ALL CODING TASKS MUST USE SUB-AGENTS - NO EXCEPTIONS**

Claude Code operates in agent-driven mode for ONEX infrastructure development. For any coding task:
1. **NEVER code directly** - Always delegate to specialized sub-agents
2. **ALWAYS use orchestration agents** for complex workflows
3. **MANDATORY routing** through `agent-onex-coordinator` for multi-step tasks
4. **REQUIRED delegation** to domain specialists for implementation

### When to Use Which Orchestration Pattern

**Simple Tasks** → Direct specialist agent
```bash
> Use agent-commit to create semantic commit messages
> Use agent-testing for comprehensive test strategy
> Use agent-contract-validator for contract compliance
```

**Complex Workflows** → Orchestration agents
```bash
> Use agent-onex-coordinator for intelligent routing and workflow orchestration
> Use agent-workflow-coordinator for multi-step execution and progress management
> Use agent-ticket-manager for comprehensive ticket operations
```

**Multi-Domain Operations** → Combined orchestration
```bash
> Use agent-onex-coordinator to route to agent-workflow-coordinator for complex execution
> Use agent-workflow-coordinator to manage sub-agent fleets for parallel processing
> Use agent-ticket-manager for project planning and dependency management
```

## 🚫 CRITICAL POLICY: NO BACKWARDS COMPATIBILITY

**NEVER KEEP BACKWARDS COMPATIBILITY EVER EVER EVER**

This project follows a **ZERO BACKWARDS COMPATIBILITY** policy:
- **Breaking changes are always acceptable**
- **No deprecated code maintenance**
- **All models MUST conform to current protocols**
- **Clean, modern architecture only**
- **Remove old patterns immediately**

## 🎯 HIGHEST Priority ONEX Core Principles

### Strong Typing & Models
- **NEVER use `Any`** - Always use specific types
- **Use Pydantic Models** - All data structures must be proper Pydantic models
- **CamelCase Models** - All model classes: `ModelUserData`
- **snake_case files** - All filenames: `model_user_data.py`
- **One model per file** - Each file contains exactly one `Model*` class

### ONEX Architecture
- **Contract-Driven** - All tools/services follow contract patterns
- **Container Injection** - All dependencies injected via container: `def __init__(self, container: ONEXContainer)`
- **Protocol Resolution** - Use duck typing through protocols, never isinstance
- **OnexError Only** - All exceptions converted to OnexError with chaining: `raise OnexError(...) from e`

## 🚨 Infrastructure Error Usage Patterns

### Error Class Selection Guide

| Scenario | Error Class | Example |
|----------|-------------|---------|
| Service configuration invalid | `ProtocolConfigurationError` | Missing required config field |
| Secret/credential not found | `SecretResolutionError` | Vault secret missing |
| Cannot connect to service | `InfraConnectionError` | Database connection refused |
| Operation times out | `InfraTimeoutError` | Consul health check timeout |
| Authentication fails | `InfraAuthenticationError` | Invalid API key |
| Service unavailable | `InfraResourceUnavailableError` | Kafka broker down |

### Error Context Usage

All infrastructure errors accept `ModelInfraErrorContext` for structured context:

```python
from uuid import uuid4
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Create structured context
context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    target_name="postgresql-primary",
    correlation_id=request.correlation_id,  # Propagate from request
)

# Raise with proper error chaining
try:
    connection.execute(query)
except Exception as original_error:
    raise InfraConnectionError(
        "Failed to connect to database",
        context=context,
        host="db.example.com",  # Additional context via kwargs
        port=5432,
    ) from original_error
```

### Correlation ID Assignment Rules

Correlation IDs enable distributed tracing across infrastructure components:

1. **Always propagate**: Pass `correlation_id` from incoming requests to error context
2. **Auto-generation**: If no `correlation_id` exists, generate one using `uuid4()`
3. **UUID format**: Use UUID4 format for all new correlation IDs
4. **Include everywhere**: Add `correlation_id` in all error context for tracing

```python
from uuid import UUID, uuid4

# Pattern 1: Propagate from request
correlation_id = request.correlation_id or uuid4()

# Pattern 2: Generate if not available
context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.KAFKA,
    operation="produce_message",
    correlation_id=correlation_id,
)

# Pattern 3: Extract from incoming event
correlation_id = event.metadata.get("correlation_id")
if isinstance(correlation_id, str):
    correlation_id = UUID(correlation_id)
```

### Error Sanitization Guidelines

**NEVER include in error messages or context**:
- Passwords, API keys, tokens, secrets
- Full connection strings with credentials
- PII (names, emails, SSNs, phone numbers)
- Internal IP addresses (in production logs)
- Private keys or certificates
- Session tokens or cookies

**SAFE to include**:
- Service names (e.g., "postgresql", "kafka")
- Operation names (e.g., "connect", "query", "authenticate")
- Correlation IDs (always include for tracing)
- Error codes (e.g., `EnumCoreErrorCode.DATABASE_CONNECTION_ERROR`)
- Sanitized hostnames (e.g., "db.example.com")
- Port numbers
- Retry counts and timeout values
- Resource identifiers (non-sensitive)

```python
# BAD - Exposes credentials
raise InfraConnectionError(
    f"Failed to connect with password={password}",  # NEVER DO THIS
    context=context,
)

# GOOD - Sanitized error message
raise InfraConnectionError(
    "Failed to connect to database",
    context=context,
    host="db.example.com",
    port=5432,
    retry_count=3,
)

# BAD - Full connection string
raise InfraConnectionError(
    f"Connection failed: {connection_string}",  # May contain credentials
    context=context,
)

# GOOD - Sanitized connection info
raise InfraConnectionError(
    "Connection failed",
    context=context,
    host=parsed_host,
    port=parsed_port,
    database=database_name,
)
```

### Error Hierarchy Reference

```
ModelOnexError (from omnibase_core)
└── RuntimeHostError (base infrastructure error)
    ├── ProtocolConfigurationError  # Config validation failures
    ├── SecretResolutionError       # Secret/credential resolution
    ├── InfraConnectionError        # Connection failures
    ├── InfraTimeoutError           # Operation timeouts
    ├── InfraAuthenticationError    # Auth/authz failures
    └── InfraResourceUnavailableError  # Resource unavailable
```

### Transport Type Reference

Use `EnumInfraTransportType` for transport identification in error context:

| Transport Type | Value | Usage |
|---------------|-------|-------|
| `HTTP` | `"http"` | REST API transport |
| `DATABASE` | `"db"` | PostgreSQL, etc. |
| `KAFKA` | `"kafka"` | Kafka message broker |
| `CONSUL` | `"consul"` | Service discovery |
| `VAULT` | `"vault"` | Secret management |
| `REDIS` | `"redis"` | Cache/message transport |
| `GRPC` | `"grpc"` | gRPC protocol |

## 🏗️ Infrastructure-Specific Patterns

### Service Integration Architecture
- **Adapter Pattern** - External services wrapped in ONEX adapters (Consul, Kafka, Vault)
- **Connection Pooling** - Database connections managed through dedicated pool managers
- **Event-Driven Communication** - Infrastructure events flow through Kafka adapters
- **Service Discovery** - Consul integration for dynamic service resolution
- **Secret Management** - Vault integration for secure credential handling

### Infrastructure 4-Node Pattern
Infrastructure tools follow ONEX 4-node architecture:
- **EFFECT** - External service interactions (Consul, Kafka, Vault adapters)
- **COMPUTE** - Message processing and transformation (aggregators, wrappers)
- **REDUCER** - State consolidation and decision making
- **ORCHESTRATOR** - Workflow coordination and service orchestration

### Infrastructure Tool Categories

**Service Adapters:**
- `node_infrastructure_consul_adapter_effect` - Consul service discovery integration
- `node_infrastructure_kafka_adapter_effect` - Kafka event streaming integration
- `node_infrastructure_vault_adapter_effect` - Vault secret management integration

**Processing Components:**
- `node_infrastructure_message_aggregator_compute` - Event message aggregation
- `node_infrastructure_kafka_wrapper_compute` - Kafka message processing
- `node_infrastructure_node_resolver_compute` - Dynamic tool resolution

**Core Architecture:**
- `node_infrastructure_reducer` - Infrastructure state reduction
- `node_infrastructure_orchestrator` - Infrastructure workflow orchestration
- `node_infrastructure_group_gateway_effect` - Service group gateway management

**Projectors:**
- `node_infrastructure_consul_projector_effect` - Consul state projection

**Connection Management:**
- `postgres_connection_manager` - PostgreSQL connection pooling and management

## 🤖 Consolidated Agent Architecture

### Orchestration Agents (Use for Complex Workflows)

**`agent-onex-coordinator`** - Primary routing and workflow orchestration
- Intelligent routing decisions and workflow delegation
- Multi-agent coordination and resource allocation
- Entry point for complex multi-step operations

**`agent-workflow-coordinator`** - Multi-step execution and progress management
- Unified workflow execution across all ONEX domains
- Sub-agent fleet coordination and progress tracking
- Background task orchestration and result aggregation

**`agent-ticket-manager`** - Comprehensive ticket operations
- AI-powered ticket creation, analysis, and lifecycle management
- Graph algorithms for dependency analysis and critical path optimization
- Directory-based workflow orchestration and status management

### Infrastructure Specialist Agents

**Development & Generation:**
- `agent-contract-validator` - Contract validation and standards compliance
- `agent-contract-driven-generator` - Model and code generation from contracts
- `agent-ast-generator` - AST-based code generation for infrastructure structures
- `agent-commit` - Git commit messages without AI attribution

**DevOps & Infrastructure:**
- `agent-devops-infrastructure` - Container orchestration and deployment automation
- `agent-security-audit` - Infrastructure security analysis and compliance
- `agent-performance` - Infrastructure performance optimization
- `agent-production-monitor` - 24/7 infrastructure monitoring and observability

**Quality & Analysis:**
- `agent-pr-review` - PR review and merge readiness assessment
- `agent-pr-create` - PR creation with ONEX standards
- `agent-address-pr-comments` - Address PR feedback with automatic sync
- `agent-testing` - Comprehensive test strategy for infrastructure components

**Intelligence & Research:**
- `agent-research` - Infrastructure research and investigation workflows
- `agent-debug-intelligence` - AI-enhanced infrastructure incident analysis
- `agent-rag-query` - RAG knowledge retrieval for infrastructure patterns
- `agent-rag-update` - RAG knowledge updates for infrastructure learning

## 🔒 Critical Enforcement Rules

**ZERO TOLERANCE POLICIES:**
- `Any` types are absolutely forbidden under all circumstances
- Direct coding without agent delegation is prohibited
- Hand-written Pydantic models (must be contract-generated)
- Hardcoded service configurations (must be contract-driven)

## 🔧 MANDATORY DevOps Container Troubleshooting Workflow

**CRITICAL WORKFLOW UPDATE - MANDATORY LOG INSPECTION FIRST**

All DevOps infrastructure operations involving container troubleshooting MUST follow this mandatory workflow:

### 🚨 STEP 1: LOG INSPECTION (MANDATORY FIRST STEP)
```bash
# ALWAYS run container logs inspection first - NO EXCEPTIONS
docker logs <container-name>
docker logs <container-name> --tail 50  # For recent entries
docker logs <container-name> --follow   # For real-time monitoring
```

### 📊 STEP 2: LOG ANALYSIS FRAMEWORK
**Analyze logs to determine actual container behavior:**

**Expected Behaviors (NOT failures):**
- Exit code 0 after successful task completion
- "TOPIC_ALREADY_EXISTS" messages (topic creation idempotency)
- "Setup completed successfully" followed by container exit
- One-time initialization containers that exit after completion

**Actual Failure Indicators:**
- Non-zero exit codes with error messages
- Connection refused errors
- Authentication failures
- Resource unavailable errors
- Exception stack traces

### 🎯 STEP 3: STATUS VERIFICATION
```bash
# Check container exit codes and timing
docker ps -a --filter "name=<container-name>"
docker inspect <container-name> --format='{{.State.ExitCode}}'
docker inspect <container-name> --format='{{.State.FinishedAt}}'
```

### 🔍 STEP 4: ROOT CAUSE DETERMINATION
**Only proceed with fixes after confirming actual problems exist:**
- ✅ Exit code 0 + success logs = Expected behavior (no action needed)
- ❌ Exit code != 0 + error logs = Actual failure (requires investigation)
- ⚠️ Continuous restarts = Configuration or dependency issue

### 📝 STEP 5: EVIDENCE-BASED DOCUMENTATION
**Document findings with log evidence:**
```bash
# Capture evidence for analysis
docker logs <container-name> > container_analysis.log 2>&1
echo "Container Status: $(docker inspect <container-name> --format='{{.State.Status}}')"
echo "Exit Code: $(docker inspect <container-name> --format='{{.State.ExitCode}}')"
```

### 🛑 ZERO TOLERANCE POLICY
**NEVER assume container status without log evidence:**
- ❌ No assumptions based on container status alone
- ❌ No fixes without log-confirmed issues
- ❌ No troubleshooting without understanding actual behavior
- ✅ Always inspect logs first
- ✅ Always distinguish expected vs actual failures
- ✅ Always document evidence-based findings

### 📋 Container Behavior Classification
**One-time Setup Containers (Expected Exit 0):**
- RedPanda topics creation
- Database migrations
- SSL certificate generation
- Configuration initialization

**Long-running Service Containers (Should Stay Running):**
- Web services
- Message brokers
- Databases
- Load balancers

### 🎯 DevOps Agent Integration
All DevOps infrastructure agents (`agent-devops-infrastructure`, `agent-production-monitor`, `agent-performance`) MUST implement this workflow as their first diagnostic step.

## ⚙️ Infrastructure Agent Usage Patterns

### Primary Workflow: Infrastructure Orchestrated Development
```bash
# 1. Route infrastructure tasks through coordinator
> Use agent-onex-coordinator to analyze infrastructure requirements and route to specialists

# 2. Execute infrastructure workflows
> Use agent-workflow-coordinator for coordinated infrastructure deployment and management

# 3. Monitor infrastructure lifecycle
> Use agent-production-monitor for 24/7 infrastructure observability and alerting
```

### Infrastructure-Specific Workflows
```bash
# Infrastructure deployment and management
> Use agent-devops-infrastructure for container orchestration and CI/CD
> Use agent-security-audit for infrastructure security compliance
> Use agent-performance for infrastructure optimization and scaling

# Service integration development
> Use agent-contract-driven-generator for service adapter generation
> Use agent-testing for infrastructure component validation
```

### Tool Integration
```bash
# Agent Delegation (Primary Method for Infrastructure Tasks)
> Use agent-contract-driven-generator for infrastructure tool generation
> Use agent-ast-generator for infrastructure AST-based code generation
> Use agent-devops-infrastructure for deployment automation
```

## 🧠 RAG Intelligence Integration

All infrastructure agents leverage RAG intelligence for enhanced performance:
- **Pre-execution queries** via `agent-rag-query` for infrastructure patterns
- **Post-execution learning** via `agent-rag-update` for infrastructure knowledge capture
- **Incident analysis** to `agent-debug-intelligence` for infrastructure troubleshooting
- **Research enhancement** through `agent-research` with infrastructure RAG integration

## 📊 MCP Tools Integration

**Available MCP Servers:**
- **Context7**: Service integration documentation and deployment patterns
- **Sequential Thinking**: Infrastructure architecture analysis and troubleshooting
- **Archon**: Project management and infrastructure task orchestration
- **Playwright**: Infrastructure E2E testing and monitoring

#### Agent-MCP Integration for Infrastructure
Agents automatically leverage appropriate MCP tools:
- `agent-contract-validator` + Context7 for service integration patterns
- `agent-devops-infrastructure` + Sequential for deployment analysis
- `agent-production-monitor` + Archon for incident tracking
- `agent-testing` + Playwright for infrastructure E2E validation
- `agent-performance` + Sequential for infrastructure bottleneck analysis

## 🎫 Infrastructure Work Ticket Execution

**Intelligent Workflow Orchestrator** provides infrastructure-focused work ticket execution:

### Infrastructure Entry Points
```bash
# Infrastructure deployment tickets
orchestrator.execute_work_ticket("infrastructure/deployment/service_rollout.yaml")

# Service integration tickets
orchestrator.execute_work_ticket("infrastructure/integration/kafka_setup.yaml")

# Monitoring and observability tickets
orchestrator.execute_work_ticket("infrastructure/monitoring/alerting_setup.yaml")
```

### Infrastructure Work Ticket Requirements
Infrastructure tickets must include service-specific metadata:
```yaml
metadata:
  infrastructure:
    service_type: "consul|kafka|vault|postgres"
    deployment_environment: "dev|staging|prod"
    scaling_requirements: "horizontal|vertical|auto"
    monitoring_enabled: true
    security_scan_required: true
  workflow_orchestration:
    orchestrator_enabled: true
    session_id: "infrastructure-session-id"
    initial_phase: "deployment_validation"
    max_cycles: 25
    rag_strategy: "infrastructure_adaptive"
    agent_coordination_required: true
```

### Infrastructure Execution Flow
1. **Parse** infrastructure work ticket YAML and extract service parameters
2. **Initialize** infrastructure workflow session with environment context
3. **Coordinate** infrastructure agents via Claude Code Python SDK
4. **Validate** deployment readiness and security compliance
5. **Monitor** infrastructure health and performance during execution
6. **Return** infrastructure deployment status and health metrics

## 🏗️ Infrastructure Hub Architecture Standards

### Infrastructure Domain Hub (Static Implementation)
**Purpose**: Infrastructure service tool groupings and orchestration

**Characteristics**:
- Infrastructure tools loaded from contracts at startup via `managed_tools` configuration
- Service discovery integration with Consul
- Event streaming integration with Kafka
- Secret management integration with Vault
- Database connection pooling with PostgreSQL
- Uses `NodeHubBase` for unified functionality

**Key Features**:
- Contract-driven infrastructure configuration (`contract.yaml` defines everything)
- Automatic tool discovery via infrastructure filtering (`tools/infrastructure/*`)
- Unified HTTP endpoints: `/health`, `/tools`, `/metrics`, `/services`
- Service registry integration with health monitoring
- Performance tracking and infrastructure observability

**Example Usage**:
```bash
# All infrastructure tools loaded from contract at startup
# onex run infrastructure_hub --health-check
# onex run infrastructure_hub --list-services
```

**Implementation Pattern** (Copy from `canary_example_hub`):
```python
class InfrastructureHub(NodeHubBase):
    def __init__(self, container: ONEXContainer):
        contract_path = Path(__file__).parent / CONTRACT_FILENAME
        super().__init__(container, contract_path)  # 80+ lines of functionality!
```

## 🎯 Infrastructure Debug Intelligence Dashboard

Monitor infrastructure operations via web interface:
- **Dashboard**: http://localhost:8096/
- **Infrastructure Metrics**: http://localhost:8096/infrastructure
- **Service Health**: http://localhost:8096/services
- **Real-time monitoring** of infrastructure components and performance
- **Debug intelligence** integration for infrastructure incident analysis

## 📦 Infrastructure Service Ports

- Event Bus: port 8083 (HTTP)
- Infrastructure Hub: port 8085 (contract-driven)
- Consul: port 8500 (HTTP), port 8600 (DNS)
- Kafka: port 9092 (plaintext), port 9093 (SSL)
- Vault: port 8200 (HTTP)
- PostgreSQL: port 5432 (TCP)
- Debug Intelligence Dashboard: port 8096
- Check `docker-compose.infrastructure.yml` for service dependencies

## 📣 Infrastructure Key Reminders

- **Agent-first infrastructure**: Always delegate infrastructure coding to specialists
- **Service integration patterns**: Use adapters for external service integration
- **Contract-driven deployment**: All infrastructure configuration in contracts
- **Strong typing mandatory**: Never use `Any` in infrastructure components
- **Security-first design**: All infrastructure components must pass security audits
- **Monitoring by design**: All infrastructure tools must include observability
- **Event-driven architecture**: Infrastructure events flow through Kafka adapters
- **Connection pooling**: Database connections managed through dedicated managers

## 🚀 Infrastructure Migration Plan

### Phase 1: PostgreSQL Adapter Node Creation (FOUNDATIONAL)
Create PostgreSQL adapter node following message bus bridge pattern:

#### 1.1 PostgreSQL Adapter Architecture
**Message Bus Bridge Pattern**: Event envelopes → PostgreSQL Adapter → postgres_connection_manager → PostgreSQL Database

The adapter receives event envelopes from the message bus containing:
- SQL queries and parameters
- Transaction operations
- Connection pool management commands
- Database health check requests

#### 1.2 Contract-First Approach
- Create `src/omnibase_infra/nodes/postgres_adapter/v1_0_0/contract.yaml`
- Define node_type: "EFFECT" (message bus to database bridge)
- Define input_model/output_model for PostgreSQL operations
- Specify io_operations for queries, transactions, connections, health checks
- Define dependencies (postgres_connection_manager, event bus protocols)

#### 1.3 Shared Model Architecture (DRY Pattern)
**Shared models** (reusable across multiple nodes):
- `src/omnibase_infra/models/postgres/model_postgres_query_request.py`
- `src/omnibase_infra/models/postgres/model_postgres_transaction_request.py`  
- `src/omnibase_infra/models/postgres/model_postgres_health_response.py`

**Node-specific models** (adapter interface only):
- `model_postgres_adapter_input.py` - Message bus envelope payload
- `model_postgres_adapter_output.py` - Adapter response format

#### 1.4 Contract Model Dependencies
Reference shared models as dependencies in contract:
```yaml
dependencies:
  - name: "model_postgres_query_request"
    type: "model"
    class_name: "ModelPostgresQueryRequest"
    module: "omnibase_infra.models.postgres.model_postgres_query_request"
  - name: "model_postgres_transaction_request"
    type: "model"
    class_name: "ModelPostgresTransactionRequest"
    module: "omnibase_infra.models.postgres.model_postgres_transaction_request"
```

#### 1.5 Node Structure Creation
```
src/omnibase_infra/
├── models/                         # Shared models (reusable)
│   ├── postgres/
│   │   ├── model_postgres_query_request.py
│   │   ├── model_postgres_transaction_request.py
│   │   └── model_postgres_health_response.py
│   ├── consul/
│   │   ├── model_consul_kv_request.py
│   │   └── model_consul_service_registration.py
│   ├── kafka/
│   │   └── model_kafka_message.py
│   └── vault/
│       └── model_vault_secret_request.py
└── nodes/postgres_adapter/v1_0_0/
    ├── contract.yaml               # References shared models as dependencies
    ├── node.py                     # NodeEffectService implementation (bridge)
    ├── models/                     # Node-specific adapter models only
    │   ├── model_postgres_adapter_input.py
    │   └── model_postgres_adapter_output.py
    └── registry/
        └── registry_postgres_adapter.py
```

#### 1.6 Implementation Strategy
- **postgres_connection_manager.py remains as utility**: Used by the adapter for actual database operations
- **Adapter handles message bus integration**: Converts event envelopes to database calls
- **Follows consistent bridge pattern**: Same as consul_adapter, kafka_adapter, vault_adapter

#### 1.6 Import Updates
- Update all imports: `omnibase.core` → `omnibase_core`  
- Update all imports: `omnibase.exceptions` → `omnibase_core.exceptions`
- Update all imports: `omnibase.enums` → `omnibase_core.enums`
- Ensure proper OnexError chaining with CoreErrorCode usage

### Phase 2: Infrastructure Node Migration (Contract-Driven)

#### 2.1 Migration Priority Order
1. **consul_adapter** (Service discovery foundation)
2. **consul_projector** (Consul state projection)
3. **kafka_adapter** (Event streaming backbone)
4. **kafka_wrapper** (Message processing)
5. **vault_adapter** (Secret management)
6. **infrastructure_reducer** (State consolidation)
7. **infrastructure_orchestrator** (Workflow coordination)  
8. **message_aggregator** (Event aggregation)
9. **tool_resolver** (Dynamic tool resolution)
10. **group_gateway** (Service group management)

#### 2.2 Per-Node Migration Process
For each infrastructure node:

**Step 1: Contract Analysis**
- Read original `contract.yaml` from omnibase_3
- Identify node_type (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR)
- Extract model definitions from `definitions` section
- Document io_operations and dependencies

**Step 2: Contract Migration**
- Update contract with corrected naming (`tool_infrastructure_*` → node names)
- Update all import references (`omnibase.` → `omnibase_core.`)
- Ensure contract_version and node_version consistency
- Validate definitions section completeness

**Step 3: Model Extraction (Shared Model Pattern)**
- **Shared models**: Extract reusable models to `src/omnibase_infra/models/{service}/`
- **Node-specific models**: Only adapter input/output models in node's models/ directory
- **Contract dependencies**: Reference shared models as dependencies in contract
- Follow naming: contract "ModelConsulKVResponse" → `model_consul_kv_response.py`
- Ensure one model per file with proper Pydantic inheritance

**Step 4: Node Implementation**  
- Migrate `node.py` with updated imports and class naming
- Update base class inheritance (NodeEffectService, etc.)
- Update container injection and registry patterns
- Ensure proper OnexError chaining throughout

**Step 5: Registry Creation**
- Create registry/ directory with dependency injection setup
- Define protocol dependencies and injection patterns  
- Follow container injection pattern: `def __init__(self, container: ONEXContainer)`

### Phase 3: Target Directory Structure

#### 3.1 Final Repository Structure (Shared Model Pattern)
```
src/omnibase_infra/
├── infrastructure/                 # Legacy - will be removed
│   └── postgres_connection_manager.py  # Remains as utility
├── models/                         # Shared models (DRY pattern)
│   ├── postgres/
│   │   ├── model_postgres_query_request.py
│   │   ├── model_postgres_transaction_request.py
│   │   └── model_postgres_health_response.py
│   ├── consul/
│   │   ├── model_consul_kv_request.py
│   │   ├── model_consul_kv_response.py
│   │   └── model_consul_service_registration.py
│   ├── kafka/
│   │   ├── model_kafka_message.py
│   │   └── model_kafka_event_envelope.py
│   ├── vault/
│   │   ├── model_vault_secret_request.py
│   │   └── model_vault_secret_response.py
│   └── infrastructure/
│       ├── model_health_check.py
│       └── model_service_status.py
└── nodes/                         # Contract-driven node architecture
    ├── postgres_adapter/v1_0_0/
    │   ├── contract.yaml           # References shared postgres models as dependencies
    │   ├── node.py (NodeEffectService - message bus to database bridge)
    │   ├── models/                 # Node-specific adapter models only
    │   │   ├── model_postgres_adapter_input.py
    │   │   └── model_postgres_adapter_output.py
    │   └── registry/
    ├── consul_adapter/v1_0_0/
    │   ├── contract.yaml           # References shared consul models as dependencies
    │   ├── node.py (NodeEffectService - message bus to consul bridge)
    │   ├── models/                 # Node-specific adapter models only
    │   │   ├── model_consul_adapter_input.py
    │   │   └── model_consul_adapter_output.py
    │   └── registry/
    ├── kafka_adapter/v1_0_0/
    │   ├── contract.yaml           # References shared kafka models as dependencies
    │   ├── node.py (NodeEffectService - message bus to kafka bridge)
    │   ├── models/                 # Node-specific adapter models only
    │   │   ├── model_kafka_adapter_input.py
    │   │   └── model_kafka_adapter_output.py
    │   └── registry/
    ├── vault_adapter/v1_0_0/
    │   ├── contract.yaml           # References shared vault models as dependencies
    │   ├── node.py (NodeEffectService - message bus to vault bridge)
    │   ├── models/                 # Node-specific adapter models only
    │   │   ├── model_vault_adapter_input.py
    │   │   └── model_vault_adapter_output.py
    │   └── registry/
    ├── infrastructure_reducer/v1_0_0/
    │   ├── contract.yaml           # References shared infrastructure models
    │   ├── node.py (NodeReducerService - state consolidation)
    │   └── [...]
    ├── infrastructure_orchestrator/v1_0_0/
    │   ├── contract.yaml           # References shared infrastructure models
    │   ├── node.py (NodeOrchestratorService - workflow coordination)
    │   └── [...]
    └── [...other infrastructure nodes following same pattern]
```

#### 3.2 Shared Model Dependency Pattern

**Contract Model Dependencies** (consistent across all nodes):
```yaml
dependencies:
  # Protocol dependencies (existing pattern)
  - name: "protocol_event_bus"
    type: "protocol"
    class_name: "ProtocolEventBus"
    module: "omnibase_core.protocol.protocol_event_bus"

  # Shared model dependencies (new pattern)
  - name: "model_postgres_query_request"
    type: "model"
    class_name: "ModelPostgresQueryRequest"
    module: "omnibase_infra.models.postgres.model_postgres_query_request"
  - name: "model_consul_kv_request"
    type: "model"
    class_name: "ModelConsulKvRequest"
    module: "omnibase_infra.models.consul.model_consul_kv_request"
```

**Benefits**:
- ✅ **DRY principle**: No duplicate models across nodes
- ✅ **Consistent interfaces**: Same models used by multiple nodes
- ✅ **Versioning flexibility**: Shared models can evolve independently
- ✅ **Dependency injection**: Models resolved through container like protocols

#### 3.3 Contract Architecture Standards

**Required Contract Sections:**
- `contract_version`, `node_version`, `version` - Semantic versioning
- `node_name`, `contract_name`, `name` - Consistent naming
- `node_type` - EFFECT/COMPUTE/REDUCER/ORCHESTRATOR classification  
- `input_model`, `output_model` - Strongly typed I/O
- `dependencies` - Protocol-based dependency injection
- `io_operations` - For EFFECT nodes (external interactions)
- `definitions` - All models, schemas, responses defined here

**Model Definition Standards:**
- All models defined in contract `definitions` section
- One model per file in models/ directory  
- CamelCase model names: `ModelConsulKVResponse`
- snake_case file names: `model_consul_kv_response.py`
- Proper Pydantic inheritance and validation

### Phase 4: Validation & Testing

#### 4.1 Contract Validation
- Ensure all contracts follow ONEX compliance standards
- Validate model completeness (no missing definitions)
- Verify dependency injection patterns
- Check protocol resolution (no isinstance usage)

#### 4.2 Import Verification  
- All `omnibase.` imports updated to `omnibase_core.`
- OnexError chaining with CoreErrorCode usage
- Proper exception handling throughout nodes

#### 4.3 Architecture Compliance
- Registry injection pattern in all nodes
- Strong typing (zero Any type usage)  
- Contract-driven configuration (no hardcoded values)
- Protocol-based resolution for dependencies

### Phase 5: Cleanup & Documentation

#### 5.1 Legacy Cleanup
- Remove `src/omnibase_infra/infrastructure/` directory
- Update pyproject.toml scripts and entry points
- Clean up any legacy import references

#### 5.2 Documentation Updates  
- Update README.md with new node structure
- Document contract architecture patterns
- Provide migration examples for other repositories

---

**Bottom Line**: Claude Code for ONEX Infrastructure is agent-driven. Route through orchestrators, delegate to infrastructure specialists, never code directly. All infrastructure follows service integration patterns with strong typing and contract-driven configuration.
