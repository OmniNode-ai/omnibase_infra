# OmniNode Bridge Architecture

**Version**: 2.0 (Post Phase 1 & 2 Completion)
**Last Updated**: October 15, 2025
**Status**: Production MVP Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [ONEX v2.0 Compliance](#onex-v20-compliance)
4. [Core Components](#core-components)
5. [Bridge Nodes Architecture](#bridge-nodes-architecture)
6. [Event-Driven Architecture](#event-driven-architecture)
7. [Data Flow](#data-flow)
8. [Design Patterns](#design-patterns)
9. [Integration Points](#integration-points)
10. [Performance Architecture](#performance-architecture)
11. [Security Architecture](#security-architecture)
12. [Scalability and Resilience](#scalability-and-resilience)

---

## Executive Summary

**OmniNode Bridge** is a production-ready MVP foundation implementing ONEX v2.0 compliant bridge nodes for the omninode ecosystem. The architecture emphasizes:

- **Contract-Driven Design**: YAML-based configuration with subcontract composition
- **Event-Driven Coordination**: Kafka-based event streaming with OnexEnvelopeV1 format
- **High Performance**: Sub-2ms BLAKE3 hashing, <10ms API responses
- **Production Quality**: 92.8% test coverage, comprehensive observability
- **Future-Ready**: Designed for horizontal scaling and repository split

### Quick Overview
The following ASCII diagram provides a concise, 10000-foot view of the OmniNode Bridge system architecture. This serves as a **mental model** for understanding the flow of data and interactions between core components.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLIENT/EXTERNAL SYSTEMS                          │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ HTTP Requests
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     MetadataStampingService (FastAPI)                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ API Layer: POST /stamp, /validate, /hash, /health, /metrics       │  │
│  │  - Unified response format (O.N.E. v0.1)                           │  │
│  │  - Authentication & validation middleware                          │  │
│  │  - Rate limiting & circuit breakers                                │  │
│  └────────────────────┬───────────────────────────────────────────────┘  │
│                       │                                                   │
│  ┌────────────────────▼───────────────────────────────────────────────┐  │
│  │ BLAKE3HashGenerator: <2ms hash generation, pool of 100 instances   │  │
│  └────────────────────┬───────────────────────────────────────────────┘  │
│                       │                                                   │
│  ┌────────────────────▼───────────────────────────────────────────────┐  │
│  │ ProtocolFileTypeHandler: Multi-modal file type detection           │  │
│  └────────────────────┬───────────────────────────────────────────────┘  │
└────────────────────────┼───────────────────────────────────────────────┘
                        │ Invokes Bridge Orchestrator
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      NodeBridgeOrchestrator (ONEX v2.0)                  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Contract: ModelContractOrchestrator                                │  │
│  │  - Workflow coordination subcontract                               │  │
│  │  - Routing subcontract (MetadataStamping, OnexTree)                │  │
│  │  - FSM subcontract (PENDING → PROCESSING → COMPLETED/FAILED)       │  │
│  │  - Event type subcontract (Kafka event publishing)                 │  │
│  └────────────────────┬───────────────────────────────────────────────┘  │
│                       │                                                   │
│  Workflow Steps:      │                                                   │
│  1. Validate input    ├──► OnexTree Intelligence Service (optional)      │
│  2. Generate hash     │    AI-enhanced validation & routing               │
│  3. Create stamp      │                                                   │
│  4. Publish events ───┼──► Kafka/Redpanda (13 topics)                    │
│  5. Update FSM state  │                                                   │
└────────────────────────┼───────────────────────────────────────────────┘
                        │ Metadata stream
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       NodeBridgeReducer (ONEX v2.0)                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Contract: ModelContractReducer                                     │  │
│  │  - Aggregation subcontract (NAMESPACE_GROUPING primary)            │  │
│  │  - State management subcontract (PostgreSQL persistence)           │  │
│  │  - FSM subcontract (aggregation state tracking)                    │  │
│  │  - Caching subcontract (result caching)                            │  │
│  └────────────────────┬───────────────────────────────────────────────┘  │
│                       │                                                   │
│  Aggregation Flow:    │                                                   │
│  1. Stream metadata   │                                                   │
│  2. Group by strategy ├──► Batch size: 100 items                         │
│  3. Compute stats     ├──► Window: 5000ms                                │
│  4. Track FSM states  │                                                   │
│  5. Persist state ────┼──► PostgreSQL (transaction with rollback)        │
│  6. Return results    │                                                   │
└────────────────────────┼───────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     PostgreSQL Database (7 Tables)                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Tables:                                                            │  │
│  │  • workflow_executions    - Orchestrator workflow tracking         │  │
│  │  • workflow_steps         - Individual step results                │  │
│  │  • fsm_transitions        - State machine transition history       │  │
│  │  • bridge_states          - Reducer aggregation states             │  │
│  │  • node_registrations     - Service discovery & health             │  │
│  │  • metadata_stamps        - Stamp metadata storage                 │  │
│  │  • event_logs             - Kafka event tracing                    │  │
│  └────────────────────┬───────────────────────────────────────────────┘  │
│                       │                                                   │
│  Performance:         │                                                   │
│  - CRUD: <10ms (p95)  │                                                   │
│  - Queries: <50ms (p95)                                                  │
│  - Connection pool: 10-50 connections                                    │
│  - Circuit breaker: 5 failures → open                                    │
└───────────────────────┼──────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   Kafka/Redpanda Event Infrastructure                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Topics (13 total):                                                 │  │
│  │                                                                    │  │
│  │ Request Topics (omniclaude → omniarchon):                          │  │
│  │  • omninode_codegen_request_analyze_v1                             │  │
│  │  • omninode_codegen_request_validate_v1                            │  │
│  │  • omninode_codegen_request_pattern_v1                             │  │
│  │  • omninode_codegen_request_mixin_v1                               │  │
│  │                                                                    │  │
│  │ Response Topics (omniarchon → omniclaude):                         │  │
│  │  • omninode_codegen_response_analyze_v1                            │  │
│  │  • omninode_codegen_response_validate_v1                           │  │
│  │  • omninode_codegen_response_pattern_v1                            │  │
│  │  • omninode_codegen_response_mixin_v1                              │  │
│  │                                                                    │  │
│  │ Status Topics (bidirectional):                                     │  │
│  │  • omninode_codegen_status_session_v1 (6 partitions)               │  │
│  │                                                                    │  │
│  │ DLQ Topics (failed events):                                        │  │
│  │  • omninode_codegen_dlq_analyze_v1                                 │  │
│  │  • omninode_codegen_dlq_validate_v1                                │  │
│  │  • omninode_codegen_dlq_pattern_v1                                 │  │
│  │  • omninode_codegen_dlq_mixin_v1                                   │  │
│  └────────────────────┬───────────────────────────────────────────────┘  │
│                       │                                                   │
│  Event Format: OnexEnvelopeV1                                            │
│  - correlation_id: UUID-based request/response correlation               │
│  - Partitioning by correlation_id for ordered processing                 │
│  - Producer success rate: 100% in tests                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Design Principles**:
1. **Separation of Concerns**: Each component has a single responsibility
2. **Contract-Driven**: YAML contracts define node capabilities
3. **Event-Driven**: Loose coupling via Kafka event streams
4. **Observable**: Comprehensive metrics, logging, and tracing
5. **Resilient**: Circuit breakers, retries, graceful degradation

---

## System Overview

### Purpose and Scope

**OmniNode Bridge** serves as the **MVP foundation** for the omninode ecosystem, providing:

1. **Metadata Stamping Service**: BLAKE3 hash generation with O.N.E. v0.1 protocol compliance
2. **Bridge Nodes**: ONEX v2.0 compliant orchestrator, reducer, and registry nodes
3. **Event Infrastructure**: Kafka-based event streaming with 13 topics
4. **Database Layer**: PostgreSQL persistence with 7 tables and 50+ indexes
5. **Intelligence Integration**: OnexTree AI service integration (optional)

**This is NOT**:
- ❌ A production deployment (monorepo structure, will be split)
- ❌ An enterprise feature-complete system (Phase 3-5 deferred)
- ❌ A general-purpose stamping service (focused on omninode ecosystem)

### System Context

```
┌────────────────────────────────────────────────────────────────────┐
│                       External Systems                             │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ omniclaude   │  │ omniarchon   │  │ Other omninode services  │ │
│  │ (Client)     │  │ (Archon MCP) │  │ (Future)                 │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────────────┘ │
│         │                 │                   │                    │
└─────────┼─────────────────┼───────────────────┼────────────────────┘
          │                 │                   │
          │ HTTP Requests   │ Kafka Events      │ Service Calls
          │                 │                   │
┌─────────▼─────────────────▼───────────────────▼────────────────────┐
│                       OmniNode Bridge                              │
│                                                                    │
│  ┌───────────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ Metadata Stamping │  │ Bridge Nodes  │  │ Event            │  │
│  │ Service           │  │ (Orchestrator,│  │ Infrastructure   │  │
│  │                   │  │  Reducer,     │  │ (Kafka/Redpanda) │  │
│  │                   │  │  Registry)    │  │                  │  │
│  └─────────┬─────────┘  └───────┬───────┘  └────────┬─────────┘  │
│            │                    │                    │            │
│            └────────────────────┼────────────────────┘            │
│                                 │                                 │
│                  ┌──────────────▼──────────────┐                  │
│                  │ PostgreSQL Database         │                  │
│                  │ (7 tables, 50+ indexes)     │                  │
│                  └─────────────────────────────┘                  │
└────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Primary development language |
| **Web Framework** | FastAPI | 0.104.1 | Async HTTP API |
| **Database** | PostgreSQL | 15 | Persistent storage |
| **Event Streaming** | Redpanda (Kafka) | Latest | Event infrastructure |
| **Service Discovery** | Consul | Latest | Service registry |
| **Hashing** | BLAKE3 | 0.4.1 | Cryptographic hashing |
| **Serialization** | Pydantic | v2.5.0 | Data validation |
| **Logging** | structlog | 23.2.0 | Structured logging |
| **Metrics** | Prometheus | 0.19.0 | Observability |
| **Container** | Docker | Latest | Development environment |
| **Orchestration** | Docker Compose | v2 | Local development |
| **Migration** | Alembic | Latest | Database migrations |
| **Testing** | pytest | 7.4.3 | Test framework |
| **Type Checking** | mypy | 1.7.1 | Static type analysis |
| **Code Quality** | ruff, black | Latest | Linting and formatting |

---

## ONEX v2.0 Compliance

### ONEX Architecture Overview

**ONEX** (One.Node.Enterprise) v2.0 is the architectural standard for omninode ecosystem nodes. Key principles:

1. **Suffix-Based Naming**: All entities use suffix-based naming
2. **Contract-Driven**: Nodes configured via YAML contracts
3. **Subcontract Composition**: Capabilities added via subcontracts
4. **Dependency Injection**: Services provided via ModelONEXContainer
5. **Four Node Types**: Effect, Compute, Reducer, Orchestrator

### Naming Convention

```python
# Classes
Node<Name><Type>              # e.g., NodeBridgeOrchestrator, NodeBridgeReducer
Model<Name>                   # e.g., ModelBridgeState, ModelStampMetadata
Enum<Name>                    # e.g., EnumWorkflowState, EnumAggregationType
ModelContract<Type>           # e.g., ModelContractOrchestrator

# Files
node_<name>_<type>.py         # e.g., node_bridge_orchestrator.py
model_<name>.py               # e.g., model_bridge_state.py
enum_<name>.py                # e.g., enum_workflow_state.py
model_contract_<type>.py      # e.g., model_contract_orchestrator.py
```

### Method Signatures by Node Type

```python
# Orchestrator
async def execute_orchestration(
    self,
    contract: ModelContractOrchestrator
) -> ModelStampResponseOutput:
    """Coordinate workflow execution."""

# Reducer
async def execute_reduction(
    self,
    contract: ModelContractReducer
) -> ModelReducerOutputState:
    """Aggregate data and manage state."""

# Effect (not implemented in MVP)
async def execute_effect(
    self,
    contract: ModelContractEffect
) -> Any:
    """Perform side effects (I/O, APIs)."""

# Compute (not implemented in MVP)
async def execute_compute(
    self,
    contract: ModelContractCompute
) -> Any:
    """Perform pure computation."""
```

### Contract Structure

```yaml
# orchestrator/v1_0_0/contracts/contract.yaml

contract_version: {major: 1, minor: 0, patch: 0}
node_type: ORCHESTRATOR

# Base configuration
name: "NodeBridgeOrchestrator"
description: "Stamping workflow coordination"
version: "1.0.0"

# Subcontract composition
workflow_coordination:
  $ref: "./subcontracts/workflow_steps.yaml"

routing:
  $ref: "./subcontracts/routing_rules.yaml"

fsm:
  $ref: "./subcontracts/fsm_states.yaml"

event_type:
  $ref: "./subcontracts/events.yaml"
```

### Available Subcontracts

From `omnibase_core.models.contracts.subcontracts`:

| Subcontract | Purpose | Applicable Nodes |
|-------------|---------|------------------|
| `ModelAggregationSubcontract` | Data aggregation strategies | Reducer |
| `ModelStateManagementSubcontract` | State persistence (PostgreSQL) | Reducer, Orchestrator |
| `ModelFSMSubcontract` | Finite state machine patterns | Orchestrator, Reducer |
| `ModelWorkflowCoordinationSubcontract` | Workflow orchestration steps | Orchestrator |
| `ModelRoutingSubcontract` | Service routing & load balancing | Orchestrator |
| `ModelEventTypeSubcontract` | Event definitions & publishing | Orchestrator, Effect |
| `ModelCachingSubcontract` | Cache strategies | All |
| `ModelConfigurationSubcontract` | Configuration management | All |

---

## Core Components

### 1. MetadataStampingService

**Purpose**: FastAPI-based HTTP service for metadata stamping and BLAKE3 hash generation.

**Key Features**:
- BLAKE3 hash generation (<2ms p99 latency)
- O.N.E. v0.1 protocol compliance
- Multi-tenant namespace support
- Unified response format
- Circuit breaker patterns
- Pool exhaustion monitoring

**API Endpoints**:
```python
POST   /stamp              # Create metadata stamp
POST   /validate           # Validate existing stamps
POST   /hash               # Generate BLAKE3 hash
GET    /stamp/{file_hash}  # Retrieve stamp by hash
GET    /health             # Service health check
GET    /metrics            # Prometheus metrics
```

**Performance**:
- Hash generation: <1ms average (<2ms p99)
- API response: <5ms average (<10ms p95)
- Concurrent requests: 1500+ sustained
- Memory usage: <300MB under load

**Architecture**:
```python
# src/metadata_stamping/main.py

app = FastAPI(title="Metadata Stamping Service")

@app.post("/stamp")
async def create_stamp(request: ModelStampRequest) -> ModelStampResponse:
    # 1. Validate input
    validate_stamp_request(request)

    # 2. Generate BLAKE3 hash
    hash_generator = container.get_service('blake3_generator')
    file_hash = await hash_generator.generate_hash(request.content)

    # 3. Detect file type
    file_type_handler = container.get_service('file_type_handler')
    content_type = file_type_handler.detect(request.content)

    # 4. Create stamp with O.N.E. v0.1 compliance
    stamp = create_metadata_stamp(
        file_hash=file_hash,
        namespace=request.namespace,
        content_type=content_type,
        metadata_version="0.1"
    )

    # 5. Invoke bridge orchestrator
    orchestrator = container.get_service('bridge_orchestrator')
    result = await orchestrator.execute_orchestration(
        ModelContractOrchestrator(
            correlation_id=uuid4(),
            input_data=stamp
        )
    )

    # 6. Return unified response
    return ModelStampResponse(
        stamp_id=result.stamp_id,
        file_hash=result.file_hash,
        stamped_content=result.stamped_content,
        metadata=result.stamp_metadata
    )
```

### 2. BLAKE3HashGenerator

**Purpose**: High-performance cryptographic hash generation with pooling and optimization.

**Key Features**:
- Pre-allocated hasher pool (100 instances)
- Adaptive buffer sizing (8KB-1MB)
- Thread pool for CPU-intensive operations
- Zero-allocation hot path
- Weak references for automatic cleanup

**Implementation**:
```python
# src/metadata_stamping/hash_generator.py

class BLAKE3HashGenerator:
    """
    High-performance BLAKE3 hash generator.

    Performance targets:
    - <2ms per operation (p99)
    - <1ms average
    - 1000+ concurrent operations

    Optimization techniques:
    - Memory pool: Pre-allocated hashers
    - Buffer sizing: Adaptive based on file size
    - Thread pool: Non-blocking CPU-intensive ops
    """

    def __init__(self, pool_size: int = 100):
        # Pre-allocate hasher pool
        self._hasher_pool = [blake3.blake3() for _ in range(pool_size)]
        self._pool_semaphore = asyncio.Semaphore(pool_size)

        # Thread pool for CPU-intensive work
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    async def generate_hash(self, content: bytes) -> str:
        """
        Generate BLAKE3 hash with <2ms target latency.

        Args:
            content: Bytes to hash

        Returns:
            Hexadecimal hash string

        Performance:
            - Small files (<1KB): ~0.5ms
            - Medium files (1-100KB): ~1ms
            - Large files (>100KB): ~1.5ms
        """
        async with self._pool_semaphore:
            # Get hasher from pool
            hasher = self._get_hasher_from_pool()

            # Adaptive buffer sizing
            buffer_size = self._determine_buffer_size(len(content))

            # Execute hash in thread pool (CPU-intensive)
            loop = asyncio.get_event_loop()
            hash_result = await loop.run_in_executor(
                self._thread_pool,
                self._compute_hash,
                hasher,
                content,
                buffer_size
            )

            # Return hasher to pool
            self._return_hasher_to_pool(hasher)

            return hash_result

    def _determine_buffer_size(self, content_size: int) -> int:
        """Adaptive buffer sizing based on content size."""
        if content_size < 1024:           # <1KB
            return 8192                    # 8KB buffer
        elif content_size < 102400:       # <100KB
            return 65536                   # 64KB buffer
        else:
            return 1048576                 # 1MB buffer
```

**Performance Metrics**:
```python
# Performance benchmarks from tests
{
    "small_files_1kb": {
        "avg_latency_ms": 0.5,
        "p99_latency_ms": 0.8,
        "throughput_ops_sec": 2000
    },
    "medium_files_100kb": {
        "avg_latency_ms": 1.0,
        "p99_latency_ms": 1.5,
        "throughput_ops_sec": 1000
    },
    "large_files_1mb": {
        "avg_latency_ms": 1.5,
        "p99_latency_ms": 1.9,
        "throughput_ops_sec": 650
    }
}
```

### 3. ProtocolFileTypeHandler

**Purpose**: Multi-modal file type detection with O.N.E. v0.1 protocol compliance.

**Supported Types**:
- **Images**: .jpg, .jpeg, .png, .gif, .bmp, .webp
- **Documents**: .pdf, .doc, .docx, .txt, .md
- **Audio**: .mp3, .wav, .flac, .aac, .ogg
- **Video**: .mp4, .avi, .mkv, .mov, .webm
- **Archives**: .zip, .tar, .gz, .rar, .7z

**Implementation**:
```python
# src/metadata_stamping/file_type_handler.py

class ProtocolFileTypeHandler:
    """
    O.N.E. v0.1 compliant file type detection.

    Detection strategies:
    1. Magic number analysis (libmagic)
    2. File extension fallback
    3. Content inspection for text types
    """

    def detect(self, content: bytes, file_path: str | None = None) -> str:
        """
        Detect MIME type with O.N.E. v0.1 compliance.

        Args:
            content: File content bytes
            file_path: Optional file path for extension-based fallback

        Returns:
            MIME type string (e.g., "application/pdf")
        """
        # Strategy 1: Magic number detection
        mime_type = magic.from_buffer(content, mime=True)

        if mime_type and mime_type != "application/octet-stream":
            return mime_type

        # Strategy 2: Extension-based fallback
        if file_path:
            extension = Path(file_path).suffix.lower()
            mime_type = self._extension_to_mime.get(extension)

            if mime_type:
                return mime_type

        # Strategy 3: Content inspection
        if self._is_text_content(content):
            return "text/plain"

        # Default fallback
        return "application/octet-stream"
```

### 4. PostgresConnectionManager

**Purpose**: High-performance database connection management with circuit breaker and pooling.

**Key Features**:
- Connection pooling (10-50 connections)
- Prepared statement caching
- Circuit breaker pattern
- Pool exhaustion monitoring (>90% triggers alert)
- Transaction management with rollback
- Health monitoring

**Implementation**:
```python
# src/omninode_bridge/persistence/postgres_connection_manager.py

class PostgresConnectionManager:
    """
    Production-grade PostgreSQL connection manager.

    Features:
    - Connection pooling (asyncpg)
    - Circuit breaker for resilience
    - Pool exhaustion detection
    - Prepared statement caching
    - Transaction support
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool: Pool | None = None
        self._circuit_breaker = DatabaseCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            half_open_max_calls=3
        )

    async def initialize(self) -> None:
        """Initialize connection pool with circuit breaker protection."""
        try:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.pool_min_size,    # 10
                max_size=self.config.pool_max_size,    # 50
                command_timeout=self.config.command_timeout,  # 60s
                max_queries=50000,  # Prepared statement cache
                max_cached_statement_lifetime=300,  # 5 minutes
            )

            logger.info("PostgreSQL connection pool initialized",
                       pool_size=self._pool.get_size(),
                       pool_max=self._pool.get_max_size())

        except Exception as e:
            self._circuit_breaker.record_failure()
            raise OnexError(
                code=CoreErrorCode.RESOURCE_ERROR,
                message="Failed to initialize database connection pool",
                details={"error": str(e)}
            )

    async def acquire(self) -> Connection:
        """
        Acquire connection from pool with circuit breaker.

        Monitors pool exhaustion and logs warnings when >90% utilized.
        """
        if self._circuit_breaker.is_open():
            raise OnexError(
                code=CoreErrorCode.RESOURCE_ERROR,
                message="Database circuit breaker is OPEN",
                details={"state": "OPEN", "reason": "Too many failures"}
            )

        try:
            # Check pool exhaustion
            stats = self.get_pool_stats()
            if stats['utilization_percent'] > 90.0:
                logger.warning("Pool exhaustion detected",
                              utilization=stats['utilization_percent'],
                              used=stats['used_connections'],
                              max=stats['pool_max'])

            # Acquire connection
            conn = await self._pool.acquire()
            self._circuit_breaker.record_success()
            return conn

        except Exception as e:
            self._circuit_breaker.record_failure()
            raise OnexError(
                code=CoreErrorCode.RESOURCE_ERROR,
                message="Failed to acquire database connection",
                details={"error": str(e)}
            )

    def get_pool_stats(self) -> dict[str, Any]:
        """
        Get connection pool statistics with exhaustion metrics.

        Returns:
            {
                "pool_size": 45,
                "pool_free": 5,
                "pool_max": 50,
                "used_connections": 45,
                "utilization_percent": 90.0,
                "exhaustion_threshold_percent": 90.0,
                "exhaustion_warning_count": 12
            }
        """
        if not self._pool:
            return {}

        pool_size = self._pool.get_size()
        pool_free = self._pool.get_idle_size()
        pool_max = self._pool.get_max_size()
        used = pool_size - pool_free
        utilization = (used / pool_max * 100) if pool_max > 0 else 0

        return {
            "pool_size": pool_size,
            "pool_free": pool_free,
            "pool_max": pool_max,
            "used_connections": used,
            "utilization_percent": round(utilization, 2),
            "exhaustion_threshold_percent": 90.0,
            "exhaustion_warning_count": self._exhaustion_count
        }
```

**Performance Characteristics**:
```python
# Database operation performance from Phase 2
{
    "crud_operations": {
        "target_p95_ms": 20,
        "actual_p95_ms": 10,
        "status": "✅ Exceeded"
    },
    "query_operations": {
        "target_p95_ms": 100,
        "actual_p95_ms": 50,
        "status": "✅ Exceeded"
    },
    "connection_pool": {
        "efficiency_target": 0.90,
        "efficiency_actual": 1.00,
        "status": "✅ Exceeded"
    }
}
```

### 5. KafkaEventProducer

**Purpose**: Kafka event publishing with OnexEnvelopeV1 format and DLQ support.

**Key Features**:
- OnexEnvelopeV1 standardized envelope
- Correlation ID tracking
- Partitioning by correlation_id
- DLQ (Dead Letter Queue) for failed events
- aiokafka (primary) with confluent-kafka (fallback)
- Producer success rate: 100% in tests

**Implementation**:
```python
# src/omninode_bridge/events/kafka_producer.py

class KafkaEventProducer:
    """
    Production Kafka event producer with OnexEnvelopeV1 format.

    Features:
    - OnexEnvelopeV1 envelope format
    - Correlation ID tracking
    - DLQ for failed events
    - Dual client support (aiokafka, confluent-kafka)
    """

    async def publish_event(
        self,
        topic: str,
        event: BaseModel,
        correlation_id: UUID,
        session_id: UUID | None = None
    ) -> None:
        """
        Publish event to Kafka with OnexEnvelopeV1 envelope.

        Args:
            topic: Kafka topic name
            event: Event payload (Pydantic model)
            correlation_id: UUID for request/response correlation
            session_id: Optional session tracking ID

        Envelope Format:
            {
                "envelope_version": "1.0",
                "correlation_id": "uuid...",
                "session_id": "uuid..." | null,
                "event_type": "CodegenAnalysisRequest",
                "timestamp": "2025-10-15T12:34:56.789Z",
                "payload": { ... }
            }
        """
        # Create OnexEnvelopeV1
        envelope = OnexEnvelopeV1(
            envelope_version="1.0",
            correlation_id=correlation_id,
            session_id=session_id,
            event_type=event.__class__.__name__,
            timestamp=datetime.utcnow(),
            payload=event.model_dump()
        )

        # Serialize to JSON
        message = json.dumps(envelope.model_dump(), default=str)

        try:
            # Publish to Kafka (partition by correlation_id for ordering)
            await self._producer.send(
                topic=topic,
                value=message.encode('utf-8'),
                key=str(correlation_id).encode('utf-8')  # Partition by correlation_id
            )

            logger.info("Event published",
                       topic=topic,
                       event_type=envelope.event_type,
                       correlation_id=str(correlation_id))

        except Exception as e:
            # Publish to DLQ
            dlq_topic = f"{topic}_dlq"
            await self._publish_to_dlq(dlq_topic, envelope, error=str(e))

            logger.error("Event publish failed, sent to DLQ",
                        topic=topic,
                        dlq_topic=dlq_topic,
                        error=str(e))
```

**Kafka Topics**:
```python
# 13 topics defined in docs/events/codegen-topics-config.yaml

REQUEST_TOPICS = [
    "omninode_codegen_request_analyze_v1",
    "omninode_codegen_request_validate_v1",
    "omninode_codegen_request_pattern_v1",
    "omninode_codegen_request_mixin_v1",
]

RESPONSE_TOPICS = [
    "omninode_codegen_response_analyze_v1",
    "omninode_codegen_response_validate_v1",
    "omninode_codegen_response_pattern_v1",
    "omninode_codegen_response_mixin_v1",
]

STATUS_TOPICS = [
    "omninode_codegen_status_session_v1",  # 6 partitions
]

DLQ_TOPICS = [
    "omninode_codegen_dlq_analyze_v1",
    "omninode_codegen_dlq_validate_v1",
    "omninode_codegen_dlq_pattern_v1",
    "omninode_codegen_dlq_mixin_v1",
]
```

---

## Bridge Nodes Architecture

### Node Type Overview

OmniNode Bridge implements 3 of the 4 ONEX v2.0 node types:

1. **NodeBridgeOrchestrator** (Orchestrator) - Workflow coordination
2. **NodeBridgeReducer** (Reducer) - Aggregation and state management
3. **NodeBridgeRegistry** (Registry) - Service discovery (limited implementation)

**Not implemented** (future scope):
- Effect nodes - I/O and external side effects
- Compute nodes - Pure computation transformations

### NodeBridgeOrchestrator

**Purpose**: Coordinate multi-step stamping workflows with FSM state management.

**Contract Structure**:
```yaml
# orchestrator/v1_0_0/contracts/contract.yaml

contract_version: {major: 1, minor: 0, patch: 0}
node_type: ORCHESTRATOR

workflow_coordination:
  steps:
    - name: "validate_input"
      type: "validation"
      timeout_ms: 1000
    - name: "onextree_intelligence"
      type: "routing"
      optional: true
      timeout_ms: 5000
    - name: "hash_generation"
      type: "compute"
      timeout_ms: 2000
    - name: "stamp_creation"
      type: "effect"
      timeout_ms: 1000
    - name: "event_publishing"
      type: "effect"
      timeout_ms: 500

routing:
  services:
    - name: "metadata_stamping_service"
      url: "${METADATA_STAMPING_SERVICE_URL}"
      health_check_path: "/health"
    - name: "onextree_intelligence"
      url: "${ONEXTREE_SERVICE_URL}"
      health_check_path: "/health"
      optional: true

fsm:
  initial_state: "PENDING"
  states:
    - name: "PENDING"
      transitions: ["PROCESSING"]
    - name: "PROCESSING"
      transitions: ["COMPLETED", "FAILED"]
    - name: "COMPLETED"
      terminal: true
    - name: "FAILED"
      terminal: true

event_type:
  events:
    - WORKFLOW_STARTED
    - WORKFLOW_COMPLETED
    - WORKFLOW_FAILED
    - STEP_COMPLETED
    - STATE_TRANSITION
    - HASH_GENERATED
    - STAMP_CREATED
    - INTELLIGENCE_REQUESTED
    - INTELLIGENCE_RECEIVED
```

**Implementation**:
```python
# src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py

class NodeBridgeOrchestrator(NodeOrchestrator):
    """
    Bridge Orchestrator for stamping workflow coordination.

    Performance:
    - Standard workflow: <50ms
    - With OnexTree intelligence: <150ms
    - Throughput: 100+ workflows/second

    FSM States: PENDING → PROCESSING → COMPLETED/FAILED
    """

    async def execute_orchestration(
        self,
        contract: ModelContractOrchestrator
    ) -> ModelStampResponseOutput:
        """
        Execute stamping workflow with FSM state management.

        Workflow Steps:
        1. Validate input → FSM: PENDING
        2. Transition FSM: PENDING → PROCESSING
        3. Route to OnexTree (optional)
        4. Route to MetadataStamping for hash
        5. Create stamp with O.N.E. v0.1
        6. Publish events to Kafka
        7. Transition FSM: PROCESSING → COMPLETED
        8. Return stamped content

        Args:
            contract: Orchestrator contract with workflow config

        Returns:
            ModelStampResponseOutput with stamp metadata

        Raises:
            OnexError: If workflow execution fails
        """
        workflow_id = contract.correlation_id
        start_time = time.time()

        try:
            # Step 1: Validate input
            self._validate_input(contract)

            # Step 2: Initialize FSM
            self._fsm[workflow_id] = EnumWorkflowState.PENDING
            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_STARTED,
                {"workflow_id": str(workflow_id)}
            )

            # Step 3: Transition to PROCESSING
            await self._transition_state(
                workflow_id,
                EnumWorkflowState.PENDING,
                EnumWorkflowState.PROCESSING
            )

            # Step 4: Execute workflow steps
            workflow_results = await self._execute_workflow_steps(
                contract,
                workflow_id
            )

            # Step 5: Create stamp response
            stamp_response = self._create_stamp_response(
                workflow_results,
                workflow_id,
                start_time
            )

            # Step 6: Transition to COMPLETED
            await self._transition_state(
                workflow_id,
                EnumWorkflowState.PROCESSING,
                EnumWorkflowState.COMPLETED
            )

            # Step 7: Publish completion event
            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_COMPLETED,
                {
                    "workflow_id": str(workflow_id),
                    "stamp_id": stamp_response.stamp_id,
                    "processing_time_ms": stamp_response.processing_time_ms
                }
            )

            return stamp_response

        except Exception as e:
            # Transition to FAILED
            await self._transition_state(
                workflow_id,
                EnumWorkflowState.PROCESSING,
                EnumWorkflowState.FAILED
            )

            await self._publish_event(
                EnumWorkflowEvent.WORKFLOW_FAILED,
                {
                    "workflow_id": str(workflow_id),
                    "error": str(e)
                }
            )

            raise OnexError(
                code=CoreErrorCode.OPERATION_FAILED,
                message="Workflow execution failed",
                details={"workflow_id": str(workflow_id), "error": str(e)}
            )
```

**Performance Characteristics**:
```python
# Performance benchmarks from Phase 1
{
    "standard_workflow": {
        "target_ms": 50,
        "actual_avg_ms": 45,
        "actual_p95_ms": 48,
        "status": "✅ Met"
    },
    "with_onextree_intelligence": {
        "target_ms": 150,
        "actual_avg_ms": 140,
        "actual_p95_ms": 145,
        "status": "✅ Met"
    },
    "throughput": {
        "target_workflows_per_sec": 100,
        "actual_workflows_per_sec": 150,
        "status": "✅ Exceeded"
    }
}
```

### NodeBridgeReducer

**Purpose**: Aggregate stamping metadata across workflows and manage bridge state persistence.

**Contract Structure**:
```yaml
# reducer/v1_0_0/contracts/contract.yaml

contract_version: {major: 1, minor: 0, patch: 0}
node_type: REDUCER

aggregation:
  strategy: "NAMESPACE_GROUPING"
  window:
    mode: "time_based"
    size_ms: 5000
  batch:
    size: 100
    max_wait_ms: 1000

state_management:
  persistence:
    backend: "postgresql"
    table: "bridge_states"
    transaction_isolation: "READ_COMMITTED"
  checkpointing:
    enabled: true
    interval_ms: 1000

fsm:
  states:
    - name: "IDLE"
      transitions: ["AGGREGATING"]
    - name: "AGGREGATING"
      transitions: ["PERSISTING", "IDLE"]
    - name: "PERSISTING"
      transitions: ["IDLE"]

caching:
  enabled: true
  backend: "memory"
  ttl_ms: 60000
  max_size: 1000
```

**Implementation**:
```python
# src/omninode_bridge/nodes/reducer/v1_0_0/node.py

class NodeBridgeReducer(NodeReducer):
    """
    Bridge Reducer for metadata aggregation and state management.

    Performance:
    - Throughput: >1000 items/second
    - Latency: <100ms for 1000 items
    - Streaming with windowing (5000ms windows)

    FSM States: IDLE ↔ AGGREGATING ↔ PERSISTING
    """

    async def execute_reduction(
        self,
        contract: ModelContractReducer
    ) -> ModelReducerOutputState:
        """
        Execute metadata aggregation with streaming architecture.

        Aggregation Strategy:
        1. Stream metadata from input (async iterator)
        2. Group by namespace (primary) using windowing
        3. Compute aggregations (count, sum, avg, distinct)
        4. Track FSM states for each workflow
        5. Persist aggregated state to PostgreSQL
        6. Return aggregation results

        Args:
            contract: Reducer contract with aggregation config

        Returns:
            ModelReducerOutputState with aggregation results

        Performance:
            - Target: >1000 items/second
            - Latency: <100ms for 1000 items
        """
        start_time = time.time()

        # Initialize aggregation state
        aggregated_data: dict[str, dict[str, Any]] = {}
        total_items = 0
        total_size_bytes = 0
        fsm_states: dict[str, str] = {}

        # Stream metadata with windowing
        async for batch in self._stream_metadata(contract, batch_size=100):
            total_items += len(batch)

            for item in batch:
                # Group by namespace
                namespace = item.namespace

                if namespace not in aggregated_data:
                    aggregated_data[namespace] = {
                        "total_stamps": 0,
                        "total_size_bytes": 0,
                        "file_types": set(),
                        "workflow_ids": set()
                    }

                # Accumulate aggregations
                aggregated_data[namespace]["total_stamps"] += 1
                aggregated_data[namespace]["total_size_bytes"] += item.file_size
                aggregated_data[namespace]["file_types"].add(item.content_type)
                aggregated_data[namespace]["workflow_ids"].add(str(item.workflow_id))

                # Track FSM state
                fsm_states[str(item.workflow_id)] = item.workflow_state

                # Update totals
                total_size_bytes += item.file_size

        # Persist aggregated state to PostgreSQL
        await self._persist_state(aggregated_data, fsm_states, contract)

        # Calculate performance metrics
        duration_ms = (time.time() - start_time) * 1000
        items_per_second = (total_items / duration_ms * 1000) if duration_ms > 0 else 0

        # Return aggregation results
        return ModelReducerOutputState(
            aggregation_type=EnumAggregationType.NAMESPACE_GROUPING,
            total_items=total_items,
            total_size_bytes=total_size_bytes,
            namespaces=list(aggregated_data.keys()),
            aggregations={
                ns: {
                    "total_stamps": data["total_stamps"],
                    "total_size_bytes": data["total_size_bytes"],
                    "file_types": list(data["file_types"]),
                    "workflow_ids": list(data["workflow_ids"])
                }
                for ns, data in aggregated_data.items()
            },
            fsm_states=fsm_states,
            aggregation_duration_ms=duration_ms,
            items_per_second=items_per_second
        )
```

**Performance Characteristics**:
```python
# Performance benchmarks from Phase 1
{
    "throughput": {
        "target_items_per_sec": 1000,
        "actual_items_per_sec": 1100,
        "status": "✅ Exceeded"
    },
    "latency_1000_items": {
        "target_ms": 100,
        "actual_avg_ms": 85,
        "actual_p95_ms": 95,
        "status": "✅ Exceeded"
    },
    "postgresql_persistence": {
        "target_p95_ms": 50,
        "actual_p95_ms": 30,
        "status": "✅ Exceeded"
    }
}
```

### NodeBridgeRegistry

**Purpose**: Service discovery and node registration (limited MVP implementation).

**Key Features**:
- Node registration with health metadata
- Service discovery via Consul integration
- Health monitoring
- Circuit breaker integration

**Status**: Limited implementation in MVP, full registry functionality deferred to Phase 3+.

---

## Event-Driven Architecture

### Kafka Topic Architecture

**13 Topics** organized by direction and purpose:

```
Request Topics (omniclaude → omniarchon):
├── omninode_codegen_request_analyze_v1     # PRD analysis requests
├── omninode_codegen_request_validate_v1    # Code validation requests
├── omninode_codegen_request_pattern_v1     # Pattern matching requests
└── omninode_codegen_request_mixin_v1       # Mixin recommendation requests

Response Topics (omniarchon → omniclaude):
├── omninode_codegen_response_analyze_v1    # Analysis results
├── omninode_codegen_response_validate_v1   # Validation results
├── omninode_codegen_response_pattern_v1    # Pattern matches
└── omninode_codegen_response_mixin_v1      # Mixin recommendations

Status Topics (bidirectional):
└── omninode_codegen_status_session_v1      # Real-time session status (6 partitions)

DLQ Topics (failed events):
├── omninode_codegen_dlq_analyze_v1         # Failed analysis events
├── omninode_codegen_dlq_validate_v1        # Failed validation events
├── omninode_codegen_dlq_pattern_v1         # Failed pattern events
└── omninode_codegen_dlq_mixin_v1           # Failed mixin events
```

### OnexEnvelopeV1 Format

**Standardized event envelope** for all Kafka messages:

```python
class OnexEnvelopeV1(BaseModel):
    """
    Standardized event envelope for all Kafka messages.

    Ensures:
    - Version compatibility
    - Correlation tracking
    - Event type identification
    - Temporal ordering
    """
    envelope_version: str = "1.0"
    correlation_id: UUID
    session_id: UUID | None = None
    event_type: str
    timestamp: datetime
    payload: dict[str, Any]
```

**Example Event**:
```json
{
  "envelope_version": "1.0",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "event_type": "CodegenAnalysisRequest",
  "timestamp": "2025-10-15T12:34:56.789Z",
  "payload": {
    "prd_content": "As a user, I want...",
    "target_language": "python",
    "analysis_depth": "comprehensive"
  }
}
```

### Event Schemas

**9 Event Schemas** for request/response/status:

```python
# Request schemas
class CodegenAnalysisRequest(BaseModel):
    prd_content: str
    target_language: str
    analysis_depth: Literal["basic", "comprehensive"]

class CodegenValidationRequest(BaseModel):
    code_content: str
    language: str
    validation_rules: list[str]

class CodegenPatternRequest(BaseModel):
    node_type: str
    functionality_description: str
    language: str

class CodegenMixinRequest(BaseModel):
    node_name: str
    required_capabilities: list[str]
    language: str

# Response schemas
class CodegenAnalysisResponse(BaseModel):
    analysis_result: dict[str, Any]
    confidence_score: float
    recommendations: list[str]

class CodegenValidationResponse(BaseModel):
    is_valid: bool
    issues: list[dict]
    suggestions: list[str]

class CodegenPatternResponse(BaseModel):
    similar_patterns: list[dict]
    recommended_approach: str
    code_snippets: list[str]

class CodegenMixinResponse(BaseModel):
    recommended_mixins: list[str]
    implementation_notes: str
    dependencies: list[str]

# Status schema
class CodegenStatusEvent(BaseModel):
    session_id: UUID
    status: Literal["started", "in_progress", "completed", "failed"]
    progress_percent: float
    current_step: str
    metadata: dict[str, Any]
```

### DLQ Monitoring

**Dead Letter Queue** monitoring with threshold-based alerting:

```python
# src/omninode_bridge/events/dlq_monitor.py

class DLQMonitor:
    """
    Monitor Dead Letter Queue topics for failed events.

    Features:
    - Threshold-based alerting (>10 failures → alert)
    - Real-time monitoring of all DLQ topics
    - Automatic retry logic for transient failures
    - Event analysis and reporting
    """

    DLQ_TOPICS = [
        "omninode_codegen_dlq_analyze_v1",
        "omninode_codegen_dlq_validate_v1",
        "omninode_codegen_dlq_pattern_v1",
        "omninode_codegen_dlq_mixin_v1",
    ]

    ALERT_THRESHOLD = 10  # Alert if >10 failed events

    async def monitor_dlqs(self) -> dict[str, int]:
        """
        Monitor all DLQ topics and trigger alerts if thresholds exceeded.

        Returns:
            Dict mapping DLQ topic to failure count
        """
        dlq_counts = {}

        for dlq_topic in self.DLQ_TOPICS:
            # Get message count from Kafka
            count = await self._get_topic_message_count(dlq_topic)
            dlq_counts[dlq_topic] = count

            # Trigger alert if threshold exceeded
            if count > self.ALERT_THRESHOLD:
                await self._trigger_alert(dlq_topic, count)

        return dlq_counts

    async def _trigger_alert(self, dlq_topic: str, count: int) -> None:
        """Send alert for DLQ threshold breach."""
        logger.error("DLQ threshold exceeded",
                    dlq_topic=dlq_topic,
                    count=count,
                    threshold=self.ALERT_THRESHOLD)

        # Publish alert event
        await self.event_producer.publish_event(
            topic="omninode_system_alerts",
            event=DLQAlertEvent(
                dlq_topic=dlq_topic,
                failure_count=count,
                threshold=self.ALERT_THRESHOLD,
                timestamp=datetime.utcnow()
            ),
            correlation_id=uuid4()
        )
```

---

## Data Flow

### End-to-End Stamp Creation Flow

```
1. CLIENT REQUEST
   │
   ├─► POST /stamp
   │   {
   │     "content": "Hello World",
   │     "namespace": "my_app",
   │     "file_path": "/data/file.txt"
   │   }
   │
2. METADATA STAMPING SERVICE
   │
   ├─► Validate input
   ├─► Generate BLAKE3 hash (<2ms)
   ├─► Detect file type (multi-modal)
   ├─► Create stamp metadata (O.N.E. v0.1)
   │
3. NODE BRIDGE ORCHESTRATOR
   │
   ├─► FSM: PENDING → PROCESSING
   ├─► [Optional] Route to OnexTree for intelligence
   ├─► Coordinate stamping workflow (6 steps)
   ├─► Publish WORKFLOW_STARTED event → Kafka
   ├─► Publish STAMP_CREATED event → Kafka
   ├─► FSM: PROCESSING → COMPLETED
   │
4. KAFKA EVENT STREAM
   │
   ├─► Topic: omninode_codegen_status_session_v1
   │   Envelope: OnexEnvelopeV1
   │   Payload: { stamp_id, file_hash, metadata }
   │
5. NODE BRIDGE REDUCER
   │
   ├─► Stream stamp metadata (async iterator)
   ├─► Batch processing (100 items/batch)
   ├─► Group by namespace
   ├─► Compute aggregations:
   │   • total_stamps++
   │   • total_size_bytes += file_size
   │   • unique_file_types.add(content_type)
   │   • workflow_ids.add(workflow_id)
   ├─► Track FSM states per workflow
   │
6. POSTGRESQL PERSISTENCE
   │
   ├─► BEGIN TRANSACTION
   ├─► INSERT/UPDATE bridge_states
   │   {
   │     namespace: "my_app",
   │     total_stamps: 100,
   │     total_size_bytes: 10485760,
   │     unique_file_types: ["text/plain", "application/pdf"],
   │     current_fsm_state: "AGGREGATING"
   │   }
   ├─► INSERT workflow_executions
   ├─► INSERT event_logs
   ├─► COMMIT TRANSACTION
   │
7. RESPONSE TO CLIENT
   │
   └─► 200 OK
       {
         "stamp_id": "stamp_abc123...",
         "file_hash": "blake3_def456...",
         "stamped_content": "---BEGIN ONEX STAMP---\n...",
         "namespace": "my_app",
         "metadata_version": "0.1",
         "processing_time_ms": 45.2,
         "workflow_state": "COMPLETED"
       }
```

### Aggregation Data Flow

```
Input Stream (Orchestrator → Reducer):
│
├─► Async Iterator<ModelStampMetadataInput>
│   │
│   ├─► Item 1: { stamp_id, file_hash, namespace: "app1", file_size: 1024, ... }
│   ├─► Item 2: { stamp_id, file_hash, namespace: "app1", file_size: 2048, ... }
│   ├─► Item 3: { stamp_id, file_hash, namespace: "app2", file_size: 512, ... }
│   └─► ... (streaming)
│
Windowing & Batching:
│
├─► Window: 5000ms time-based
├─► Batch: 100 items per batch
│
Aggregation (Namespace Grouping):
│
├─► app1:
│   ├─► total_stamps: 2
│   ├─► total_size_bytes: 3072 (1024 + 2048)
│   ├─► file_types: ["text/plain"]
│   └─► workflow_ids: ["uuid1", "uuid2"]
│
├─► app2:
│   ├─► total_stamps: 1
│   ├─► total_size_bytes: 512
│   ├─► file_types: ["application/pdf"]
│   └─► workflow_ids: ["uuid3"]
│
PostgreSQL Persistence:
│
├─► UPDATE bridge_states
│   SET total_stamps = total_stamps + 2,
│       total_size_bytes = total_size_bytes + 3072,
│       unique_file_types = unique_file_types || ARRAY['text/plain']
│   WHERE namespace = 'app1'
│
└─► UPDATE bridge_states
    SET total_stamps = total_stamps + 1,
        total_size_bytes = total_size_bytes + 512,
        unique_file_types = unique_file_types || ARRAY['application/pdf']
    WHERE namespace = 'app2'
```

---

## Design Patterns

### 1. Circuit Breaker Pattern

**Purpose**: Prevent cascading failures when external services are unavailable.

**Implementation**:
```python
# src/omninode_bridge/patterns/circuit_breaker.py

class DatabaseCircuitBreaker:
    """
    Circuit breaker for database resilience.

    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, reject requests
    - HALF_OPEN: Testing recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def record_success(self) -> None:
        """Record successful operation, potentially close circuit."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful recovery")

    def record_failure(self) -> None:
        """Record failed operation, potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened",
                         failure_count=self.failure_count,
                         threshold=self.failure_threshold)

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == CircuitState.CLOSED:
            return False

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
                return False
            return True

        return False
```

### 2. FSM (Finite State Machine) Pattern

**Purpose**: Manage workflow state transitions with validation.

**Implementation**:
```python
# src/omninode_bridge/nodes/orchestrator/v1_0_0/models/enum_workflow_state.py

class EnumWorkflowState(str, Enum):
    """
    FSM states for orchestrator workflows.

    State Transitions:
    - PENDING → PROCESSING (workflow started)
    - PROCESSING → COMPLETED (success)
    - PROCESSING → FAILED (error)

    Terminal States: COMPLETED, FAILED
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    def can_transition_to(self, target: "EnumWorkflowState") -> bool:
        """
        Check if transition to target state is valid.

        Args:
            target: Desired target state

        Returns:
            True if transition is allowed, False otherwise
        """
        VALID_TRANSITIONS = {
            EnumWorkflowState.PENDING: [EnumWorkflowState.PROCESSING],
            EnumWorkflowState.PROCESSING: [
                EnumWorkflowState.COMPLETED,
                EnumWorkflowState.FAILED
            ],
            EnumWorkflowState.COMPLETED: [],  # Terminal
            EnumWorkflowState.FAILED: []       # Terminal
        }

        return target in VALID_TRANSITIONS.get(self, [])

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in [EnumWorkflowState.COMPLETED, EnumWorkflowState.FAILED]
```

### 3. Saga Pattern (Planned)

**Purpose**: Distributed transaction coordination with compensating actions.

**Status**: Deferred to Phase 3+ (enterprise features).

**Design**:
```python
# Future implementation: src/omninode_bridge/patterns/saga.py

class StampingWorkflowSaga:
    """
    Saga pattern for distributed stamping workflow.

    Steps:
    1. Validate input (compensate: none)
    2. Generate hash (compensate: delete hash)
    3. Create stamp (compensate: delete stamp)
    4. Publish event (compensate: publish rollback event)
    5. Persist state (compensate: rollback transaction)

    If any step fails, execute compensating actions in reverse.
    """
    # Implementation deferred
```

### 4. Dependency Injection Pattern

**Purpose**: Decouple components via container-based service resolution.

**Implementation**:
```python
# src/omnibase_core/models/container/model_onex_container.py

class ModelONEXContainer:
    """
    ONEX dependency injection container.

    Provides service resolution for all nodes.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._services: dict[str, Any] = {}

    def register_service(self, name: str, instance: Any) -> None:
        """Register service instance."""
        self._services[name] = instance

    def get_service(self, name: str) -> Any:
        """Resolve service by name."""
        if name not in self._services:
            raise OnexError(
                code=CoreErrorCode.RESOURCE_ERROR,
                message=f"Service '{name}' not registered in container"
            )
        return self._services[name]
```

**Usage**:
```python
# Initialize container with services
container = ModelONEXContainer(
    config={
        "metadata_stamping_service_url": "http://metadata-stamping:8053",
        "onextree_service_url": "http://onextree:8080"
    }
)

# Register services
container.register_service('postgresql_client', postgresql_manager)
container.register_service('kafka_producer', kafka_producer)
container.register_service('blake3_generator', hash_generator)
container.register_service('bridge_orchestrator', orchestrator)
container.register_service('bridge_reducer', reducer)

# Nodes resolve dependencies from container
orchestrator = NodeBridgeOrchestrator(container)
# orchestrator.container.get_service('kafka_producer') → kafka_producer instance
```

### 5. Repository Pattern

**Purpose**: Abstract database access behind clean interface.

**Implementation**:
```python
# src/omninode_bridge/persistence/repositories/bridge_state_repository.py

class BridgeStateRepository:
    """
    Repository for bridge_states table access.

    Provides CRUD operations with transaction support.
    """

    def __init__(self, connection_manager: PostgresConnectionManager):
        self.connection_manager = connection_manager

    async def create(self, bridge_state: ModelBridgeState) -> UUID:
        """Create new bridge state record."""
        async with self.connection_manager.transaction() as tx:
            result = await tx.fetchrow(
                """
                INSERT INTO bridge_states (
                    state_id, version, namespace, metadata_version,
                    total_stamps, total_size_bytes, unique_file_types,
                    current_fsm_state, created_at, last_updated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING state_id
                """,
                bridge_state.state_id,
                bridge_state.version,
                bridge_state.namespace,
                bridge_state.metadata_version,
                bridge_state.total_stamps,
                bridge_state.total_size_bytes,
                list(bridge_state.unique_file_types),
                bridge_state.current_fsm_state,
                bridge_state.created_at,
                bridge_state.last_updated
            )
            return result['state_id']

    async def get_by_namespace(self, namespace: str) -> ModelBridgeState | None:
        """Retrieve bridge state by namespace."""
        async with self.connection_manager.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM bridge_states
                WHERE namespace = $1
                """,
                namespace
            )

            if not row:
                return None

            return ModelBridgeState(
                state_id=row['state_id'],
                version=row['version'],
                namespace=row['namespace'],
                metadata_version=row['metadata_version'],
                total_stamps=row['total_stamps'],
                total_size_bytes=row['total_size_bytes'],
                unique_file_types=set(row['unique_file_types']),
                current_fsm_state=row['current_fsm_state'],
                created_at=row['created_at'],
                last_updated=row['last_updated']
            )

    async def update(self, bridge_state: ModelBridgeState) -> None:
        """Update existing bridge state with optimistic locking."""
        async with self.connection_manager.transaction() as tx:
            result = await tx.execute(
                """
                UPDATE bridge_states
                SET version = $1,
                    total_stamps = $2,
                    total_size_bytes = $3,
                    unique_file_types = $4,
                    current_fsm_state = $5,
                    last_updated = $6
                WHERE namespace = $7
                  AND version = $8
                """,
                bridge_state.version + 1,
                bridge_state.total_stamps,
                bridge_state.total_size_bytes,
                list(bridge_state.unique_file_types),
                bridge_state.current_fsm_state,
                datetime.utcnow(),
                bridge_state.namespace,
                bridge_state.version
            )

            if result == "UPDATE 0":
                raise OnexError(
                    code=CoreErrorCode.STATE_ERROR,
                    message="Optimistic locking failure: state version conflict",
                    details={"namespace": bridge_state.namespace}
                )
```

---

## Integration Points

### 1. MetadataStampingService → NodeBridgeOrchestrator

**Integration Type**: Direct method invocation

```python
# src/metadata_stamping/main.py

@app.post("/stamp")
async def create_stamp(request: ModelStampRequest) -> ModelStampResponse:
    # ... validation and hash generation ...

    # Invoke orchestrator
    orchestrator = container.get_service('bridge_orchestrator')

    contract = ModelContractOrchestrator(
        correlation_id=uuid4(),
        input_data={
            "content": request.content,
            "namespace": request.namespace,
            "file_path": request.file_path,
            "file_hash": file_hash,
            "content_type": content_type
        }
    )

    result = await orchestrator.execute_orchestration(contract)

    return ModelStampResponse(
        stamp_id=result.stamp_id,
        file_hash=result.file_hash,
        stamped_content=result.stamped_content,
        metadata=result.stamp_metadata
    )
```

### 2. NodeBridgeOrchestrator → OnexTree Intelligence

**Integration Type**: HTTP REST API with circuit breaker

```python
# src/omninode_bridge/clients/onextree_client.py

class OnexTreeClient:
    """
    HTTP client for OnexTree intelligence service.

    Features:
    - Circuit breaker for resilience
    - Timeout handling (5s)
    - Retry logic (3 attempts)
    - JSON request/response
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        self.client = httpx.AsyncClient(timeout=5.0)

    async def analyze_content(
        self,
        content: str,
        analysis_type: str
    ) -> dict[str, Any]:
        """
        Request content analysis from OnexTree.

        Args:
            content: Content to analyze
            analysis_type: Type of analysis ("security", "classification", etc.)

        Returns:
            Analysis result dictionary

        Raises:
            OnexError: If request fails or circuit is open
        """
        if self.circuit_breaker.is_open():
            raise OnexError(
                code=CoreErrorCode.RESOURCE_ERROR,
                message="OnexTree circuit breaker is OPEN"
            )

        try:
            response = await self.client.post(
                f"{self.base_url}/analyze",
                json={
                    "content": content,
                    "analysis_type": analysis_type
                }
            )

            response.raise_for_status()
            self.circuit_breaker.record_success()

            return response.json()

        except Exception as e:
            self.circuit_breaker.record_failure()
            raise OnexError(
                code=CoreErrorCode.OPERATION_FAILED,
                message="OnexTree analysis request failed",
                details={"error": str(e)}
            )
```

### 3. NodeBridgeOrchestrator → Kafka

**Integration Type**: Event publishing with OnexEnvelopeV1

```python
# Event publishing in orchestrator
async def _publish_event(
    self,
    event_type: EnumWorkflowEvent,
    data: dict[str, Any]
) -> None:
    """Publish workflow event to Kafka."""
    kafka_producer = self.container.get_service('kafka_producer')

    # Create event payload
    event = WorkflowEvent(
        event_type=event_type.value,
        workflow_id=data.get('workflow_id'),
        timestamp=datetime.utcnow(),
        data=data
    )

    # Publish with OnexEnvelopeV1 envelope
    await kafka_producer.publish_event(
        topic=f"omninode_codegen_status_session_v1",
        event=event,
        correlation_id=UUID(data['workflow_id'])
    )
```

### 4. NodeBridgeReducer → PostgreSQL

**Integration Type**: Direct database access via repository pattern

```python
# PostgreSQL persistence in reducer
async def _persist_state(
    self,
    aggregated_data: dict[str, dict[str, Any]],
    fsm_states: dict[str, str],
    contract: ModelContractReducer
) -> None:
    """Persist aggregated state to PostgreSQL."""
    repository = self.container.get_service('bridge_state_repository')

    for namespace, data in aggregated_data.items():
        # Get existing state or create new
        bridge_state = await repository.get_by_namespace(namespace)

        if not bridge_state:
            bridge_state = ModelBridgeState(
                state_id=uuid4(),
                version=1,
                namespace=namespace,
                metadata_version="0.1",
                total_stamps=0,
                total_size_bytes=0,
                unique_file_types=set(),
                current_fsm_state="IDLE",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )

        # Update with aggregated data
        bridge_state.total_stamps += data['total_stamps']
        bridge_state.total_size_bytes += data['total_size_bytes']
        bridge_state.unique_file_types.update(data['file_types'])
        bridge_state.last_updated = datetime.utcnow()

        # Persist to database (with optimistic locking)
        if bridge_state.version == 1:
            await repository.create(bridge_state)
        else:
            await repository.update(bridge_state)
```

### 5. External Services → MetadataStampingService

**Integration Type**: HTTP REST API

**Example Client Code**:
```python
import httpx

# External service calling MetadataStampingService
async def stamp_document(content: str, namespace: str) -> dict:
    """
    Create metadata stamp via MetadataStampingService API.

    Args:
        content: Document content to stamp
        namespace: Multi-tenant namespace

    Returns:
        Stamp metadata dictionary
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://metadata-stamping:8053/stamp",
            json={
                "content": content,
                "namespace": namespace,
                "file_path": "/data/document.txt"
            },
            timeout=10.0
        )

        response.raise_for_status()
        return response.json()

# Usage
stamp_result = await stamp_document("Important data", "omniclaude.docs")
print(f"Stamp ID: {stamp_result['stamp_id']}")
print(f"File Hash: {stamp_result['file_hash']}")
```

---

## Performance Architecture

### Performance Targets and Actual Results

| Component | Metric | Target | Actual | Status |
|-----------|--------|--------|--------|--------|
| **BLAKE3 Hash** | p99 latency | <2ms | <1ms (avg) | ✅ Exceeded |
| **API Response** | p95 latency | <10ms | <5ms | ✅ Exceeded |
| **Database CRUD** | p95 latency | <20ms | <10ms | ✅ Exceeded |
| **Event Log Queries** | p95 latency | <100ms | <50ms | ✅ Exceeded |
| **Orchestrator (standard)** | Avg latency | <50ms | ~45ms | ✅ Met |
| **Orchestrator (OnexTree)** | Avg latency | <150ms | ~140ms | ✅ Met |
| **Reducer Throughput** | Items/sec | >1000 | ~1100 | ✅ Exceeded |
| **Reducer Latency (1000)** | Latency | <100ms | ~85ms | ✅ Exceeded |
| **Kafka Producer** | Success rate | >95% | 100% | ✅ Exceeded |
| **Concurrent Requests** | Throughput | 1000+ | 1500+ | ✅ Exceeded |
| **Memory Usage** | Under load | <512MB | <300MB | ✅ Exceeded |

### Performance Optimization Techniques

#### 1. Connection Pooling

```python
# PostgreSQL connection pool configuration
CONNECTION_POOL_CONFIG = {
    "min_size": 10,          # Minimum connections
    "max_size": 50,          # Maximum connections
    "max_queries": 50000,    # Prepared statement cache
    "max_cached_statement_lifetime": 300,  # 5 minutes
    "command_timeout": 60,   # 60 second timeout
}
```

#### 2. Prepared Statement Caching

```python
# asyncpg automatically caches prepared statements
# up to max_queries limit (50,000)

# Frequently executed queries benefit from caching
CACHED_QUERIES = [
    "SELECT * FROM bridge_states WHERE namespace = $1",
    "INSERT INTO event_logs (event_type, correlation_id, payload) VALUES ($1, $2, $3)",
    "UPDATE bridge_states SET total_stamps = $1 WHERE namespace = $2"
]
```

#### 3. Batch Processing

```python
# Reducer batch processing configuration
BATCH_CONFIG = {
    "batch_size": 100,        # Process 100 items per batch
    "window_size_ms": 5000,   # 5 second time windows
    "max_wait_ms": 1000,      # Max wait for batch fill
}

# Benefits:
# - Reduced database round-trips (1 transaction for 100 items vs 100 transactions)
# - Better throughput (1100 items/sec vs ~200 items/sec without batching)
# - Lower latency (85ms for 1000 items vs ~500ms without batching)
```

#### 4. Hasher Pool

```python
# BLAKE3 hasher pool configuration
HASHER_POOL_CONFIG = {
    "pool_size": 100,         # Pre-allocated hashers
    "buffer_size_small": 8192,       # 8KB for <1KB files
    "buffer_size_medium": 65536,     # 64KB for 1-100KB files
    "buffer_size_large": 1048576,    # 1MB for >100KB files
}

# Benefits:
# - Zero-allocation hot path
# - Adaptive buffer sizing based on file size
# - <1ms average hash generation (target: <2ms)
```

#### 5. Async Iterator Streaming

```python
# Reducer streaming configuration
async def _stream_metadata(
    self,
    contract: ModelContractReducer,
    batch_size: int = 100
) -> AsyncIterator[list[ModelStampMetadataInput]]:
    """
    Stream metadata with windowing and batching.

    Benefits:
    - Constant memory usage (batch size * item size)
    - No buffering of entire dataset
    - Parallel processing of batches
    """
    batch = []

    async for item in contract.input_stream:
        batch.append(item)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
```

### Performance Monitoring

```python
# Prometheus metrics exposed at /metrics

# Hash generation latency
hash_generation_duration_seconds = Histogram(
    'hash_generation_duration_seconds',
    'Hash generation latency',
    buckets=[0.001, 0.002, 0.005, 0.010, 0.050, 0.100]
)

# API request latency
api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    buckets=[0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0]
)

# Database query latency
database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query latency',
    buckets=[0.001, 0.005, 0.010, 0.025, 0.050, 0.100]
)

# Active database connections
active_connections = Gauge(
    'active_connections',
    'Current active database connections'
)

# Pool exhaustion warnings
pool_exhaustion_warnings_total = Counter(
    'pool_exhaustion_warnings_total',
    'Total pool exhaustion warnings'
)
```

---

## Security Architecture

### Authentication and Authorization

**Current Status**: Basic authentication middleware in place, enterprise features deferred to Phase 3+.

```python
# src/omninode_bridge/middleware/auth_middleware.py

class AuthenticationMiddleware:
    """
    Basic authentication middleware.

    MVP Implementation:
    - API key validation
    - Request signing verification

    Phase 3+ (Deferred):
    - JWT token validation
    - OAuth2 integration
    - Role-based access control (RBAC)
    - Multi-tenant isolation enforcement
    """

    async def __call__(self, request: Request, call_next):
        # Extract API key from header
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing API key"}
            )

        # Validate API key (basic validation for MVP)
        if not self._validate_api_key(api_key):
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )

        # Continue processing
        response = await call_next(request)
        return response
```

### Input Validation

```python
# Pydantic v2 strict validation for all inputs

class ModelStampRequest(BaseModel):
    """
    Stamp request with strict validation.

    Security measures:
    - Content size limits (10MB max)
    - Namespace format validation
    - Path traversal prevention
    - SQL injection prevention (parameterized queries)
    """
    content: str = Field(..., max_length=10485760)  # 10MB max
    namespace: str = Field(..., pattern=r"^[a-zA-Z0-9_\-\.]+$")  # Alphanumeric + safe chars
    file_path: str | None = Field(None, pattern=r"^[^\.]{2}.*$")  # Prevent path traversal

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str | None) -> str | None:
        """Prevent path traversal attacks."""
        if v and ('..' in v or v.startswith('/')):
            raise ValueError("Invalid file path: potential path traversal")
        return v
```

### SQL Injection Prevention

```python
# Always use parameterized queries with asyncpg

# ❌ NEVER do this:
# query = f"SELECT * FROM bridge_states WHERE namespace = '{namespace}'"

# ✅ ALWAYS do this:
query = "SELECT * FROM bridge_states WHERE namespace = $1"
result = await conn.fetchrow(query, namespace)
```

### Secrets Management

```python
# src/omninode_bridge/config/secrets.py

class SecretsManager:
    """
    Secrets management for sensitive configuration.

    MVP: Environment variables
    Phase 3+: HashiCorp Vault, AWS Secrets Manager
    """

    @staticmethod
    def get_database_password() -> str:
        """Retrieve database password from environment."""
        password = os.getenv('POSTGRES_PASSWORD')

        if not password:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message="Database password not configured"
            )

        return password

    @staticmethod
    def get_kafka_credentials() -> tuple[str, str]:
        """Retrieve Kafka credentials."""
        username = os.getenv('KAFKA_USERNAME', '')
        password = os.getenv('KAFKA_PASSWORD', '')
        return (username, password)
```

### Network Security

**MVP Implementation**:
- Docker network isolation
- Service-to-service communication within Docker network
- No public exposure of internal services

**Phase 3+ (Deferred)**:
- TLS/SSL for all connections
- mTLS for service-to-service communication
- API Gateway with rate limiting
- DDoS protection
- WAF (Web Application Firewall)

---

## Scalability and Resilience

### Horizontal Scaling Architecture

**Current Status**: Single-instance MVP, designed for future horizontal scaling.

**Scaling Strategy** (Phase 3+):

```
┌──────────────────────────────────────────────────────────────┐
│                     Load Balancer (Nginx)                    │
└────────────┬─────────────────────────────────┬───────────────┘
             │                                 │
             ▼                                 ▼
┌────────────────────────┐         ┌────────────────────────┐
│ MetadataStamping       │         │ MetadataStamping       │
│ Service Instance 1     │         │ Service Instance 2     │
└────────────┬───────────┘         └────────────┬───────────┘
             │                                 │
             └─────────────┬───────────────────┘
                           │
                           ▼
             ┌────────────────────────────┐
             │ NodeBridgeOrchestrator     │
             │ (Stateless, horizontally   │
             │  scalable)                 │
             └────────────┬───────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │ NodeBridgeReducer          │
             │ (Streaming aggregation,    │
             │  partitioned by namespace) │
             └────────────┬───────────────┘
                          │
                          ▼
             ┌────────────────────────────┐
             │ PostgreSQL (Primary/       │
             │ Replica with pgpool)       │
             └────────────────────────────┘
```

### Resilience Patterns

#### 1. Circuit Breaker

**Implemented**: Database circuit breaker
**Status**: Production-ready

```python
# Configuration
DB_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5       # Open after 5 failures
DB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60       # 60s before retry
DB_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3    # Test with 3 calls
```

#### 2. Graceful Degradation

**Implemented**: OnexTree optional intelligence
**Status**: Production-ready

```python
# Orchestrator gracefully degrades if OnexTree unavailable
try:
    intelligence_data = await onextree_client.analyze_content(content)
except OnexError:
    # Continue without intelligence data
    intelligence_data = None
    logger.warning("OnexTree unavailable, continuing without intelligence")
```

#### 3. Retry Logic

**Implemented**: Kafka producer retries
**Status**: Production-ready

```python
# Kafka producer retry configuration
KAFKA_PRODUCER_CONFIG = {
    "max_retries": 3,
    "retry_backoff_ms": 100,
    "request_timeout_ms": 30000,
}
```

#### 4. Health Checks

**Implemented**: Component health monitoring
**Status**: Production-ready

```python
@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Comprehensive health check for all components.

    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "components": {
                "database": "healthy" | "unhealthy",
                "kafka": "healthy" | "unhealthy",
                "onextree": "healthy" | "unhealthy" | "unavailable"
            },
            "uptime_seconds": 3600
        }
    """
    health_status = {}

    # Check database
    try:
        async with connection_manager.acquire() as conn:
            await conn.fetchval("SELECT 1")
        health_status['database'] = 'healthy'
    except Exception:
        health_status['database'] = 'unhealthy'

    # Check Kafka
    try:
        await kafka_producer.ping()
        health_status['kafka'] = 'healthy'
    except Exception:
        health_status['kafka'] = 'unhealthy'

    # Check OnexTree (optional)
    try:
        await onextree_client.health_check()
        health_status['onextree'] = 'healthy'
    except Exception:
        health_status['onextree'] = 'unavailable'

    # Determine overall status
    if health_status['database'] == 'unhealthy' or health_status['kafka'] == 'unhealthy':
        overall_status = 'unhealthy'
    elif health_status['onextree'] == 'unavailable':
        overall_status = 'degraded'
    else:
        overall_status = 'healthy'

    return {
        "status": overall_status,
        "components": health_status,
        "uptime_seconds": time.time() - app.state.start_time
    }
```

#### 5. DLQ (Dead Letter Queue)

**Implemented**: Kafka DLQ for failed events
**Status**: Production-ready

```python
# Failed events automatically routed to DLQ topics
DLQ_TOPICS = {
    "omninode_codegen_request_analyze_v1": "omninode_codegen_dlq_analyze_v1",
    "omninode_codegen_request_validate_v1": "omninode_codegen_dlq_validate_v1",
    "omninode_codegen_request_pattern_v1": "omninode_codegen_dlq_pattern_v1",
    "omninode_codegen_request_mixin_v1": "omninode_codegen_dlq_mixin_v1",
}

# DLQ monitoring triggers alerts when threshold exceeded (>10 failures)
```

---

## Conclusion

OmniNode Bridge provides a production-ready MVP foundation for the omninode ecosystem with:

1. **ONEX v2.0 Compliance**: Suffix-based naming, contract-driven architecture, subcontract composition
2. **High Performance**: All performance targets met or exceeded (92.8% test coverage)
3. **Event-Driven**: 13 Kafka topics with OnexEnvelopeV1 standardization
4. **Database Persistence**: 7 PostgreSQL tables with 50+ indexes
5. **Resilience**: Circuit breakers, retries, graceful degradation
6. **Observability**: Prometheus metrics, structured logging, health checks

**Future Evolution**: Repository split into specialized repos (omninode-events, omninode-bridge-nodes, etc.) post-MVP validation.

---

**Document Version**: 2.0
**Maintained By**: omninode_bridge team
**Last Review**: October 15, 2025
**Next Review**: November 15, 2025

**Related Documentation**:
- [Getting Started](../GETTING_STARTED.md) - Quick start guide
- [Setup Guide](../SETUP.md) - Development environment
- [Bridge Nodes Guide](../guides/BRIDGE_NODES_GUIDE.md) - Bridge node implementation
- [API Reference](../api/API_REFERENCE.md) - Complete API documentation
- [Event System Guide](../events/EVENT_SYSTEM_GUIDE.md) - Event infrastructure
- [Database Guide](../database/DATABASE_GUIDE.md) - Database schema and migrations
- [Testing Guide](../testing/TESTING_GUIDE.md) - Test organization and execution
- [Operations Guide](../operations/OPERATIONS_GUIDE.md) - Deployment and monitoring
