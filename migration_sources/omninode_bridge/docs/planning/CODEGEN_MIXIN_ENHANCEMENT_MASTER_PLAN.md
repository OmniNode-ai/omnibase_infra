# Code Generator Mixin Enhancement Master Plan

**Version**: 1.0
**Created**: 2025-11-04
**Status**: 🎯 Ready for Implementation
**Estimated Timeline**: 6 weeks

---

## Executive Summary

### The Problem

The omninode_bridge code generator currently uses LLM-based generation without leveraging **40+ production-ready mixins** from omnibase_core. The database adapter node manually reimplemented features (circuit breakers, health checks, metrics, DLQ handling) that already exist in omnibase_core mixins.

### The Gap

1. **Generator doesn't know about mixins** - No contract schema for mixin declarations
2. **No mixin injection** - Templates don't import or apply mixins
3. **Manual reimplementation** - Developers duplicate omnibase_core features
4. **Quality gap** - Generated nodes lack production features hand-coded nodes have

### The Solution

Enhance the code generator to:
- Read mixin declarations from YAML contracts
- Inject omnibase_core mixins into generated nodes
- Leverage NodeEffect built-in features (circuit breakers, retry policies, transactions)
- Generate production-quality nodes comparable to hand-coded implementations

### Success Metrics

- ✅ Generated nodes use omnibase_core mixins instead of manual implementations
- ✅ LOC reduction: 30-50% fewer lines for equivalent functionality
- ✅ Feature parity: Generated nodes = hand-coded quality
- ✅ Zero feature regressions in existing nodes
- ✅ 100% test pass rate after regeneration

---

## Phase 1: Discovery & Cataloging (Week 1)

### 1.1 Mixin Catalog Creation

**Goal**: Document all 40+ omnibase_core mixins with capabilities and usage patterns.

**Discovery Results** (Already Completed):

```
OMNIBASE_CORE MIXINS (33 Total):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEALTH & MONITORING:
  • MixinHealthCheck - Health check endpoints and monitoring
  • MixinMetrics - Performance metrics collection
  • MixinLogData - Structured logging support
  • MixinRequestResponseIntrospection - Request/response tracking

EVENT-DRIVEN PATTERNS:
  • MixinEventDrivenNode - Event-driven node base functionality
  • MixinEventBus - Event bus integration
  • MixinEventHandler - Event handler registration
  • MixinEventListener - Event consumption patterns
  • MixinIntrospectionPublisher - Introspection event publishing

SERVICE INTEGRATION:
  • MixinServiceRegistry - Service discovery and registration
  • MixinDiscoveryResponder - Discovery protocol responder
  • MixinNodeService - Node-as-service capabilities

EXECUTION PATTERNS:
  • MixinNodeExecutor - Node execution orchestration
  • MixinNodeLifecycle - Lifecycle management (init, shutdown)
  • MixinNodeSetup - Setup and initialization helpers
  • MixinHybridExecution - Hybrid sync/async execution
  • MixinToolExecution - Tool execution framework
  • MixinDagSupport - DAG workflow support

DATA HANDLING:
  • MixinHashComputation - Hash computation utilities
  • MixinCaching - Caching capabilities
  • MixinLazyEvaluation - Lazy loading patterns
  • MixinCompletionData - Completion tracking

SERIALIZATION:
  • MixinCanonicalYAMLSerializer - Canonical YAML serialization
  • MixinYAMLSerialization - Standard YAML operations
  • SerializableMixin - Generic serialization support
  • MixinSensitiveFieldRedaction - Sensitive data redaction

CONTRACT & METADATA:
  • MixinContractMetadata - Contract metadata handling
  • MixinContractStateReducer - Contract state reduction
  • MixinIntrospectFromContract - Contract-driven introspection
  • MixinNodeIdFromContract - Node ID from contract
  • MixinNodeIntrospection - Node introspection support

CLI & DEBUGGING:
  • MixinCLIHandler - CLI command handling
  • MixinDebugDiscoveryLogging - Debug logging for discovery
  • MixinFailFast - Fail-fast validation patterns

NODEEFFECT BUILT-IN FEATURES (No Mixin Required):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ Circuit Breakers - ModelCircuitBreaker with failure threshold
  ✅ Retry Policies - Exponential backoff with configurable attempts
  ✅ Transaction Support - ModelEffectTransaction with rollback
  ✅ Timeout Management - Per-operation timeout configuration
  ✅ Concurrent Execution - Semaphore-based concurrency control
  ✅ Performance Metrics - Built-in metrics tracking
  ✅ Event Emission - State change event publishing
  ✅ File Operations - Atomic file I/O with rollback

**Usage**: These are ALREADY available in NodeEffect base class.
          Generator should leverage them instead of reimplementing.
```

**Deliverable**: `docs/reference/OMNIBASE_CORE_MIXIN_CATALOG.md`

**Action Items**:
- [x] ~~Clone omnibase_core v0.1.0~~ (Completed in discovery)
- [x] ~~List all mixin files~~ (33 mixins found)
- [ ] For each mixin, document:
  - Purpose and capabilities
  - Required dependencies
  - Configuration options
  - Usage examples
  - Common patterns
  - NodeEffect overlap (avoid duplication)

**Code Example - Mixin Documentation Format**:
```yaml
# Example from catalog
mixin_name: MixinHealthCheck
module: omnibase_core.mixins.mixin_health_check
class: MixinHealthCheck

capabilities:
  - Health status reporting (HEALTHY, DEGRADED, UNHEALTHY)
  - Component-level health checks
  - Timeout-based health validation
  - Health check aggregation

configuration:
  check_interval_ms: 30000  # Health check interval
  timeout_seconds: 5.0      # Health check timeout
  components:               # Components to monitor
    - name: "database"
      critical: true
      timeout_seconds: 5.0

usage_example: |
  class NodeMyEffect(NodeEffect, MixinHealthCheck):
      async def initialize(self):
          await super().initialize()
          self.register_health_check("database", self._check_db_health)

      async def _check_db_health(self) -> bool:
          return await self.postgres_client.check_health()

when_to_use:
  - Node interacts with external services (databases, APIs)
  - Need component-level health monitoring
  - Integration with health check endpoints

overlap_with_nodeeffect: NONE
  # MixinHealthCheck is complementary to NodeEffect
  # NodeEffect handles effect execution, mixin adds health checks
```

### 1.2 Current Generator Mapping

**Goal**: Understand current generator architecture and entry points.

**Discovery Results** (Already Completed):

```
GENERATOR ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Entry Points:
  • CodeGenerationService (src/omninode_bridge/codegen/service.py)
    - Unified facade for all generation strategies
    - Strategy selection and orchestration

  • CodeGenerationPipeline (src/omninode_bridge/codegen/pipeline.py)
    - End-to-end generation orchestration
    - Quality gates and validation

  • BusinessLogicGenerator (src/omninode_bridge/codegen/business_logic/generator.py)
    - LLM-based method implementation generation
    - Uses NodeLLMEffect for generation

Generation Strategies:
  • Jinja2Strategy - Template-based generation (TEST files only)
  • TemplateLoadingStrategy - LLM-powered generation
  • HybridStrategy - Combines both approaches
  • AutoStrategy - Intelligent strategy selection

Template System:
  • Jinja2 templates - Located in src/omninode_bridge/codegen/templates/
  • Currently ONLY for test files (test_unit.py.j2, test_integration.py.j2)
  • NO node generation templates exist yet
  • Templates use .j2 extension

Contract Processing:
  • ContractIntrospector (src/omninode_bridge/codegen/contract_introspector.py)
    - Parses contract YAML files
    - Extracts node metadata, models, operations

  • ModelPRDRequirements (src/omninode_bridge/codegen/prd_analyzer.py)
    - Analyzes PRD requirements for generation
    - Drives generation strategy selection

Current Contract Schema:
  ✓ node_type, name, version, description
  ✓ input_model, output_model, definitions
  ✓ io_operations, capabilities, dependencies
  ✓ configuration, performance_requirements
  ✓ error_handling (retry_policy, circuit_breaker, timeout_policy)
  ✓ subcontracts (effect, event_type, state_management)

  ✗ NO MIXIN DECLARATIONS
  ✗ NO ADVANCED FEATURE CONFIGURATION (DLQ, security, observability)
```

**Deliverable**: Architecture diagram + component interaction map

**Action Items**:
- [x] ~~Map service.py → pipeline.py → strategies~~ (Completed)
- [x] ~~Identify template engine location~~ (Jinja2 for tests only)
- [ ] Document contract processing flow
- [ ] Identify injection points for mixin handling
- [ ] Map LLM generation vs template generation paths

**Diagram** (To be created in docs/):
```
┌─────────────────────────────────────────────────────────────┐
│                 CodeGenerationService                        │
│                    (Unified Facade)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              Strategy Registry & Selection                   │
│  [Jinja2 | TemplateLoading | Hybrid | Auto]                │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ↓                          ↓
┌──────────────────┐      ┌──────────────────┐
│ TemplateEngine   │      │ BusinessLogic    │
│  (Jinja2)        │      │  Generator       │
│  - Tests only    │      │  - LLM-based     │
│  - .j2 files     │      │  - NodeLLMEffect │
└──────────────────┘      └──────────────────┘
        │                          │
        └────────────┬─────────────┘
                     ↓
              Generated Artifacts
        (node.py, contract.yaml, tests)
```

### 1.3 Existing Node Analysis

**Goal**: Catalog all generated nodes for regression testing.

**Existing Nodes** (Already Discovered):

```
HAND-CODED NODES (Reference Implementations):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. database_adapter_effect/v1_0_0/
   • Custom: HealthCheckMixin, CircuitBreaker, StructuredLogger
   • Custom: SecurityValidator, GenericCRUDHandlers
   • Manual: DLQ handling, metrics tracking
   • Status: Production-ready, 500+ LOC
   • Regeneration Priority: HIGH (proves mixin value)

2. llm_effect/v1_0_0/
   • Uses: omnibase_core.ModelCircuitBreaker ✓
   • Manual: Retry logic, HTTP client management
   • Manual: Cost tracking, token metrics
   • Status: Production-ready, Phase 1 complete
   • Regeneration Priority: MEDIUM

3. orchestrator/v1_0_0/
   • FSM state management
   • Workflow coordination
   • Event publishing
   • Regeneration Priority: MEDIUM

4. reducer/v1_0_0/
   • Streaming aggregation
   • State persistence
   • Regeneration Priority: MEDIUM

5. registry/v1_0_0/
   • Service discovery
   • Node registration
   • Regeneration Priority: LOW

6. deployment_receiver_effect/v1_0_0/
   • Docker API integration
   • Container deployment
   • Regeneration Priority: LOW

7. deployment_sender_effect/v1_0_0/
   • Deployment coordination
   • Regeneration Priority: LOW

8. distributed_lock_effect/v1_0_0/
   • Distributed locking
   • Regeneration Priority: LOW

9. store_effect/v1_0_0/
   • Generic storage operations
   • Regeneration Priority: LOW

10. test_generator_effect/v1_0_0/
    • Test generation
    • Regeneration Priority: LOW

11. codegen_orchestrator/v1_0_0/
    • Code generation orchestration
    • Regeneration Priority: LOW

12. codegen_metrics_reducer/v1_0_0/
    • Metrics aggregation
    • Regeneration Priority: LOW

GENERATED DEMO NODES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• vault_secrets_effect/ - Secret management
• demo_generated_mysql_adapter/ - MySQL adapter demo
```

**Deliverable**: Node inventory with feature matrix

**Action Items**:
- [x] ~~List all nodes in src/omninode_bridge/nodes/~~ (12 found)
- [ ] For each node, document:
  - Current features (circuit breaker, retry, metrics, etc.)
  - Manual implementations vs omnibase_core features
  - LOC count
  - Test coverage
  - Regeneration priority (HIGH/MEDIUM/LOW)
- [ ] Create baseline metrics for comparison

**Feature Matrix Template**:
```markdown
| Node | Circuit Breaker | Retry | Health | Metrics | DLQ | Tests | LOC | Priority |
|------|----------------|-------|--------|---------|-----|-------|-----|----------|
| database_adapter | Custom | ✓ | Custom | Custom | Custom | ✓ | 500+ | HIGH |
| llm_effect | ✓ (omnibase) | Manual | - | Manual | - | ✓ | 300+ | MEDIUM |
| orchestrator | - | - | - | - | - | ✓ | 400+ | MEDIUM |
...
```

---

## Phase 2: Contract Schema Design (Week 1-2)

### 2.1 Extended Contract Schema

**Goal**: Design YAML schema extension for mixin declarations and advanced features.

**Proposed Schema Extension**:

```yaml
# NEW: Mixin declarations section
mixins:
  # Health monitoring
  - name: MixinHealthCheck
    enabled: true
    config:
      check_interval_ms: 30000
      timeout_seconds: 5.0
      components:
        - name: "database"
          critical: true
          timeout_seconds: 5.0
        - name: "kafka_consumer"
          critical: false
          timeout_seconds: 3.0

  # Metrics collection
  - name: MixinMetrics
    enabled: true
    config:
      collect_latency: true
      collect_throughput: true
      collect_error_rates: true
      percentiles: [50, 95, 99]
      histogram_buckets: [10, 50, 100, 500, 1000]

  # Event-driven capabilities
  - name: MixinEventDrivenNode
    enabled: true
    config:
      kafka_bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
      consumer_group: "${SERVICE_NAME}_consumers"
      topics:
        - "workflow-started"
        - "workflow-completed"

  # Service discovery
  - name: MixinServiceRegistry
    enabled: false  # Optional
    config:
      consul_host: "${CONSUL_HOST}"
      consul_port: 8500
      service_name: "${SERVICE_NAME}"
      health_check_interval_seconds: 10

  # Structured logging
  - name: MixinLogData
    enabled: true
    config:
      log_level: "INFO"
      structured: true
      include_correlation_id: true
      redact_sensitive_fields: true

# NEW: Advanced features configuration
advanced_features:
  # Circuit breaker (NodeEffect built-in, just configure)
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout_ms: 60000
    half_open_max_calls: 3
    services:
      - name: "postgres"
        failure_threshold: 3
        recovery_timeout_ms: 30000
      - name: "kafka"
        failure_threshold: 10
        recovery_timeout_ms: 120000

  # Retry policy (NodeEffect built-in)
  retry_policy:
    enabled: true
    max_attempts: 3
    initial_delay_ms: 1000
    max_delay_ms: 30000
    backoff_multiplier: 2.0
    retryable_exceptions:
      - "TimeoutError"
      - "ConnectionError"
      - "TemporaryFailure"
    retryable_status_codes: [429, 500, 502, 503, 504]

  # Dead Letter Queue
  dead_letter_queue:
    enabled: true
    max_retries: 3
    topic_suffix: ".dlq"
    retry_delay_ms: 5000
    alert_threshold: 100  # Alert if >100 messages in DLQ
    monitoring:
      enabled: true
      check_interval_seconds: 60

  # Transaction support (NodeEffect built-in)
  transactions:
    enabled: true
    isolation_level: "READ_COMMITTED"
    timeout_seconds: 30
    rollback_on_error: true
    savepoints: true

  # Security validation
  security_validation:
    enabled: true
    sanitize_inputs: true
    sanitize_logs: true
    validate_sql: true
    max_input_length: 10000
    forbidden_patterns:
      - "(?i)(DROP|DELETE|TRUNCATE)\\s+TABLE"
      - "(?i)EXEC(UTE)?\\s+"
    redact_fields:
      - "password"
      - "api_key"
      - "secret"
      - "token"

  # Observability
  observability:
    tracing:
      enabled: true
      exporter: "otlp"
      endpoint: "${OTEL_EXPORTER_OTLP_ENDPOINT}"
      sample_rate: 1.0
    metrics:
      enabled: true
      prometheus_port: 9090
      export_interval_seconds: 15
    logging:
      structured: true
      json_format: true
      correlation_tracking: true

# EXISTING: Kept for backward compatibility
error_handling:
  retry_policy:  # Duplicated above, prefer advanced_features
    max_attempts: 3
    backoff_multiplier: 2.0
  circuit_breaker:  # Duplicated above, prefer advanced_features
    enabled: true
    failure_threshold: 5

# EXISTING: Subcontracts (kept as-is)
subcontracts:
  effect:
    operations: ["initialize", "shutdown", "health_check"]
  event_type:
    consumed_events: ["WORKFLOW_STARTED"]
```

**Schema Validation Rules**:

1. **Mixin Validation**:
   - Validate mixin name exists in omnibase_core
   - Check required dependencies are satisfied
   - Validate config against mixin's expected schema

2. **Feature Overlap Detection**:
   - Warn if both `error_handling.circuit_breaker` and `advanced_features.circuit_breaker` defined
   - Prefer `advanced_features` over legacy `error_handling`
   - Detect NodeEffect built-in features vs mixin overlap

3. **Configuration Inheritance**:
   - Environment variables resolved at runtime
   - Default values from omnibase_core
   - Contract can override defaults

**Deliverable**: JSON Schema for validation + migration guide

**Action Items**:
- [ ] Create JSON Schema for `mixins` section
- [ ] Create JSON Schema for `advanced_features` section
- [ ] Document migration from old schema to new
- [ ] Add validation examples
- [ ] Create schema versioning strategy

### 2.2 Backward Compatibility Strategy

**Goal**: Ensure existing contracts continue to work without modification.

**Compatibility Rules**:

1. **Default Behavior** (no mixins declared):
   ```python
   # Generator produces minimal node (as today)
   class NodeMyEffect(NodeEffect):
       # Only NodeEffect built-ins (circuit breaker, retry)
       pass
   ```

2. **Explicit Mixin Declaration**:
   ```python
   # Generator adds mixins to inheritance chain
   class NodeMyEffect(NodeEffect, MixinHealthCheck, MixinMetrics):
       # Mixins + NodeEffect built-ins
       pass
   ```

3. **Legacy Contract Support**:
   - Old contracts without `mixins` section → work as before
   - Old contracts with `error_handling` → map to `advanced_features`
   - Migration warnings logged but not errors

4. **Opt-In Enhancement**:
   - Existing nodes regenerate with `--enhance-with-mixins` flag
   - New nodes get mixins by default
   - `--minimal` flag for basic generation

**Migration Script**:
```python
# scripts/migrate_contracts_to_mixins.py
def migrate_contract(old_contract: dict) -> dict:
    """Migrate old contract schema to new mixin-enhanced schema."""
    new_contract = old_contract.copy()

    # Map error_handling to advanced_features
    if "error_handling" in old_contract:
        new_contract["advanced_features"] = {
            "circuit_breaker": old_contract["error_handling"].get("circuit_breaker"),
            "retry_policy": old_contract["error_handling"].get("retry_policy"),
        }

    # Add recommended mixins based on node type
    if old_contract["node_type"] == "effect":
        new_contract["mixins"] = [
            {"name": "MixinHealthCheck", "enabled": True},
            {"name": "MixinMetrics", "enabled": True},
        ]

    return new_contract
```

**Deliverable**: Migration script + backward compatibility tests

---

## Phase 3: Generator Architecture Enhancement (Week 2-3)

### 3.1 Contract Reader Enhancement

**Goal**: Parse mixin declarations and advanced features from contracts.

**Current State**:
```python
# src/omninode_bridge/codegen/contract_introspector.py
class ContractIntrospector:
    def parse_contract(self, contract_path: Path) -> dict:
        # Parses YAML but doesn't handle mixins
        pass
```

**Enhanced Implementation**:

```python
# src/omninode_bridge/codegen/contract_introspector.py
from dataclasses import dataclass
from typing import Optional
from omnibase_core.mixins import __all__ as AVAILABLE_MIXINS

@dataclass
class ModelMixinDeclaration:
    """Parsed mixin declaration from contract."""
    name: str
    enabled: bool = True
    config: dict = field(default_factory=dict)
    import_path: str = ""
    validation_errors: list[str] = field(default_factory=list)

@dataclass
class ModelAdvancedFeatures:
    """Parsed advanced features from contract."""
    circuit_breaker: Optional[dict] = None
    retry_policy: Optional[dict] = None
    dead_letter_queue: Optional[dict] = None
    transactions: Optional[dict] = None
    security_validation: Optional[dict] = None
    observability: Optional[dict] = None

@dataclass
class ModelEnhancedContract:
    """Enhanced contract with mixin and feature data."""
    # Original contract fields
    node_name: str
    node_type: str
    version: dict
    description: str

    # NEW: Mixin declarations
    mixins: list[ModelMixinDeclaration] = field(default_factory=list)

    # NEW: Advanced features
    advanced_features: Optional[ModelAdvancedFeatures] = None

    # Validation results
    has_errors: bool = False
    validation_errors: list[str] = field(default_factory=list)

class ContractIntrospector:
    """
    Enhanced contract parser with mixin and advanced feature support.
    """

    def __init__(self):
        """Initialize with omnibase_core mixin registry."""
        self.available_mixins = self._load_mixin_registry()

    def _load_mixin_registry(self) -> dict[str, dict]:
        """
        Load mixin metadata from omnibase_core.

        Returns:
            Dict mapping mixin name to metadata:
            {
                "MixinHealthCheck": {
                    "module": "omnibase_core.mixins.mixin_health_check",
                    "class": "MixinHealthCheck",
                    "dependencies": [],
                    "config_schema": {...},
                },
                ...
            }
        """
        registry = {}
        for mixin_name in AVAILABLE_MIXINS:
            # Load from omnibase_core or predefined catalog
            registry[mixin_name] = {
                "module": f"omnibase_core.mixins.{mixin_name.lower()}",
                "class": mixin_name,
                "dependencies": [],  # TODO: Extract from mixin
            }
        return registry

    def parse_contract(self, contract_path: Path) -> ModelEnhancedContract:
        """
        Parse contract YAML with mixin and advanced feature support.

        Args:
            contract_path: Path to contract YAML file

        Returns:
            ModelEnhancedContract with parsed data

        Raises:
            ContractValidationError: If contract is invalid
        """
        with open(contract_path) as f:
            raw_contract = yaml.safe_load(f)

        # Parse basic contract fields (existing logic)
        node_name = raw_contract["name"]
        node_type = raw_contract["node_type"]

        # NEW: Parse mixin declarations
        mixins = []
        if "mixins" in raw_contract:
            for mixin_decl in raw_contract["mixins"]:
                mixins.append(self._parse_mixin_declaration(mixin_decl))

        # NEW: Parse advanced features
        advanced_features = None
        if "advanced_features" in raw_contract:
            advanced_features = self._parse_advanced_features(
                raw_contract["advanced_features"]
            )

        # Create enhanced contract
        contract = ModelEnhancedContract(
            node_name=node_name,
            node_type=node_type,
            version=raw_contract["version"],
            description=raw_contract.get("description", ""),
            mixins=mixins,
            advanced_features=advanced_features,
        )

        # Validate contract
        self._validate_contract(contract)

        return contract

    def _parse_mixin_declaration(
        self, mixin_decl: dict
    ) -> ModelMixinDeclaration:
        """
        Parse and validate mixin declaration.

        Args:
            mixin_decl: Raw mixin declaration from contract

        Returns:
            Validated ModelMixinDeclaration
        """
        name = mixin_decl["name"]
        enabled = mixin_decl.get("enabled", True)
        config = mixin_decl.get("config", {})

        # Validate mixin exists
        validation_errors = []
        if name not in self.available_mixins:
            validation_errors.append(
                f"Mixin '{name}' not found in omnibase_core. "
                f"Available: {list(self.available_mixins.keys())}"
            )
            import_path = ""
        else:
            mixin_meta = self.available_mixins[name]
            import_path = f"{mixin_meta['module']}.{mixin_meta['class']}"

            # TODO: Validate config against mixin's schema

        return ModelMixinDeclaration(
            name=name,
            enabled=enabled,
            config=config,
            import_path=import_path,
            validation_errors=validation_errors,
        )

    def _parse_advanced_features(self, features: dict) -> ModelAdvancedFeatures:
        """Parse advanced features section."""
        return ModelAdvancedFeatures(
            circuit_breaker=features.get("circuit_breaker"),
            retry_policy=features.get("retry_policy"),
            dead_letter_queue=features.get("dead_letter_queue"),
            transactions=features.get("transactions"),
            security_validation=features.get("security_validation"),
            observability=features.get("observability"),
        )

    def _validate_contract(self, contract: ModelEnhancedContract) -> None:
        """
        Validate enhanced contract.

        Checks:
        - Mixin existence
        - Mixin dependency satisfaction
        - Feature overlap detection
        - NodeEffect built-in vs mixin overlap
        """
        errors = []

        # Collect mixin validation errors
        for mixin in contract.mixins:
            errors.extend(mixin.validation_errors)

        # Check mixin dependencies
        for mixin in contract.mixins:
            if not mixin.enabled:
                continue

            mixin_meta = self.available_mixins.get(mixin.name)
            if not mixin_meta:
                continue

            for dep_name in mixin_meta.get("dependencies", []):
                if not any(m.name == dep_name and m.enabled for m in contract.mixins):
                    errors.append(
                        f"Mixin '{mixin.name}' requires '{dep_name}' "
                        f"but it's not enabled"
                    )

        # Detect feature overlap
        if contract.advanced_features:
            # Circuit breaker is built into NodeEffect
            if contract.advanced_features.circuit_breaker:
                # This is fine - NodeEffect provides it
                pass

            # Retry policy is built into NodeEffect
            if contract.advanced_features.retry_policy:
                # This is fine - NodeEffect provides it
                pass

        # Set validation results
        if errors:
            contract.has_errors = True
            contract.validation_errors = errors
```

**Deliverable**: Enhanced ContractIntrospector with tests

**Action Items**:
- [ ] Create `ModelMixinDeclaration`, `ModelAdvancedFeatures`, `ModelEnhancedContract`
- [ ] Implement `_load_mixin_registry()` - load from catalog
- [ ] Implement `_parse_mixin_declaration()` with validation
- [ ] Implement `_parse_advanced_features()`
- [ ] Implement `_validate_contract()` with overlap detection
- [ ] Add unit tests for each validation rule
- [ ] Add integration test with real contracts

### 3.2 Mixin Injector Implementation

**Goal**: Generate code that imports and applies mixins to nodes.

**Implementation**:

```python
# src/omninode_bridge/codegen/mixin_injector.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelGeneratedImports:
    """Generated import statements."""
    standard_library: list[str]
    third_party: list[str]
    omnibase_core: list[str]
    omnibase_mixins: list[str]
    project_local: list[str]

@dataclass
class ModelGeneratedClass:
    """Generated class definition."""
    class_name: str
    base_classes: list[str]
    docstring: str
    init_method: str
    initialize_method: str
    methods: list[str]

class MixinInjector:
    """
    Generate code that imports and applies mixins to nodes.

    Responsibilities:
    - Generate import statements for mixins
    - Generate class inheritance chain
    - Generate mixin initialization code
    - Generate configuration setup
    """

    def generate_imports(
        self,
        contract: ModelEnhancedContract,
    ) -> ModelGeneratedImports:
        """
        Generate import statements for node file.

        Args:
            contract: Enhanced contract with mixin declarations

        Returns:
            Organized import statements
        """
        imports = ModelGeneratedImports(
            standard_library=[],
            third_party=[],
            omnibase_core=[],
            omnibase_mixins=[],
            project_local=[],
        )

        # Always import NodeEffect base class
        imports.omnibase_core.append(
            "from omnibase_core.nodes.node_effect import NodeEffect"
        )

        # Import mixins
        for mixin in contract.mixins:
            if not mixin.enabled:
                continue

            # e.g., from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
            imports.omnibase_mixins.append(
                f"from {mixin.import_path.rsplit('.', 1)[0]} import {mixin.name}"
            )

        # Import models for advanced features
        if contract.advanced_features:
            if contract.advanced_features.circuit_breaker:
                imports.omnibase_core.append(
                    "from omnibase_core.nodes.model_circuit_breaker import ModelCircuitBreaker"
                )

        return imports

    def generate_class_definition(
        self,
        contract: ModelEnhancedContract,
    ) -> ModelGeneratedClass:
        """
        Generate class definition with mixin inheritance.

        Args:
            contract: Enhanced contract with mixin declarations

        Returns:
            Generated class code
        """
        # Build inheritance chain: NodeEffect + Mixins
        base_classes = ["NodeEffect"]
        base_classes.extend(
            mixin.name for mixin in contract.mixins if mixin.enabled
        )

        # Generate class docstring
        docstring = self._generate_docstring(contract)

        # Generate __init__ method
        init_method = self._generate_init_method(contract)

        # Generate initialize() method (async)
        initialize_method = self._generate_initialize_method(contract)

        # Generate mixin-specific methods
        methods = self._generate_mixin_methods(contract)

        return ModelGeneratedClass(
            class_name=f"Node{contract.node_name}",
            base_classes=base_classes,
            docstring=docstring,
            init_method=init_method,
            initialize_method=initialize_method,
            methods=methods,
        )

    def _generate_docstring(self, contract: ModelEnhancedContract) -> str:
        """Generate comprehensive docstring."""
        lines = [
            f'"""',
            f"{contract.description}",
            "",
            "ONEX v2.0 Compliant Effect Node",
            "",
            "Capabilities:",
        ]

        # List NodeEffect built-in features
        lines.append("  Built-in Features (NodeEffect):")
        lines.append("    - Circuit breakers with failure threshold")
        lines.append("    - Retry policies with exponential backoff")
        lines.append("    - Transaction support with rollback")
        lines.append("    - Concurrent execution control")
        lines.append("    - Performance metrics tracking")

        # List mixin features
        if contract.mixins:
            lines.append("")
            lines.append("  Enhanced Features (Mixins):")
            for mixin in contract.mixins:
                if mixin.enabled:
                    lines.append(f"    - {mixin.name}")

        lines.append('"""')
        return "\n".join(lines)

    def _generate_init_method(self, contract: ModelEnhancedContract) -> str:
        """
        Generate __init__ method with mixin initialization.

        Example output:
            def __init__(self, container: ModelContainer):
                # Initialize base classes
                super().__init__(container)

                # Mixin configuration
                self.health_check_config = {
                    "check_interval_ms": 30000,
                    "timeout_seconds": 5.0,
                }
        """
        lines = [
            "    def __init__(self, container: ModelContainer):",
            '        """Initialize node with container and mixins."""',
            "        # Initialize base classes (NodeEffect + Mixins)",
            "        super().__init__(container)",
            "",
        ]

        # Generate mixin configuration
        for mixin in contract.mixins:
            if not mixin.enabled or not mixin.config:
                continue

            config_name = f"{mixin.name.lower().replace('mixin', '')}_config"
            lines.append(f"        # Configure {mixin.name}")
            lines.append(f"        self.{config_name} = {{")
            for key, value in mixin.config.items():
                lines.append(f'            "{key}": {repr(value)},')
            lines.append("        }")
            lines.append("")

        return "\n".join(lines)

    def _generate_initialize_method(self, contract: ModelEnhancedContract) -> str:
        """
        Generate async initialize() method.

        Calls super().initialize() and mixin setup methods.
        """
        lines = [
            "    async def initialize(self) -> None:",
            '        """Initialize node resources and mixins."""',
            "        # Initialize base NodeEffect",
            "        await super().initialize()",
            "",
        ]

        # Initialize mixins
        for mixin in contract.mixins:
            if not mixin.enabled:
                continue

            # Some mixins have setup methods
            if mixin.name == "MixinHealthCheck":
                lines.append("        # Setup health checks")
                lines.append("        self.register_health_check(")
                lines.append('            "database",')
                lines.append("            self._check_database_health,")
                lines.append("        )")
                lines.append("")

            elif mixin.name == "MixinEventDrivenNode":
                lines.append("        # Setup event consumption")
                lines.append("        await self.start_event_consumption()")
                lines.append("")

        return "\n".join(lines)

    def _generate_mixin_methods(self, contract: ModelEnhancedContract) -> list[str]:
        """Generate mixin-required methods."""
        methods = []

        for mixin in contract.mixins:
            if not mixin.enabled:
                continue

            # MixinHealthCheck requires health check methods
            if mixin.name == "MixinHealthCheck":
                method = """
    async def _check_database_health(self) -> bool:
        \"\"\"Check database health status.\"\"\"
        try:
            # TODO: Implement actual health check
            return await self._postgres_client.check_health()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
"""
                methods.append(method)

        return methods

    def generate_node_file(
        self,
        contract: ModelEnhancedContract,
    ) -> str:
        """
        Generate complete node.py file with mixins.

        Args:
            contract: Enhanced contract

        Returns:
            Complete node.py file content
        """
        imports = self.generate_imports(contract)
        class_def = self.generate_class_definition(contract)

        # Build file content
        lines = [
            '#!/usr/bin/env python3',
            '"""',
            f'{contract.node_name} - ONEX v2.0 Effect Node',
            '',
            'Generated by OmniNode Code Generator',
            'DO NOT EDIT MANUALLY',
            '"""',
            '',
        ]

        # Add imports (sorted and organized)
        if imports.standard_library:
            lines.extend(sorted(imports.standard_library))
            lines.append("")

        if imports.third_party:
            lines.extend(sorted(imports.third_party))
            lines.append("")

        if imports.omnibase_core:
            lines.extend(sorted(imports.omnibase_core))
            lines.append("")

        if imports.omnibase_mixins:
            lines.extend(sorted(imports.omnibase_mixins))
            lines.append("")

        # Add class definition
        inheritance = ", ".join(class_def.base_classes)
        lines.append(f"class {class_def.class_name}({inheritance}):")
        lines.append(class_def.docstring)
        lines.append("")
        lines.append(class_def.init_method)
        lines.append("")
        lines.append(class_def.initialize_method)

        # Add methods
        for method in class_def.methods:
            lines.append("")
            lines.append(method)

        return "\n".join(lines)
```

**Deliverable**: MixinInjector with comprehensive tests

**Action Items**:
- [ ] Implement `generate_imports()` with proper organization
- [ ] Implement `generate_class_definition()` with inheritance
- [ ] Implement `_generate_init_method()` with mixin config
- [ ] Implement `_generate_initialize_method()` with mixin setup
- [ ] Implement `_generate_mixin_methods()` for required methods
- [ ] Implement `generate_node_file()` for complete file
- [ ] Add unit tests for each generation method
- [ ] Add integration test generating full node

### 3.3 Template System Updates

**Goal**: Create Jinja2 templates for node generation (currently only exist for tests).

**New Templates to Create**:

```
src/omninode_bridge/codegen/templates/
├── node_templates/
│   ├── node_effect.py.j2           # Effect node template
│   ├── node_compute.py.j2          # Compute node template
│   ├── node_orchestrator.py.j2    # Orchestrator template
│   ├── node_reducer.py.j2          # Reducer template
│   ├── __init__.py.j2              # Module init
│   └── contract.yaml.j2            # Contract template
├── mixin_snippets/
│   ├── health_check_init.j2        # Health check setup
│   ├── metrics_init.j2             # Metrics setup
│   ├── event_driven_init.j2        # Event consumption
│   └── service_registry_init.j2    # Service registration
└── test_templates/  # Existing
    ├── test_unit.py.j2
    ├── test_integration.py.j2
    └── ...
```

**Example Template - node_effect.py.j2**:

```jinja2
#!/usr/bin/env python3
"""
{{ node_name }} - ONEX v2.0 Effect Node

Generated by OmniNode Code Generator
{% if generation_timestamp %}Generated at: {{ generation_timestamp }}{% endif %}

{{ description }}
"""

# Standard library imports
{% for import in imports.standard_library | sort %}
{{ import }}
{% endfor %}

# Third-party imports
{% for import in imports.third_party | sort %}
{{ import }}
{% endfor %}

# omnibase_core imports
{% for import in imports.omnibase_core | sort %}
{{ import }}
{% endfor %}

# Mixin imports
{% for import in imports.omnibase_mixins | sort %}
{{ import }}
{% endfor %}

# Project local imports
{% for import in imports.project_local | sort %}
{{ import }}
{% endfor %}

logger = logging.getLogger(__name__)


class {{ class_name }}({{ base_classes | join(', ') }}):
    {{ docstring | indent(4) }}

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize {{ node_name }} with container and mixins.

        Args:
            container: ONEX container for dependency injection
        """
        # Initialize base classes
        super().__init__(container)

        {% if mixin_configs %}
        # Mixin configuration
        {% for mixin_name, config in mixin_configs.items() %}
        self.{{ mixin_name.lower().replace('mixin', '') }}_config = {
            {% for key, value in config.items() %}
            "{{ key }}": {{ value | repr }},
            {% endfor %}
        }
        {% endfor %}
        {% endif %}

        {% if advanced_features.circuit_breaker %}
        # Circuit breaker configuration
        self._circuit_breakers = {}
        {% for service_name, cb_config in advanced_features.circuit_breaker.services.items() %}
        self._circuit_breakers["{{ service_name }}"] = ModelCircuitBreaker(
            failure_threshold={{ cb_config.failure_threshold }},
            recovery_timeout_seconds={{ cb_config.recovery_timeout_ms // 1000 }},
        )
        {% endfor %}
        {% endif %}

    async def initialize(self) -> None:
        """
        Initialize node resources and mixins.

        Performs:
        - Base NodeEffect initialization
        {% for mixin in mixins if mixin.enabled %}
        - {{ mixin.name }} setup
        {% endfor %}
        """
        # Initialize base NodeEffect
        await super().initialize()

        {% if 'MixinHealthCheck' in enabled_mixins %}
        # Setup health checks
        {% for component in health_check_components %}
        self.register_health_check(
            "{{ component.name }}",
            self._check_{{ component.name }}_health,
            critical={{ component.critical }},
            timeout_seconds={{ component.timeout_seconds }},
        )
        {% endfor %}
        {% endif %}

        {% if 'MixinEventDrivenNode' in enabled_mixins %}
        # Setup event consumption
        await self.start_event_consumption()
        {% endif %}

        {% if 'MixinServiceRegistry' in enabled_mixins %}
        # Register with service registry
        await self.register_service()
        {% endif %}

        emit_log_event(
            LogLevel.INFO,
            "{{ node_name }} initialized",
            {"node_id": str(self.node_id)},
        )

    async def shutdown(self) -> None:
        """Cleanup node resources."""
        {% if 'MixinEventDrivenNode' in enabled_mixins %}
        await self.stop_event_consumption()
        {% endif %}

        {% if 'MixinServiceRegistry' in enabled_mixins %}
        await self.deregister_service()
        {% endif %}

        await super().shutdown()

    {% if 'MixinHealthCheck' in enabled_mixins %}
    # Health check methods (generated for each component)
    {% for component in health_check_components %}
    async def _check_{{ component.name }}_health(self) -> bool:
        """Check {{ component.name }} health status."""
        try:
            # TODO: Implement actual health check for {{ component.name }}
            return True
        except Exception as e:
            logger.error(f"{{ component.name }} health check failed: {e}")
            return False
    {% endfor %}
    {% endif %}

    # Business logic methods (LLM-generated or manual)
    {% for operation in io_operations %}
    async def {{ operation.name }}(
        self,
        input_data: {{ operation.input_model }},
    ) -> {{ operation.output_model }}:
        """
        {{ operation.description }}

        Args:
            input_data: {{ operation.input_model }} instance

        Returns:
            {{ operation.output_model }} with operation results

        Raises:
            ModelOnexError: On operation failure
        """
        # TODO: Implement {{ operation.name }}
        raise NotImplementedError("{{ operation.name }} not yet implemented")
    {% endfor %}
```

**Deliverable**: Complete template set with Jinja2 templates

**Action Items**:
- [ ] Create `node_effect.py.j2` template with mixin support
- [ ] Create `node_compute.py.j2`, `node_orchestrator.py.j2`, `node_reducer.py.j2`
- [ ] Create mixin snippet templates for reusable blocks
- [ ] Create `contract.yaml.j2` for contract generation
- [ ] Update `TemplateEngine` to use new templates
- [ ] Add template validation tests
- [ ] Document template variables and filters

### 3.4 Validation Pipeline

**Goal**: Validate generated code for correctness, ONEX compliance, and security.

**Validation Stages**:

```python
# src/omninode_bridge/codegen/validation/validator.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class EnumValidationStage(str, Enum):
    """Validation stage identifier."""
    SYNTAX = "syntax"              # Python syntax check
    AST = "ast"                    # AST parsing
    TYPE_CHECKING = "type_checking" # mypy static analysis
    IMPORTS = "imports"            # Import resolution
    ONEX_COMPLIANCE = "onex_compliance"  # ONEX v2.0 compliance
    SECURITY = "security"          # Security scan
    PERFORMANCE = "performance"    # Performance check

@dataclass
class ModelValidationResult:
    """Validation result for a single stage."""
    stage: EnumValidationStage
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

class NodeValidator:
    """
    Validate generated nodes for correctness and compliance.
    """

    def __init__(
        self,
        enable_type_checking: bool = True,
        enable_security_scan: bool = True,
    ):
        """Initialize validator with configuration."""
        self.enable_type_checking = enable_type_checking
        self.enable_security_scan = enable_security_scan

    async def validate_generated_node(
        self,
        node_file_content: str,
        contract: ModelEnhancedContract,
    ) -> list[ModelValidationResult]:
        """
        Run all validation stages on generated node.

        Args:
            node_file_content: Generated node.py content
            contract: Contract used for generation

        Returns:
            List of validation results (one per stage)
        """
        results = []

        # Stage 1: Syntax check
        results.append(await self._validate_syntax(node_file_content))

        # Stage 2: AST parsing
        results.append(await self._validate_ast(node_file_content))

        # Stage 3: Import resolution
        results.append(await self._validate_imports(node_file_content))

        # Stage 4: Type checking (optional)
        if self.enable_type_checking:
            results.append(await self._validate_types(node_file_content))

        # Stage 5: ONEX compliance
        results.append(
            await self._validate_onex_compliance(node_file_content, contract)
        )

        # Stage 6: Security scan (optional)
        if self.enable_security_scan:
            results.append(await self._validate_security(node_file_content))

        return results

    async def _validate_syntax(self, code: str) -> ModelValidationResult:
        """Validate Python syntax."""
        start = time.time()
        errors = []

        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

        return ModelValidationResult(
            stage=EnumValidationStage.SYNTAX,
            passed=len(errors) == 0,
            errors=errors,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _validate_ast(self, code: str) -> ModelValidationResult:
        """Validate AST structure."""
        start = time.time()
        errors = []
        warnings = []

        try:
            tree = ast.parse(code)

            # Check for required class
            classes = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            ]
            if not classes:
                errors.append("No class definition found")

            # Check for required methods
            required_methods = {"__init__", "initialize", "shutdown"}
            for cls in classes:
                methods = {
                    node.name
                    for node in cls.body
                    if isinstance(node, ast.FunctionDef)
                }
                missing = required_methods - methods
                if missing:
                    warnings.append(
                        f"Class {cls.name} missing methods: {missing}"
                    )

        except Exception as e:
            errors.append(f"AST parsing failed: {e}")

        return ModelValidationResult(
            stage=EnumValidationStage.AST,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _validate_imports(self, code: str) -> ModelValidationResult:
        """Validate all imports can be resolved."""
        start = time.time()
        errors = []
        warnings = []

        try:
            tree = ast.parse(code)

            # Extract all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Check each import
            for import_name in imports:
                try:
                    __import__(import_name.split('.')[0])
                except ImportError:
                    # Allow omnibase_core even if not installed yet
                    if not import_name.startswith("omnibase_"):
                        warnings.append(f"Import '{import_name}' may not resolve")

        except Exception as e:
            errors.append(f"Import validation failed: {e}")

        return ModelValidationResult(
            stage=EnumValidationStage.IMPORTS,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _validate_types(self, code: str) -> ModelValidationResult:
        """Run mypy type checking."""
        start = time.time()
        errors = []

        # TODO: Integrate with mypy API
        # For now, just a placeholder

        return ModelValidationResult(
            stage=EnumValidationStage.TYPE_CHECKING,
            passed=True,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _validate_onex_compliance(
        self,
        code: str,
        contract: ModelEnhancedContract,
    ) -> ModelValidationResult:
        """Check ONEX v2.0 compliance."""
        start = time.time()
        errors = []
        warnings = []

        # Check 1: NodeEffect inheritance
        if "NodeEffect" not in code:
            errors.append("Must inherit from NodeEffect")

        # Check 2: Mixins present
        for mixin in contract.mixins:
            if mixin.enabled and mixin.name not in code:
                warnings.append(f"Mixin '{mixin.name}' not found in code")

        # Check 3: Required methods
        required = ["__init__", "initialize", "shutdown"]
        for method in required:
            if f"def {method}" not in code and f"async def {method}" not in code:
                errors.append(f"Missing required method: {method}")

        # Check 4: Async patterns
        if "async def initialize" not in code:
            errors.append("initialize() must be async")

        return ModelValidationResult(
            stage=EnumValidationStage.ONEX_COMPLIANCE,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _validate_security(self, code: str) -> ModelValidationResult:
        """Run security scan (bandit-like)."""
        start = time.time()
        errors = []
        warnings = []

        # Check for dangerous patterns
        dangerous_patterns = [
            ("eval(", "Use of eval() is dangerous"),
            ("exec(", "Use of exec() is dangerous"),
            ("__import__", "Dynamic imports should be avoided"),
            ("os.system", "Use subprocess instead of os.system"),
        ]

        for pattern, message in dangerous_patterns:
            if pattern in code:
                warnings.append(message)

        return ModelValidationResult(
            stage=EnumValidationStage.SECURITY,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=(time.time() - start) * 1000,
        )
```

**Deliverable**: Comprehensive validation pipeline with tests

**Action Items**:
- [ ] Implement `NodeValidator` with all validation stages
- [ ] Integrate with existing `QualityGateValidator`
- [ ] Add mypy integration for type checking
- [ ] Add security pattern detection
- [ ] Add ONEX compliance checks
- [ ] Create validation result reporting
- [ ] Add tests for each validation stage

---

## Phase 4: Implementation & Testing (Week 3-4)

### 4.1 Generator Integration

**Goal**: Integrate enhanced components into CodeGenerationService.

**Updated Flow**:

```python
# src/omninode_bridge/codegen/service.py (updated)
class CodeGenerationService:
    """
    Enhanced code generation service with mixin support.
    """

    def __init__(self):
        """Initialize with enhanced components."""
        self.contract_introspector = ContractIntrospector()  # Enhanced
        self.mixin_injector = MixinInjector()  # NEW
        self.node_validator = NodeValidator()  # NEW
        self.template_engine = TemplateEngine()  # Enhanced with node templates
        self.business_logic_generator = BusinessLogicGenerator()

    async def generate_node(
        self,
        requirements: ModelPRDRequirements,
        strategy: str = "auto",
        enable_llm: bool = True,
        enable_mixins: bool = True,  # NEW
        validation_level: str = "strict",
    ) -> ModelGenerationResult:
        """
        Generate node with mixin support.

        Args:
            requirements: PRD requirements
            strategy: Generation strategy
            enable_llm: Enable LLM for business logic
            enable_mixins: Enable mixin injection (NEW)
            validation_level: Validation strictness

        Returns:
            ModelGenerationResult with artifacts and metrics
        """
        # Step 1: Parse contract with mixin support
        contract = self.contract_introspector.parse_contract(
            requirements.contract_path
        )

        if contract.has_errors:
            raise ContractValidationError(contract.validation_errors)

        # Step 2: Generate node file with mixins
        if enable_mixins:
            node_content = self.mixin_injector.generate_node_file(contract)
        else:
            # Fallback to template-only generation
            node_content = await self.template_engine.generate(
                template_name="node_effect.py.j2",
                context={"contract": contract},
            )

        # Step 3: Enhance with LLM business logic (optional)
        if enable_llm:
            artifacts = ModelGeneratedArtifacts(
                node_name=contract.node_name,
                node_file=node_content,
                # ... other files
            )
            enhanced = await self.business_logic_generator.enhance_artifacts(
                artifacts=artifacts,
                requirements=requirements,
            )
            node_content = enhanced.enhanced_node_file

        # Step 4: Validate generated code
        validation_results = await self.node_validator.validate_generated_node(
            node_file_content=node_content,
            contract=contract,
        )

        # Check for validation failures
        if any(not result.passed for result in validation_results):
            failed_stages = [
                result.stage.value
                for result in validation_results
                if not result.passed
            ]
            raise GenerationValidationError(
                f"Validation failed at stages: {failed_stages}",
                validation_results=validation_results,
            )

        # Step 5: Generate supporting files (tests, models, etc.)
        artifacts = await self._generate_supporting_files(
            contract=contract,
            node_content=node_content,
        )

        return ModelGenerationResult(
            success=True,
            artifacts=artifacts,
            validation_results=validation_results,
        )
```

**Deliverable**: Integrated CodeGenerationService

**Action Items**:
- [ ] Update `CodeGenerationService.__init__()` with new components
- [ ] Update `generate_node()` method with mixin support
- [ ] Add `enable_mixins` flag
- [ ] Integrate validation pipeline
- [ ] Add error handling for validation failures
- [ ] Update CLI to support new flags
- [ ] Add integration tests

### 4.2 Test Generation

**Goal**: Regenerate key nodes and validate correctness.

**Test Targets**:

1. **database_adapter_effect** (HIGH PRIORITY)
   - Currently 500+ LOC with manual implementations
   - Should reduce to ~300 LOC with mixins
   - Expected mixins: MixinHealthCheck, MixinMetrics, MixinEventDrivenNode

2. **llm_effect** (MEDIUM PRIORITY)
   - Currently 300+ LOC
   - Already uses ModelCircuitBreaker from omnibase_core
   - Should add MixinMetrics for comprehensive metrics

3. **orchestrator** (MEDIUM PRIORITY)
   - Currently 400+ LOC
   - Should add MixinHealthCheck, MixinMetrics

**Regeneration Process**:

```bash
# 1. Backup existing node
cp -r src/omninode_bridge/nodes/database_adapter_effect/ \
      src/omninode_bridge/nodes/database_adapter_effect.backup/

# 2. Update contract with mixin declarations
vim src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/contract.yaml
# Add mixins section (see Phase 2.1 example)

# 3. Regenerate node
omninode-generate \
  --contract src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/contract.yaml \
  --enable-mixins \
  --enable-llm \
  --validation-level strict \
  --output src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/

# 4. Compare generated vs original
diff -u \
  src/omninode_bridge/nodes/database_adapter_effect.backup/v1_0_0/node.py \
  src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/node.py

# 5. Run tests
pytest tests/unit/nodes/database_adapter_effect/ -v
pytest tests/integration/nodes/database_adapter_effect/ -v

# 6. Check metrics
wc -l src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/node.py
# Expected: 250-300 LOC (down from 500+)
```

**Deliverable**: Regenerated nodes + comparison reports

**Action Items**:
- [ ] Update contracts for database_adapter_effect, llm_effect, orchestrator
- [ ] Run regeneration for each node
- [ ] Compare generated vs original (LOC, features, structure)
- [ ] Run all existing tests (expect 100% pass)
- [ ] Measure LOC reduction
- [ ] Document changes and improvements
- [ ] Create comparison reports

### 4.3 Quality Metrics Collection

**Goal**: Measure improvement from mixin-enhanced generation.

**Metrics to Track**:

```python
@dataclass
class ModelRegenerationMetrics:
    """Metrics for regenerated node comparison."""
    node_name: str

    # LOC metrics
    original_loc: int
    generated_loc: int
    loc_reduction_percent: float

    # Feature comparison
    features_original: list[str]
    features_generated: list[str]
    features_added: list[str]
    features_removed: list[str]

    # Mixin usage
    mixins_applied: list[str]
    mixins_count: int

    # Test results
    tests_passed: int
    tests_failed: int
    test_pass_rate: float

    # Performance
    generation_time_seconds: float
    validation_time_seconds: float
```

**Comparison Report Example**:

```markdown
# Database Adapter Effect - Regeneration Report

## LOC Comparison
- **Original**: 523 LOC
- **Generated**: 287 LOC
- **Reduction**: 45.1% (236 lines removed)

## Feature Parity
| Feature | Original | Generated | Status |
|---------|----------|-----------|--------|
| Circuit Breaker | Manual (78 LOC) | NodeEffect Built-in | ✅ Replaced |
| Health Checks | Custom Mixin (52 LOC) | MixinHealthCheck | ✅ Replaced |
| Metrics | Manual (94 LOC) | MixinMetrics | ✅ Replaced |
| DLQ Handling | Manual (67 LOC) | Manual (32 LOC) | ✅ Simplified |
| Event Consumption | Manual (103 LOC) | MixinEventDrivenNode | ✅ Replaced |
| **Total** | **394 LOC** | **32 LOC** | **91.9% reduction** |

## Mixins Applied
1. **MixinHealthCheck** - Health monitoring
2. **MixinMetrics** - Performance metrics
3. **MixinEventDrivenNode** - Event consumption
4. **MixinLogData** - Structured logging

## Test Results
- **Unit Tests**: 45/45 passed (100%)
- **Integration Tests**: 12/12 passed (100%)
- **Performance Tests**: 3/3 passed (100%)
- **Total**: 60/60 passed (100%)

## Performance
- **Generation Time**: 2.3 seconds
- **Validation Time**: 0.8 seconds
- **Total Time**: 3.1 seconds

## Quality Gates
- ✅ Syntax validation
- ✅ AST validation
- ✅ Import resolution
- ✅ Type checking (mypy)
- ✅ ONEX compliance
- ✅ Security scan

## Conclusion
**SUCCESS** - Generated node achieves feature parity with 45.1% LOC reduction.
Mixins successfully replaced 362 lines of manual implementation.
```

**Deliverable**: Metrics collection system + comparison reports

**Action Items**:
- [ ] Implement `ModelRegenerationMetrics` data class
- [ ] Create metrics collection script
- [ ] Generate comparison reports for each node
- [ ] Create summary report across all nodes
- [ ] Add visualization (charts, graphs)
- [ ] Document findings and recommendations

---

## Phase 5: Rollout & Migration (Week 5-6)

### 5.1 Node-by-Node Migration

**Goal**: Migrate all existing nodes to mixin-enhanced generation.

**Migration Order** (by priority):

```
WEEK 5:
┌─────────────────────────────────────────────────────┐
│ HIGH PRIORITY (Immediate Value)                     │
├─────────────────────────────────────────────────────┤
│ 1. database_adapter_effect                          │
│    - Most manual code to replace                    │
│    - Proves mixin value                             │
│    - Expected LOC reduction: 45%                    │
│                                                      │
│ 2. llm_effect                                       │
│    - Already uses some omnibase_core features       │
│    - Add comprehensive metrics                      │
│    - Expected LOC reduction: 25%                    │
│                                                      │
│ 3. orchestrator                                     │
│    - Workflow coordination benefits from metrics    │
│    - Health checks critical for production          │
│    - Expected LOC reduction: 30%                    │
└─────────────────────────────────────────────────────┘

WEEK 6:
┌─────────────────────────────────────────────────────┐
│ MEDIUM PRIORITY (Steady Improvement)                │
├─────────────────────────────────────────────────────┤
│ 4. reducer                                          │
│ 5. registry                                         │
│ 6. deployment_receiver_effect                       │
│ 7. deployment_sender_effect                         │
└─────────────────────────────────────────────────────┘

WEEK 7 (OPTIONAL):
┌─────────────────────────────────────────────────────┐
│ LOW PRIORITY (Polish)                               │
├─────────────────────────────────────────────────────┤
│ 8. distributed_lock_effect                          │
│ 9. store_effect                                     │
│ 10. test_generator_effect                           │
│ 11. codegen_orchestrator                            │
│ 12. codegen_metrics_reducer                         │
└─────────────────────────────────────────────────────┘
```

**Migration Checklist** (per node):

```markdown
## Node Migration Checklist: <node_name>

### Pre-Migration
- [ ] Create backup: `<node_name>.backup/`
- [ ] Document current features and LOC
- [ ] Run baseline tests (capture results)
- [ ] Review manual implementations
- [ ] Identify mixins needed

### Contract Update
- [ ] Add `mixins` section to contract.yaml
- [ ] Add `advanced_features` section
- [ ] Validate contract schema
- [ ] Review mixin configurations

### Regeneration
- [ ] Run generator with `--enable-mixins`
- [ ] Review generated code
- [ ] Compare with original
- [ ] Check for regressions

### Testing
- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Run performance tests
- [ ] Verify feature parity

### Validation
- [ ] Syntax validation
- [ ] Type checking (mypy)
- [ ] ONEX compliance
- [ ] Security scan
- [ ] Code review

### Metrics
- [ ] Measure LOC reduction
- [ ] Collect generation metrics
- [ ] Create comparison report
- [ ] Update summary report

### Deployment
- [ ] Commit to version control
- [ ] Update documentation
- [ ] Deploy to dev environment
- [ ] Monitor for issues
- [ ] Deploy to production

### Post-Migration
- [ ] Remove backup (after 30 days)
- [ ] Update CHANGELOG.md
- [ ] Share learnings with team
```

**Deliverable**: Migration plan + per-node checklists

**Action Items**:
- [ ] Create migration order spreadsheet
- [ ] Create per-node migration checklist
- [ ] Assign nodes to team members (if applicable)
- [ ] Schedule migration timeline
- [ ] Set up monitoring for regressions
- [ ] Create rollback procedures

### 5.2 Documentation Updates

**Goal**: Update all documentation to reflect mixin-enhanced generation.

**Documentation to Update**:

1. **Contract Schema Reference** (`docs/reference/CONTRACT_SCHEMA.md`)
   - Add `mixins` section documentation
   - Add `advanced_features` section documentation
   - Add examples for each mixin
   - Add migration guide

2. **Code Generation Guide** (`docs/guides/CODE_GENERATION_GUIDE.md`)
   - Update with mixin workflow
   - Add examples of generated nodes
   - Add comparison before/after
   - Add troubleshooting section

3. **Mixin Catalog** (`docs/reference/OMNIBASE_CORE_MIXIN_CATALOG.md`)
   - Document all 33 mixins
   - Include usage examples
   - Include configuration options
   - Include when-to-use guidance

4. **API Reference** (`docs/api/API_REFERENCE.md`)
   - Update CodeGenerationService API
   - Document new flags and options
   - Add examples

5. **Getting Started** (`docs/GETTING_STARTED.md`)
   - Update quickstart examples
   - Show mixin-enhanced contracts
   - Update CLI examples

6. **Contributing Guide** (`docs/CONTRIBUTING.md`)
   - Add guidelines for using mixins
   - Add checklist for new nodes
   - Add review criteria

**Deliverable**: Updated documentation set

**Action Items**:
- [ ] Update CONTRACT_SCHEMA.md with mixin sections
- [ ] Create OMNIBASE_CORE_MIXIN_CATALOG.md (comprehensive)
- [ ] Update CODE_GENERATION_GUIDE.md with workflow
- [ ] Update API_REFERENCE.md with new methods
- [ ] Update GETTING_STARTED.md with examples
- [ ] Update CONTRIBUTING.md with guidelines
- [ ] Add diagrams and visualizations
- [ ] Review and proofread all docs

### 5.3 Training & Knowledge Transfer

**Goal**: Ensure team can effectively use mixin-enhanced generation.

**Training Materials**:

1. **Workshop: Mixin-Enhanced Code Generation** (2 hours)
   - Overview of omnibase_core mixins
   - Contract schema updates
   - Live demonstration of regeneration
   - Hands-on exercises
   - Q&A

2. **Tutorial Videos**
   - "Intro to omnibase_core Mixins" (10 min)
   - "Writing Mixin-Enhanced Contracts" (15 min)
   - "Regenerating Existing Nodes" (12 min)
   - "Validation and Quality Gates" (8 min)
   - "Troubleshooting Common Issues" (10 min)

3. **Reference Materials**
   - Mixin quick reference card (PDF)
   - Contract template library
   - Common patterns cheat sheet
   - Troubleshooting guide

**Deliverable**: Training materials + workshop slides

**Action Items**:
- [ ] Create workshop slides
- [ ] Prepare live demo environment
- [ ] Create hands-on exercises
- [ ] Record tutorial videos
- [ ] Create quick reference card
- [ ] Create contract template library
- [ ] Schedule workshop sessions
- [ ] Collect feedback and iterate

---

## Phase 6: Continuous Improvement (Ongoing)

### 6.1 Feedback Collection

**Goal**: Continuously improve generator based on usage.

**Feedback Mechanisms**:

1. **Usage Metrics**
   - Track generator invocations
   - Track mixin usage frequency
   - Track validation failures
   - Track generation times

2. **Developer Feedback**
   - Survey after each node generation
   - Weekly team retrospectives
   - GitHub issues for feature requests
   - Slack channel for questions

3. **Code Quality Metrics**
   - LOC trends over time
   - Test coverage trends
   - Bug density in generated vs hand-coded
   - Performance metrics

**Deliverable**: Feedback system + metrics dashboard

**Action Items**:
- [ ] Implement usage metrics collection
- [ ] Create metrics dashboard (Grafana?)
- [ ] Set up feedback surveys
- [ ] Create feedback Slack channel
- [ ] Schedule regular retrospectives
- [ ] Review metrics monthly
- [ ] Prioritize improvements based on feedback

### 6.2 Mixin Library Expansion

**Goal**: Add new mixins as patterns emerge.

**Candidate New Mixins**:

1. **MixinDeadLetterQueue** (if pattern is common)
   - DLQ management
   - Retry logic
   - Alert thresholds

2. **MixinSecurityValidation** (if pattern is common)
   - Input sanitization
   - SQL injection prevention
   - Log redaction

3. **MixinPerformanceOptimization**
   - Query optimization
   - Batch processing
   - Connection pooling

**Process**:
1. Identify common patterns across 3+ nodes
2. Extract into reusable mixin
3. Add to omnibase_core (via PR)
4. Update catalog
5. Update generator
6. Announce to team

**Deliverable**: Process for mixin expansion

**Action Items**:
- [ ] Create pattern identification process
- [ ] Set up contribution workflow to omnibase_core
- [ ] Document mixin creation guidelines
- [ ] Create mixin template
- [ ] Add tests for new mixins
- [ ] Update catalog when mixins added

### 6.3 Generator Performance Optimization

**Goal**: Reduce generation time and improve quality.

**Optimization Targets**:

1. **LLM Call Optimization**
   - Cache common patterns
   - Batch multiple method generations
   - Use faster models for simple cases
   - Target: 50% reduction in LLM calls

2. **Template Compilation**
   - Pre-compile Jinja2 templates
   - Cache template artifacts
   - Target: 30% faster template rendering

3. **Validation Pipeline**
   - Run stages in parallel
   - Cache validation results
   - Skip redundant checks
   - Target: 40% faster validation

4. **Quality Improvements**
   - Improve prompt engineering
   - Add more examples to prompts
   - Fine-tune model selection
   - Target: 20% fewer LLM retries

**Deliverable**: Optimized generator with metrics

**Action Items**:
- [ ] Profile current generation performance
- [ ] Identify bottlenecks
- [ ] Implement LLM call caching
- [ ] Implement template pre-compilation
- [ ] Parallelize validation stages
- [ ] Improve prompt engineering
- [ ] Measure improvements
- [ ] Document optimizations

---

## Success Metrics & Validation

### Overall Success Criteria

**Must-Have (Phase 1-5)**:
- ✅ All 33 omnibase_core mixins cataloged and documented
- ✅ Contract schema extended with `mixins` and `advanced_features`
- ✅ Generator produces mixin-enhanced nodes
- ✅ database_adapter_effect regenerated with 40%+ LOC reduction
- ✅ All existing tests pass (100%)
- ✅ Zero feature regressions
- ✅ Documentation updated

**Nice-to-Have (Phase 6)**:
- ⭐ LOC reduction across all nodes: 30-50%
- ⭐ Generation time < 5 seconds per node
- ⭐ Developer satisfaction > 4/5
- ⭐ New mixins added to omnibase_core

### Key Metrics

```python
@dataclass
class ModelProjectMetrics:
    """Overall project success metrics."""

    # LOC metrics
    total_loc_before: int
    total_loc_after: int
    total_loc_reduction_percent: float

    # Mixin usage
    nodes_using_mixins: int
    total_mixin_applications: int
    avg_mixins_per_node: float

    # Quality
    test_pass_rate: float  # Target: 100%
    validation_pass_rate: float  # Target: 95%
    feature_parity_score: float  # Target: 100%

    # Performance
    avg_generation_time_seconds: float  # Target: < 5s
    avg_validation_time_seconds: float  # Target: < 2s

    # Developer experience
    developer_satisfaction_score: float  # Target: > 4/5
    time_to_generate_node_minutes: float  # Target: < 10 min
    manual_fixes_required_percent: float  # Target: < 5%
```

### Validation Checkpoints

**End of Week 2**:
- Contract schema designed and validated
- Mixin catalog 50% complete
- Architecture design reviewed

**End of Week 4**:
- Generator implementation complete
- database_adapter_effect regenerated successfully
- All tests passing

**End of Week 6**:
- All high-priority nodes migrated
- Documentation 90% complete
- Training materials ready

**End of Week 8** (optional):
- All nodes migrated
- Metrics dashboard live
- Continuous improvement process established

---

## Risk Mitigation

### Risk 1: Breaking Existing Nodes

**Likelihood**: Medium
**Impact**: High

**Mitigation**:
- Create backups before regeneration
- Run comprehensive test suite
- Use feature flags for opt-in
- Gradual rollout (high → medium → low priority)
- Rollback procedures documented

### Risk 2: Mixin Incompatibilities

**Likelihood**: Low
**Impact**: Medium

**Mitigation**:
- Validate mixin dependencies in contract parser
- Test mixin combinations
- Document known incompatibilities
- Provide clear error messages

### Risk 3: LLM Quality Degradation

**Likelihood**: Medium
**Impact**: Medium

**Mitigation**:
- Improve prompt engineering
- Add more examples
- Implement validation gates
- Allow manual override
- Track quality metrics

### Risk 4: Performance Regression

**Likelihood**: Low
**Impact**: Medium

**Mitigation**:
- Run performance benchmarks
- Compare before/after metrics
- Optimize template rendering
- Cache LLM responses
- Profile and optimize bottlenecks

### Risk 5: Team Adoption

**Likelihood**: Low
**Impact**: Medium

**Mitigation**:
- Comprehensive training
- Clear documentation
- Easy-to-follow examples
- Support channel (Slack)
- Regular feedback sessions

---

## Appendix

### A. File Structure Changes

```
src/omninode_bridge/
├── codegen/
│   ├── contract_introspector.py     # ENHANCED (mixin parsing)
│   ├── mixin_injector.py            # NEW (mixin code generation)
│   ├── service.py                   # ENHANCED (mixin integration)
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validator.py             # NEW (comprehensive validation)
│   └── templates/
│       ├── node_templates/          # NEW (node generation templates)
│       │   ├── node_effect.py.j2
│       │   ├── node_compute.py.j2
│       │   ├── node_orchestrator.py.j2
│       │   └── node_reducer.py.j2
│       └── mixin_snippets/          # NEW (reusable mixin blocks)
│           ├── health_check_init.j2
│           ├── metrics_init.j2
│           └── event_driven_init.j2
│
├── nodes/
│   └── <node_name>/
│       └── v1_0_0/
│           ├── contract.yaml        # ENHANCED (with mixins section)
│           └── node.py              # REGENERATED (with mixins)
│
├── docs/
│   ├── reference/
│   │   ├── OMNIBASE_CORE_MIXIN_CATALOG.md  # NEW
│   │   └── CONTRACT_SCHEMA.md               # ENHANCED
│   ├── guides/
│   │   └── CODE_GENERATION_GUIDE.md         # ENHANCED
│   └── planning/
│       └── CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md  # THIS DOCUMENT
│
└── tests/
    ├── unit/codegen/
    │   ├── test_contract_introspector.py    # ENHANCED
    │   ├── test_mixin_injector.py           # NEW
    │   └── test_node_validator.py           # NEW
    └── integration/codegen/
        └── test_mixin_generation_e2e.py     # NEW
```

### B. Example Contracts (Before/After)

**Before** (llm_effect/v1_0_0/contract.yaml):
```yaml
name: llm_effect
version: {major: 1, minor: 0, patch: 0}
node_type: effect

error_handling:
  retry_policy:
    max_attempts: 3
  circuit_breaker:
    enabled: true
    failure_threshold: 5
```

**After** (llm_effect/v1_0_0/contract.yaml):
```yaml
name: llm_effect
version: {major: 1, minor: 0, patch: 0}
node_type: effect

# NEW: Mixin declarations
mixins:
  - name: MixinMetrics
    enabled: true
    config:
      collect_latency: true
      collect_token_usage: true
      collect_cost: true

  - name: MixinHealthCheck
    enabled: true
    config:
      check_interval_ms: 60000
      components:
        - name: "zai_api"
          critical: true
          timeout_seconds: 10.0

# NEW: Advanced features (replaces error_handling)
advanced_features:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout_ms: 60000

  retry_policy:
    enabled: true
    max_attempts: 3
    backoff_multiplier: 2.0
    retryable_status_codes: [429, 500, 502, 503]
```

### C. CLI Examples

**Basic Generation**:
```bash
# Generate minimal node (no mixins)
omninode-generate \
  --contract path/to/contract.yaml \
  --output src/omninode_bridge/nodes/my_node/

# Generate with mixins (default)
omninode-generate \
  --contract path/to/contract.yaml \
  --enable-mixins \
  --output src/omninode_bridge/nodes/my_node/
```

**Advanced Generation**:
```bash
# Generate with LLM + mixins + strict validation
omninode-generate \
  --contract path/to/contract.yaml \
  --enable-mixins \
  --enable-llm \
  --llm-tier CLOUD_FAST \
  --validation-level strict \
  --output src/omninode_bridge/nodes/my_node/ \
  --verbose
```

**Regeneration**:
```bash
# Regenerate existing node with mixins
omninode-generate \
  --contract src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/contract.yaml \
  --enable-mixins \
  --enable-llm \
  --output src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/ \
  --overwrite \
  --backup
```

**Validation Only**:
```bash
# Validate contract without generation
omninode-generate \
  --contract path/to/contract.yaml \
  --validate-only

# Validate generated node
omninode-generate \
  --validate-node src/omninode_bridge/nodes/my_node/v1_0_0/node.py \
  --contract src/omninode_bridge/nodes/my_node/v1_0_0/contract.yaml
```

### D. Testing Strategy

**Unit Tests** (Fast, isolated):
- `test_contract_introspector.py` - Contract parsing with mixins
- `test_mixin_injector.py` - Code generation
- `test_node_validator.py` - Validation stages
- `test_template_rendering.py` - Jinja2 template rendering

**Integration Tests** (Medium, component interaction):
- `test_end_to_end_generation.py` - Full generation pipeline
- `test_mixin_generation_e2e.py` - Mixin-enhanced generation
- `test_validation_pipeline.py` - Multi-stage validation

**Regression Tests** (Slow, full nodes):
- `test_database_adapter_regeneration.py` - Regenerate + compare
- `test_llm_effect_regeneration.py` - Regenerate + compare
- `test_all_nodes_regeneration.py` - Batch regeneration

**Performance Tests** (Benchmarks):
- `test_generation_performance.py` - Generation speed
- `test_validation_performance.py` - Validation speed
- `test_llm_call_performance.py` - LLM call optimization

### E. Rollback Procedures

**If Regeneration Fails**:

```bash
# 1. Stop and assess
# DO NOT commit or deploy

# 2. Restore from backup
rm -rf src/omninode_bridge/nodes/<node_name>/v1_0_0/
mv src/omninode_bridge/nodes/<node_name>.backup/v1_0_0/ \
   src/omninode_bridge/nodes/<node_name>/v1_0_0/

# 3. Verify tests pass
pytest tests/unit/nodes/<node_name>/ -v
pytest tests/integration/nodes/<node_name>/ -v

# 4. Report issue
# Create GitHub issue with:
# - Contract YAML
# - Generation command
# - Error messages
# - Validation results

# 5. Investigate and fix
# - Review contract
# - Check mixin compatibility
# - Fix generator bugs
# - Re-attempt generation
```

**If Tests Fail After Regeneration**:

```bash
# 1. Capture test results
pytest tests/ --json-report --json-report-file=test_failure_report.json

# 2. Compare with baseline
diff baseline_test_results.json test_failure_report.json

# 3. Analyze failures
# - Are failures in generated node?
# - Are failures in tests?
# - Are failures in dependencies?

# 4. Fix or rollback
# If fixable: Update and retest
# If not fixable: Rollback (see above)

# 5. Document lessons learned
# Add to troubleshooting guide
```

---

## Timeline Summary

```
WEEK 1: Discovery & Design
├─ Mixin catalog creation (3 days)
├─ Contract schema design (2 days)
└─ Architecture review (1 day)

WEEK 2: Architecture & Validation
├─ Contract parser enhancement (2 days)
├─ Validation pipeline (2 days)
└─ Integration (1 day)

WEEK 3: Code Generation
├─ Mixin injector (2 days)
├─ Template creation (2 days)
└─ Testing (1 day)

WEEK 4: Testing & Validation
├─ database_adapter regeneration (2 days)
├─ llm_effect regeneration (1 day)
├─ Metrics collection (1 day)
└─ Quality gates (1 day)

WEEK 5: High-Priority Rollout
├─ database_adapter migration (1 day)
├─ llm_effect migration (1 day)
├─ orchestrator migration (1 day)
├─ Documentation (1 day)
└─ Training prep (1 day)

WEEK 6: Medium-Priority Rollout
├─ reducer, registry migration (2 days)
├─ deployment nodes migration (2 days)
└─ Final documentation (1 day)

WEEK 7-8 (OPTIONAL): Polish & Optimization
├─ Low-priority node migration
├─ Performance optimization
├─ Feedback collection
└─ Continuous improvement setup
```

---

## Conclusion

This master plan provides a comprehensive, actionable roadmap for enhancing the omninode_bridge code generator to leverage omnibase_core mixins and produce production-quality nodes.

**Key Deliverables**:
1. ✅ Mixin catalog with 33 documented mixins
2. ✅ Extended contract schema with `mixins` and `advanced_features`
3. ✅ Enhanced generator with mixin injection
4. ✅ Comprehensive validation pipeline
5. ✅ Regenerated nodes with 30-50% LOC reduction
6. ✅ Updated documentation and training materials

**Expected Outcomes**:
- **Code Quality**: Generated nodes match hand-coded quality
- **Developer Productivity**: 50%+ reduction in manual coding
- **Maintainability**: Standardized patterns via mixins
- **Reliability**: Built-in production features (circuit breakers, metrics, health checks)

**Next Steps**:
1. Review and approve this master plan
2. Assign team members to phases
3. Set up project tracking (GitHub project board)
4. Begin Phase 1: Discovery & Cataloging
5. Weekly progress reviews

---

**Document Version**: 1.0
**Last Updated**: 2025-11-04
**Owner**: OmniNode Team
**Status**: 🎯 Ready for Implementation
