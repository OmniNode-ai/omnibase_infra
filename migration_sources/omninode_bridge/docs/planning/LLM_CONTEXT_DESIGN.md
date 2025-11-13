# LLM Context Design for Enhanced Code Generation

**Version**: 1.0
**Date**: 2025-11-06
**Status**: Implementation Ready

---

## Executive Summary

This document defines the comprehensive LLM context structure for Phase 3 Enhanced Code Generation. The context integrates Phase 2's BusinessLogicGenerator with Phase 3's production patterns, reference implementations, template variants, and mixin recommendations to dramatically improve code generation quality.

**Key Goals:**
- Build rich, multi-source context for LLM generation
- Manage token budget efficiently (≤8K tokens target)
- Prioritize context components intelligently
- Provide production-quality examples and patterns
- Enable fallback mechanisms for reliability

---

## Context Architecture

### Component Structure

```
┌─────────────────────────────────────────────────────────┐
│              LLM Context Components                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. System Context (500 tokens)                          │
│     - ONEX v2.0 guidelines                               │
│     - Node type information                              │
│     - Template variant details                           │
│                                                           │
│  2. Operation Specification (300 tokens)                 │
│     - Method signature                                   │
│     - Input/output models                                │
│     - Description                                        │
│                                                           │
│  3. Production Patterns (2000 tokens) [Phase 3 NEW]      │
│     - Top 5 matched patterns                             │
│     - Code examples                                      │
│     - Usage guidelines                                   │
│                                                           │
│  4. Reference Implementations (2500 tokens) [Phase 3 NEW]│
│     - Similar node implementations                       │
│     - Adaptation guidance                                │
│                                                           │
│  5. Constraints (500 tokens)                             │
│     - Mixins to use                                      │
│     - Error handling requirements                        │
│     - ONEX compliance rules                              │
│                                                           │
│  6. Generation Instructions (200 tokens)                 │
│     - Output format                                      │
│     - Quality requirements                               │
│                                                           │
├─────────────────────────────────────────────────────────┤
│  Total: ~6000 tokens (75% of 8K budget)                 │
│  Buffer: ~2000 tokens (25% for flexibility)             │
└─────────────────────────────────────────────────────────┘
```

---

## 1. System Context (500 tokens)

**Purpose**: Provide foundational ONEX v2.0 and node type context

**Contents:**
- ONEX v2.0 compliance requirements
- Node type information (Effect/Compute/Reducer/Orchestrator)
- Template variant details (from VariantSelector)
- Architecture principles

**Example:**
```markdown
# System Context

## ONEX v2.0 Requirements
You are generating code for an ONEX v2.0 compliant node.

**Core Requirements:**
- Async/await throughout
- ModelOnexError for all error handling
- emit_log_event for structured logging
- Type hints on all functions
- Comprehensive error handling

## Node Type: EFFECT
Effect nodes perform I/O operations (database, API, file system).

**Characteristics:**
- External system interaction
- Network/disk I/O
- Idempotent operations preferred
- Timeout handling required

## Template Variant: DATABASE_HEAVY
Selected variant: DATABASE_HEAVY (confidence: 0.92)
Rationale: Node type EFFECT is supported; 8 operations fits range 3-10;
4 features matched: database, connection_pooling, transactions, error_handling

**Variant Features:**
- Connection pooling (asyncpg)
- Transaction support
- Circuit breaker pattern
- Retry logic with exponential backoff
```

**Token Budget**: 400-600 tokens

---

## 2. Operation Specification (300 tokens)

**Purpose**: Define the specific method to generate

**Contents:**
- Method name and signature
- Input/output models
- Docstring (if available)
- Business description

**Example:**
```markdown
# Operation Specification

## Method: execute_effect
```python
async def execute_effect(
    self,
    contract: ModelContractEffect,
) -> ModelContractResponse:
    """Execute database query effect operation."""
```

**Input Model**: ModelContractEffect
- Contains query parameters and operation config
- Includes timeout and retry settings

**Output Model**: ModelContractResponse
- Returns query results or error
- Includes execution metrics

**Business Purpose**:
Execute database queries with connection pooling, transaction support,
and comprehensive error handling. Must handle connection failures,
timeouts, and query errors gracefully.
```

**Token Budget**: 250-350 tokens

---

## 3. Production Patterns (2000 tokens) [Phase 3 NEW]

**Purpose**: Provide battle-tested production patterns

**Contents:**
- Top 5 patterns from PatternMatcher (sorted by score)
- Code examples from pattern library
- Usage guidelines
- Adaptation instructions

**Selection Criteria:**
- Always include: `error_handling`, `standard_imports`
- High confidence (score >0.7)
- Diverse patterns (avoid redundancy)
- Relevant to operation type

**Example:**
```markdown
# Production Patterns

## Pattern 1: Database Connection Pooling (score: 0.89)
**Category**: Integration
**Tags**: database, connection_pooling, async, resource_management

**Description**:
Implement connection pooling using asyncpg for PostgreSQL operations.

**Code Example**:
```python
class NodeDatabaseEffect:
    def __init__(self, container: ModelContainer):
        self.pool: Optional[asyncpg.Pool] = None
        self.pool_config = {
            "min_size": 10,
            "max_size": 50,
            "timeout": 30.0,
        }

    async def initialize(self) -> None:
        """Initialize connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=os.getenv("POSTGRES_HOST"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DATABASE"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                **self.pool_config,
            )
            emit_log_event(
                level=EnumLogLevel.INFO,
                message="Database connection pool initialized",
                details={"pool_config": self.pool_config},
            )
        except Exception as e:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INITIALIZATION_ERROR,
                message=f"Failed to initialize connection pool: {e}",
            )

    async def cleanup(self) -> None:
        """Cleanup connection pool."""
        if self.pool:
            await self.pool.close()
```

**Usage Guidelines**:
1. Initialize pool in node's `initialize()` method
2. Use `async with self.pool.acquire() as conn:` for queries
3. Handle connection errors with ModelOnexError
4. Close pool in `cleanup()` method
5. Log pool status for observability

---

## Pattern 2: Transaction Management (score: 0.82)
**Category**: Integration
**Tags**: database, transactions, atomic, rollback

**Code Example**:
```python
async def execute_with_transaction(
    self,
    queries: list[str],
) -> list[dict[str, Any]]:
    """Execute multiple queries in a transaction."""
    if not self.pool:
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.INVALID_STATE,
            message="Connection pool not initialized",
        )

    try:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                results = []
                for query in queries:
                    result = await conn.fetch(query)
                    results.append([dict(row) for row in result])

                emit_log_event(
                    level=EnumLogLevel.INFO,
                    message="Transaction completed successfully",
                    details={"queries_executed": len(queries)},
                )
                return results
    except Exception as e:
        emit_log_event(
            level=EnumLogLevel.ERROR,
            message="Transaction failed, rolling back",
            details={"error": str(e)},
        )
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.EXECUTION_ERROR,
            message=f"Transaction execution failed: {e}",
        )
```

**Usage Guidelines**:
1. Use `async with conn.transaction():` for atomic operations
2. Automatic rollback on exceptions
3. Group related queries in single transaction
4. Log transaction start/end for tracing

[... Additional 3 patterns with similar structure ...]
```

**Token Budget**: 1800-2200 tokens (400 tokens per pattern × 5 patterns)

---

## 4. Reference Implementations (2500 tokens) [Phase 3 NEW]

**Purpose**: Provide concrete examples from similar production nodes

**Contents:**
- Implementations from similar nodes (by node type + features)
- Full method examples
- Adaptation guidance
- Common pitfalls to avoid

**Selection Strategy:**
1. Find nodes with matching node_type
2. Filter by feature overlap (>70%)
3. Extract relevant method implementations
4. Provide adaptation instructions

**Example:**
```markdown
# Reference Implementations

## Similar Node: NodePostgresQueryEffect
**Location**: `nodes/postgres_query_effect/v1_0_0/node.py`
**Similarity Score**: 0.87
**Matched Features**: database, connection_pooling, transactions, error_handling

### execute_effect Implementation
```python
async def execute_effect(
    self,
    contract: ModelContractEffect,
) -> ModelContractResponse:
    """
    Execute PostgreSQL query with connection pooling.

    Implements:
    - Connection pool management
    - Query execution with timeout
    - Transaction support
    - Comprehensive error handling
    """
    start_time = time.perf_counter()

    try:
        # Validate contract
        if not contract.input_state:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message="Contract input_state is required",
            )

        # Extract query parameters
        query = contract.input_state.get("query")
        params = contract.input_state.get("params", [])
        use_transaction = contract.input_state.get("use_transaction", False)
        timeout_seconds = contract.input_state.get("timeout", 30.0)

        # Validate inputs
        if not query:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message="Query is required",
            )

        emit_log_event(
            level=EnumLogLevel.INFO,
            message="Executing database query",
            details={
                "correlation_id": contract.correlation_id,
                "use_transaction": use_transaction,
                "timeout": timeout_seconds,
            },
        )

        # Execute query with timeout
        if not self.pool:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_STATE,
                message="Database pool not initialized",
            )

        async with asyncio.timeout(timeout_seconds):
            async with self.pool.acquire() as conn:
                if use_transaction:
                    async with conn.transaction():
                        results = await conn.fetch(query, *params)
                else:
                    results = await conn.fetch(query, *params)

        # Convert results to dict
        output_data = [dict(row) for row in results]

        # Calculate metrics
        latency_ms = (time.perf_counter() - start_time) * 1000

        emit_log_event(
            level=EnumLogLevel.INFO,
            message="Query executed successfully",
            details={
                "correlation_id": contract.correlation_id,
                "rows_returned": len(output_data),
                "latency_ms": latency_ms,
            },
        )

        return ModelContractResponse(
            success=True,
            output_state={"results": output_data, "row_count": len(output_data)},
            metadata={
                "latency_ms": latency_ms,
                "query_executed": True,
            },
        )

    except asyncio.TimeoutError:
        emit_log_event(
            level=EnumLogLevel.ERROR,
            message="Query timeout",
            details={
                "correlation_id": contract.correlation_id,
                "timeout_seconds": timeout_seconds,
            },
        )
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.TIMEOUT_ERROR,
            message=f"Query execution timed out after {timeout_seconds}s",
        )

    except ModelOnexError:
        raise

    except Exception as e:
        emit_log_event(
            level=EnumLogLevel.ERROR,
            message="Query execution failed",
            details={
                "correlation_id": contract.correlation_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.EXECUTION_ERROR,
            message=f"Query execution failed: {e}",
        )
```

### Adaptation Guidance
To adapt this implementation for your node:

1. **Connection Pool**: The pattern assumes asyncpg pool in `self.pool`
   - Initialize in your node's `initialize()` method
   - Use same pool configuration pattern

2. **Input Validation**: Extract parameters from `contract.input_state`
   - Validate all required inputs upfront
   - Use descriptive error messages

3. **Timeout Handling**: Use `asyncio.timeout()` for all I/O operations
   - Get timeout from contract or use default
   - Catch `asyncio.TimeoutError` explicitly

4. **Error Handling**: Three-tier exception handling
   - Specific errors first (TimeoutError)
   - Re-raise ModelOnexError (already formatted)
   - Catch-all for unexpected errors

5. **Logging**: Emit events at key points
   - INFO: Operation start + success
   - ERROR: Failures with context
   - Include correlation_id for tracing

6. **Metrics**: Track and return latency
   - Use `time.perf_counter()` for precision
   - Include in response metadata

### Common Pitfalls
- ❌ Don't forget to check if pool is initialized
- ❌ Don't use `time.time()` for latency (use `perf_counter()`)
- ❌ Don't catch Exception without re-raising as ModelOnexError
- ❌ Don't omit correlation_id from log events
- ❌ Don't forget to convert asyncpg.Record to dict

[... Additional 1-2 similar node examples ...]
```

**Token Budget**: 2200-2800 tokens (1000-1200 tokens per reference × 2-3 references)

---

## 5. Constraints (500 tokens)

**Purpose**: Define requirements and restrictions

**Contents:**
- Required mixins (from MixinRecommender)
- Error handling requirements
- ONEX compliance rules
- Security constraints
- Performance requirements

**Example:**
```markdown
# Constraints

## Required Mixins
Your implementation MUST use these mixins:

1. **MixinConnectionPooling** (confidence: 0.95)
   - Provides: `self.pool`, `initialize()`, `cleanup()`
   - Usage: Use `async with self.pool.acquire() as conn:` for queries

2. **MixinCircuitBreaker** (confidence: 0.82)
   - Provides: Circuit breaker for external calls
   - Usage: Wrap I/O operations with breaker

3. **MixinMetricsCollection** (confidence: 0.78)
   - Provides: Metric tracking utilities
   - Usage: Track operation latency and success rate

## Error Handling Requirements
1. All exceptions MUST be wrapped in ModelOnexError
2. Include correlation_id in all error details
3. Use appropriate error codes:
   - INVALID_INPUT: Input validation failures
   - TIMEOUT_ERROR: Operation timeouts
   - EXECUTION_ERROR: Runtime failures
   - INVALID_STATE: State validation failures

## ONEX Compliance Rules
1. All methods MUST be async
2. Use emit_log_event for all logging (no print/logger)
3. Include type hints on all parameters and returns
4. Add docstrings to all public methods
5. Track and return metrics in response

## Security Constraints
1. NO hardcoded credentials (use environment variables)
2. NO SQL injection vulnerabilities (use parameterized queries)
3. NO sensitive data in logs
4. Validate all inputs before processing

## Performance Requirements
- Database operations: < 100ms (p95)
- Connection acquisition: < 50ms
- Total method execution: < 200ms (p95)
- Log each operation for observability
```

**Token Budget**: 450-550 tokens

---

## 6. Generation Instructions (200 tokens)

**Purpose**: Specify output format and quality requirements

**Contents:**
- Output format specification
- Code structure requirements
- Quality checklist

**Example:**
```markdown
# Generation Instructions

## Output Format
Return ONLY the method body (indented, starting with try/except).

**Structure:**
```python
        # Validate inputs
        if not contract.input_state:
            raise ModelOnexError(...)

        try:
            # Main logic here
            emit_log_event(level=EnumLogLevel.INFO, ...)

            # ... implementation ...

            return ModelContractResponse(success=True, ...)

        except ModelOnexError:
            raise

        except Exception as e:
            emit_log_event(level=EnumLogLevel.ERROR, ...)
            raise ModelOnexError(...)
```

## Quality Requirements
Your implementation MUST:
- ✅ Use async/await throughout
- ✅ Include comprehensive error handling
- ✅ Emit log events (INFO and ERROR)
- ✅ Use type hints for all variables
- ✅ Follow ONEX patterns from examples
- ✅ Use recommended mixins (self.pool, etc.)
- ✅ Track and return metrics
- ✅ NO import statements (already imported)
- ✅ NO hardcoded secrets

## Important Notes
- All imports are ALREADY available in the node file
- Do NOT add import statements in your implementation
- Assume all dependencies are imported at the top of the file
```

**Token Budget**: 180-220 tokens

---

## Token Budget Management

### Prioritization Strategy

When total context exceeds 8K tokens, truncate in this order:

1. **Preserve (NEVER truncate)**:
   - System Context (500 tokens)
   - Operation Specification (300 tokens)
   - Generation Instructions (200 tokens)
   - Critical patterns: error_handling, standard_imports
   - **Total Protected**: ~1200 tokens

2. **Truncate First (if needed)**:
   - Reference Implementations (trim from 2500 → 1500 tokens)
   - Reduce from 3 references to 2, shorten examples

3. **Truncate Second (if needed)**:
   - Production Patterns (trim from 2000 → 1500 tokens)
   - Reduce from 5 patterns to 3, shorten code examples

4. **Truncate Last (if needed)**:
   - Constraints (trim from 500 → 300 tokens)
   - Keep core requirements, reduce explanations

### Token Counting

Use `tiktoken` library with `cl100k_base` encoding (GPT-4/Claude):

```python
import tiktoken

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def truncate_to_budget(
    sections: dict[str, str],
    total_budget: int = 8000,
) -> dict[str, str]:
    """Truncate sections to fit within token budget."""
    # Count tokens for each section
    token_counts = {
        section: count_tokens(text)
        for section, text in sections.items()
    }

    total_tokens = sum(token_counts.values())

    if total_tokens <= total_budget:
        return sections  # No truncation needed

    # Apply truncation strategy
    # 1. Preserve critical sections
    # 2. Truncate reference implementations
    # 3. Truncate production patterns
    # 4. Truncate constraints

    # ... implementation ...

    return truncated_sections
```

### Token Budget Allocation

| Component | Target | Min | Max | Priority |
|-----------|--------|-----|-----|----------|
| System Context | 500 | 400 | 600 | P0 (Never truncate) |
| Operation Spec | 300 | 250 | 350 | P0 (Never truncate) |
| Production Patterns | 2000 | 1000 | 2200 | P2 (Truncate second) |
| Reference Impls | 2500 | 1000 | 2800 | P1 (Truncate first) |
| Constraints | 500 | 300 | 550 | P3 (Truncate last) |
| Instructions | 200 | 180 | 220 | P0 (Never truncate) |
| **Total** | **6000** | **4530** | **6720** | - |
| **Buffer** | **2000** | **1280** | **3470** | - |

---

## Multi-Source Aggregation Strategy

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Input Requirements                      │
│  (ModelPRDRequirements, operation, node_type)           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│           1. Analyze Requirements                        │
│  - Extract features, operations, node type              │
│  - Calculate complexity score                           │
└──────────────────┬──────────────────────────────────────┘
                   │
         ┌─────────┴─────────┬─────────────────┐
         ▼                   ▼                 ▼
┌──────────────────┐  ┌──────────────┐  ┌──────────────┐
│ VariantSelector  │  │PatternMatcher│  │MixinRecommender│
│ Select template  │  │Match patterns│  │Select mixins │
│ variant          │  │(top 5)       │  │(top 3-5)     │
└─────────┬────────┘  └──────┬───────┘  └──────┬───────┘
          │                  │                 │
          └──────────────────┴─────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────┐
          │    2. Find Similar Nodes             │
          │  - Query by node_type + features     │
          │  - Calculate similarity scores       │
          │  - Select top 2-3 nodes              │
          └─────────────────┬────────────────────┘
                            │
                            ▼
          ┌──────────────────────────────────────┐
          │  3. Extract Reference Implementations│
          │  - Read similar node source code     │
          │  - Extract relevant methods          │
          │  - Generate adaptation guidance      │
          └─────────────────┬────────────────────┘
                            │
                            ▼
          ┌──────────────────────────────────────┐
          │      4. Build Complete Context       │
          │  - Aggregate all components          │
          │  - Format with Jinja2 templates      │
          │  - Apply token budget management     │
          │  - Validate completeness             │
          └─────────────────┬────────────────────┘
                            │
                            ▼
          ┌──────────────────────────────────────┐
          │       ModelLLMContext (Output)       │
          │  - Full context for LLM generation   │
          │  - Token count: ~6000 (within budget)│
          │  - All sections populated            │
          └──────────────────────────────────────┘
```

### Component Integration

```python
class EnhancedContextBuilder:
    def __init__(
        self,
        pattern_matcher: PatternMatcher,
        pattern_formatter: PatternFormatter,
        variant_selector: VariantSelector,
        mixin_recommender: MixinRecommender,
    ):
        """Initialize with all Phase 3 components."""
        self.pattern_matcher = pattern_matcher
        self.pattern_formatter = pattern_formatter
        self.variant_selector = variant_selector
        self.mixin_recommender = mixin_recommender

        # Jinja2 template environment
        self.template_env = self._setup_template_env()

    def build_context(
        self,
        requirements: ModelPRDRequirements,
        operation: dict[str, Any],
        node_type: str,
    ) -> ModelLLMContext:
        """
        Build comprehensive LLM context from all sources.

        Returns:
            ModelLLMContext with all sections populated and token budget managed
        """
        # 1. Select template variant
        variant_selection = self.variant_selector.select_variant(
            node_type=EnumNodeType[node_type.upper()],
            operation_count=len(requirements.operations),
            required_features=set(requirements.features),
        )

        # 2. Match production patterns
        pattern_matches = self.pattern_matcher.match_patterns(
            node_type=EnumNodeType[node_type.upper()],
            required_features=set(requirements.features),
            top_k=5,
            min_score=0.3,
        )

        # 3. Recommend mixins
        mixin_recommendations = self.mixin_recommender.recommend_mixins(
            node_type=node_type,
            requirements=requirements,
        )

        # 4. Find similar nodes
        similar_nodes = self._find_similar_nodes(
            node_type=node_type,
            features=set(requirements.features),
        )

        # 5. Extract reference implementations
        reference_impls = self._get_reference_implementations(
            similar_nodes=similar_nodes,
            operation_name=operation["name"],
        )

        # 6. Build context sections
        sections = {
            "system_context": self._build_system_context(
                node_type=node_type,
                variant_selection=variant_selection,
            ),
            "operation_spec": self._build_operation_spec(
                operation=operation,
                requirements=requirements,
            ),
            "production_patterns": self.pattern_formatter.format_patterns_for_llm(
                patterns=pattern_matches,
                max_tokens=2000,
            ),
            "reference_impls": self._format_reference_impls(
                implementations=reference_impls,
                max_tokens=2500,
            ),
            "constraints": self._build_constraints(
                mixins=mixin_recommendations,
                requirements=requirements,
            ),
            "instructions": self._build_instructions(),
        }

        # 7. Apply token budget management
        managed_sections = self._manage_token_budget(
            sections=sections,
            total_budget=8000,
        )

        # 8. Build final context
        return ModelLLMContext(
            system_context=managed_sections["system_context"],
            operation_spec=managed_sections["operation_spec"],
            production_patterns=managed_sections["production_patterns"],
            reference_implementations=managed_sections["reference_impls"],
            constraints=managed_sections["constraints"],
            generation_instructions=managed_sections["instructions"],
            total_tokens=self._count_total_tokens(managed_sections),
            truncation_applied=managed_sections.get("_truncated", False),
        )
```

---

## Context Quality Metrics

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Context building time | <50ms | Time from input to ModelLLMContext |
| Token budget compliance | ≤8K tokens | Total tokens across all sections |
| Pattern inclusion rate | >80% | % of relevant patterns included |
| Reference quality | >0.7 similarity | Similarity score of selected references |
| Truncation rate | <10% | % of contexts requiring truncation |
| Component completeness | 100% | All 6 sections populated |

### Monitoring

```python
class ContextBuildingMetrics:
    """Track context building metrics."""

    build_time_ms: float
    total_tokens: int
    section_tokens: dict[str, int]
    patterns_included: int
    patterns_available: int
    references_included: int
    references_available: int
    truncation_applied: bool
    truncated_sections: list[str]
```

---

## Prompt Template Structure

### Jinja2 Templates

Store templates in `src/omninode_bridge/codegen/llm/prompt_templates/`:

1. **system_context.j2** - System and ONEX guidelines
2. **operation_spec.j2** - Operation specification
3. **pattern_context.j2** - Production patterns with examples
4. **reference_context.j2** - Reference implementations
5. **constraints.j2** - Constraints and requirements
6. **generation_instructions.j2** - Output format and quality requirements

### Template Loading

```python
from jinja2 import Environment, FileSystemLoader

def _setup_template_env(self) -> Environment:
    """Setup Jinja2 environment for prompt templates."""
    template_dir = Path(__file__).parent / "prompt_templates"
    return Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )

def render_section(self, template_name: str, context: dict) -> str:
    """Render a context section using Jinja2 template."""
    template = self.template_env.get_template(template_name)
    return template.render(**context)
```

---

## Examples

### Example 1: Database Effect Node

**Input Requirements:**
- Node Type: EFFECT
- Operations: 8 (query, insert, update, delete, transaction, batch, health, metrics)
- Features: database, connection_pooling, transactions, error_handling, async

**Generated Context:** (6,247 tokens)

**Pattern Matches:**
1. Database Connection Pooling (score: 0.89)
2. Transaction Management (score: 0.82)
3. Error Handling Patterns (score: 0.95)
4. Async I/O Patterns (score: 0.78)
5. Health Check Implementation (score: 0.71)

**Reference Nodes:**
1. NodePostgresQueryEffect (similarity: 0.87)
2. NodeDatabaseBridgeEffect (similarity: 0.79)

**Selected Variant:** DATABASE_HEAVY (confidence: 0.92)

**Recommended Mixins:**
1. MixinConnectionPooling (confidence: 0.95)
2. MixinCircuitBreaker (confidence: 0.82)
3. MixinMetricsCollection (confidence: 0.78)

**Result:**
- All sections populated
- Token budget: 6,247 / 8,000 (78%)
- No truncation required
- Context quality: EXCELLENT

### Example 2: Simple Compute Node

**Input Requirements:**
- Node Type: COMPUTE
- Operations: 2 (transform, validate)
- Features: async, type_hints, error_handling

**Generated Context:** (4,103 tokens)

**Pattern Matches:**
1. Error Handling Patterns (score: 0.95)
2. Standard Imports (score: 0.88)
3. Type Hints Patterns (score: 0.79)

**Reference Nodes:**
1. NodeDataTransformCompute (similarity: 0.82)

**Selected Variant:** MINIMAL (confidence: 0.85)

**Recommended Mixins:**
1. MixinErrorHandling (confidence: 0.92)

**Result:**
- All sections populated
- Token budget: 4,103 / 8,000 (51%)
- No truncation required
- Context quality: GOOD

---

## Implementation Checklist

### Phase 3 Integration Points

- [ ] **VariantSelector**: Template variant selection
- [ ] **PatternMatcher**: Production pattern matching
- [ ] **PatternFormatter**: Format patterns for LLM (NEW)
- [ ] **MixinRecommender**: Mixin recommendations
- [ ] **ReferenceExtractor**: Extract implementations from similar nodes (NEW)
- [ ] **TokenManager**: Token counting and budget management (NEW)

### New Components to Build

- [ ] **EnhancedContextBuilder**: Main context builder class (I3)
- [ ] **PatternFormatter**: Format patterns for LLM consumption (I2)
- [ ] **TokenManager**: Token counting and truncation logic (I3)
- [ ] **ReferenceExtractor**: Extract and format reference implementations (I3)
- [ ] **Jinja2 Templates**: 6 prompt templates (I3)

### Validation Requirements

- [ ] Token counting accuracy (±5% tolerance)
- [ ] Context completeness (all sections populated)
- [ ] Template rendering (no Jinja2 errors)
- [ ] Performance (<50ms context building)
- [ ] Pattern quality (>0.7 average score)
- [ ] Reference similarity (>0.7 average)

---

## Appendix: Data Models

### ModelLLMContext

```python
class ModelLLMContext(BaseModel):
    """Complete LLM context for code generation."""

    system_context: str
    operation_spec: str
    production_patterns: str
    reference_implementations: str
    constraints: str
    generation_instructions: str

    total_tokens: int
    truncation_applied: bool

    # Metadata
    node_type: str
    method_name: str
    variant_selected: str
    patterns_included: int
    references_included: int
```

### ModelContextBuildingResult

```python
class ModelContextBuildingResult(BaseModel):
    """Result of context building operation."""

    context: ModelLLMContext
    metrics: ContextBuildingMetrics
    warnings: list[str]

    # Quality indicators
    quality_score: float  # 0.0-1.0
    completeness: float   # 0.0-1.0
    token_efficiency: float  # tokens_used / tokens_budget
```

---

## Conclusion

This design provides a comprehensive, production-ready approach to building rich LLM contexts that integrate all Phase 3 components. The multi-source aggregation strategy, token budget management, and quality metrics ensure high-quality code generation with fallback mechanisms for reliability.

**Next Steps:**
1. Implement PatternFormatter (I2)
2. Build EnhancedContextBuilder (I3)
3. Create Jinja2 prompt templates (I3)
4. Add ResponseParser and Validator (I4)
5. Implement fallback mechanisms (I5)

---

**Document Status**: Ready for Implementation
**Review Status**: Pending stakeholder review
**Version History**:
- v1.0 (2025-11-06): Initial design document
