# Workflows API Reference

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
**Module**: `omninode_bridge.agents.workflows`

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [Staged Parallel Execution](#staged-parallel-execution)
3. [Template Management](#template-management)
4. [Validation Pipeline](#validation-pipeline)
5. [AI Quorum](#ai-quorum)
6. [Data Models](#data-models)
7. [Exceptions](#exceptions)
8. [Usage Examples](#usage-examples)

---

## Module Overview

The `omninode_bridge.agents.workflows` module provides complete code generation workflow infrastructure with 4 major components:

```python
from omninode_bridge.agents.workflows import (
    # Staged Parallel Execution
    StagedParallelExecutor,
    WorkflowStage,
    WorkflowStep,
    WorkflowConfig,
    WorkflowResult,
    EnumStageStatus,
    EnumStepType,

    # Template Management
    TemplateManager,
    TemplateLRUCache,
    Template,
    TemplateType,
    TemplateMetadata,
    TemplateRenderContext,
    TemplateCacheStats,

    # Validation Pipeline
    ValidationPipeline,
    ValidationResult,
    ValidationContext,
    ValidationSummary,

    # AI Quorum
    AIQuorum,
    LLMClient,
    GeminiClient,
    GLMClient,
    CodestralClient,
    ModelConfig,
    QuorumVote,
    QuorumResult,
)
```

---

## Staged Parallel Execution

### StagedParallelExecutor

Main orchestrator for 6-phase code generation pipeline with parallel step execution.

#### Constructor

```python
class StagedParallelExecutor:
    def __init__(
        self,
        scheduler: DependencyAwareScheduler,
        metrics_collector: Optional[MetricsCollector] = None,
        signal_coordinator: Optional[SignalCoordinator] = None,
        max_parallel_tasks: int = 10,
        stage_timeout_seconds: int = 300,
        enable_fast_fail: bool = True,
    ):
        """
        Initialize staged parallel executor.

        Args:
            scheduler: Dependency-aware scheduler for stage dependencies
            metrics_collector: Optional metrics collector for performance tracking
            signal_coordinator: Optional signal coordinator for lifecycle events
            max_parallel_tasks: Maximum number of parallel tasks per stage (default: 10)
            stage_timeout_seconds: Maximum time per stage in seconds (default: 300)
            enable_fast_fail: Stop execution on first stage failure (default: True)

        Example:
            ```python
            executor = StagedParallelExecutor(
                scheduler=scheduler,
                metrics_collector=metrics,
                signal_coordinator=coordinator,
                max_parallel_tasks=3,
            )
            ```
        """
```

#### Methods

##### execute_workflow()

```python
async def execute_workflow(
    self,
    workflow_id: str,
    stages: list[WorkflowStage],
    context: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> WorkflowResult:
    """
    Execute complete workflow with all stages.

    Args:
        workflow_id: Unique identifier for this workflow execution
        stages: List of workflow stages to execute sequentially
        context: Optional initial context dictionary
        correlation_id: Optional correlation ID for distributed tracing

    Returns:
        WorkflowResult with execution details, timing, and results

    Raises:
        ValueError: If stages are invalid or dependencies cannot be resolved
        TimeoutError: If stage execution exceeds timeout
        RuntimeError: If execution fails and fast_fail is enabled

    Performance:
        - Full pipeline (3 contracts): <5s target, 4.7s actual
        - Stage transition overhead: <100ms
        - Parallelism speedup: 2.25x-4.17x

    Example:
        ```python
        result = await executor.execute_workflow(
            workflow_id="codegen-session-1",
            stages=[stage1, stage2, stage3, stage4, stage5, stage6],
            context={"contract_files": ["c1.yaml", "c2.yaml", "c3.yaml"]},
        )

        if result.success:
            print(f"✅ Completed in {result.total_duration_ms:.2f}ms")
            print(f"   Speedup: {result.overall_speedup:.2f}x")
        else:
            print(f"❌ Failed at stage: {result.failed_stage}")
        ```
    """
```

##### execute_stage()

```python
async def execute_stage(
    self,
    stage: WorkflowStage,
    context: dict[str, Any],
) -> StageResult:
    """
    Execute a single workflow stage with all steps.

    Args:
        stage: Workflow stage to execute
        context: Execution context dictionary

    Returns:
        StageResult with stage execution details

    Raises:
        TimeoutError: If stage execution exceeds timeout
        RuntimeError: If step execution fails

    Example:
        ```python
        stage = WorkflowStage(
            stage_id="models",
            stage_number=2,
            steps=[step1, step2, step3],
        )

        result = await executor.execute_stage(stage, context)
        print(f"Stage duration: {result.duration_ms:.2f}ms")
        print(f"Steps completed: {result.completed_steps}/{result.total_steps}")
        ```
    """
```

##### get_statistics()

```python
def get_statistics(self) -> dict[str, Any]:
    """
    Get workflow execution statistics.

    Returns:
        Dictionary with execution statistics:
        - total_workflows: Total workflows executed
        - total_stages: Total stages executed
        - total_steps: Total steps executed
        - avg_workflow_duration_ms: Average workflow duration
        - avg_stage_duration_ms: Average stage duration
        - avg_speedup: Average parallelism speedup

    Example:
        ```python
        stats = executor.get_statistics()
        print(f"Workflows: {stats['total_workflows']}")
        print(f"Avg duration: {stats['avg_workflow_duration_ms']:.2f}ms")
        print(f"Avg speedup: {stats['avg_speedup']:.2f}x")
        ```
    """
```

---

### WorkflowStage

Represents a single stage in the workflow pipeline.

```python
@dataclass
class WorkflowStage:
    """
    Single stage in workflow pipeline.

    Attributes:
        stage_id: Unique stage identifier
        stage_number: Stage sequence number (1-6)
        steps: List of steps to execute in this stage
        dependencies: List of stage IDs this stage depends on
        timeout_seconds: Maximum execution time for this stage
        allow_partial_failure: Continue if some steps fail
    """
    stage_id: str
    stage_number: int
    steps: list[WorkflowStep]
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    allow_partial_failure: bool = False

    # Example
    stage = WorkflowStage(
        stage_id="models",
        stage_number=2,
        steps=[
            WorkflowStep(step_id="gen_model_1", ...),
            WorkflowStep(step_id="gen_model_2", ...),
            WorkflowStep(step_id="gen_model_3", ...),
        ],
        dependencies=["parse"],  # Depends on stage 1
        timeout_seconds=120,
        allow_partial_failure=False,
    )
```

---

### WorkflowStep

Represents a single step within a stage.

```python
@dataclass
class WorkflowStep:
    """
    Single step within a workflow stage.

    Attributes:
        step_id: Unique step identifier
        step_type: Type of step (SEQUENTIAL or PARALLEL)
        handler: Async function to execute for this step
        context: Step-specific context data
        timeout_seconds: Maximum execution time for this step
        retry_count: Number of retries on failure
    """
    step_id: str
    step_type: EnumStepType
    handler: Callable[[WorkflowStep, dict[str, Any]], Awaitable[dict[str, Any]]]
    context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 60
    retry_count: int = 0

    # Example
    async def my_handler(step: WorkflowStep, context: dict) -> dict:
        # Step implementation
        return {"result": "success"}

    step = WorkflowStep(
        step_id="generate_model_1",
        step_type=EnumStepType.PARALLEL,
        handler=my_handler,
        context={"contract_id": "contract_1"},
        timeout_seconds=30,
    )
```

---

### WorkflowResult

Result of complete workflow execution.

```python
@dataclass
class WorkflowResult:
    """
    Result of workflow execution.

    Attributes:
        workflow_id: Workflow identifier
        success: Whether workflow completed successfully
        stage_results: Results from each stage
        total_duration_ms: Total execution time
        overall_speedup: Parallelism speedup vs sequential
        failed_stage: ID of failed stage (if any)
        errors: List of errors encountered
        generated_files: List of generated file paths
    """
    workflow_id: str
    success: bool
    stage_results: list[StageResult]
    total_duration_ms: float
    overall_speedup: float
    failed_stage: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)

    # Example usage
    if result.success:
        print(f"Generated {len(result.generated_files)} files")
        print(f"Speedup: {result.overall_speedup:.2f}x")
    else:
        print(f"Failed at: {result.failed_stage}")
        for error in result.errors:
            print(f"  - {error}")
```

---

## Template Management

### TemplateManager

High-level template management with LRU caching.

#### Constructor

```python
class TemplateManager:
    def __init__(
        self,
        template_dir: Path | str,
        cache_size: int = 100,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize template manager.

        Args:
            template_dir: Directory containing template files
            cache_size: Maximum number of cached templates (default: 100)
            metrics_collector: Optional metrics collector

        Example:
            ```python
            manager = TemplateManager(
                template_dir="./templates",
                cache_size=100,
                metrics_collector=metrics,
            )
            ```
        """
```

#### Methods

##### start()

```python
async def start(self) -> None:
    """
    Start template manager and initialize cache.

    Example:
        ```python
        await manager.start()
        ```
    """
```

##### load_template()

```python
async def load_template(
    self,
    template_id: str,
    template_type: TemplateType,
    force_reload: bool = False,
) -> Template:
    """
    Load template with LRU caching.

    Args:
        template_id: Unique template identifier
        template_type: Type of template (Effect, Compute, etc.)
        force_reload: Force reload from disk, bypass cache

    Returns:
        Template object with content and metadata

    Performance:
        - Cached lookup: <1ms (target), <2ms actual (with test overhead)
        - Uncached lookup: 10-50ms (file I/O dependent)
        - Cache hit rate: 85-95% target, 80%+ actual

    Example:
        ```python
        # First load: ~25ms (from disk)
        template = await manager.load_template(
            "node_effect_v1",
            TemplateType.EFFECT
        )

        # Second load: <1ms (from cache)
        template = await manager.load_template(
            "node_effect_v1",
            TemplateType.EFFECT
        )
        ```
    """
```

##### render_template()

```python
async def render_template(
    self,
    template_id: str,
    context: dict[str, Any] | TemplateRenderContext,
    template_type: Optional[TemplateType] = None,
) -> str:
    """
    Render template with Jinja2 context.

    Args:
        template_id: Template identifier
        context: Rendering context (dict or TemplateRenderContext)
        template_type: Optional template type (auto-detect if not provided)

    Returns:
        Rendered template as string

    Performance:
        - Rendering time: <10ms average

    Example:
        ```python
        rendered = await manager.render_template(
            "node_effect_v1",
            context={
                "node_name": "MyEffect",
                "description": "My custom effect node",
                "mixins": ["LoggingMixin", "MetricsMixin"],
            },
        )
        ```
    """
```

##### preload_templates()

```python
async def preload_templates(
    self,
    templates: list[tuple[str, TemplateType]]
) -> None:
    """
    Preload templates into cache at startup.

    Args:
        templates: List of (template_id, template_type) tuples

    Example:
        ```python
        await manager.preload_templates([
            ("node_effect_v1", TemplateType.EFFECT),
            ("node_compute_v1", TemplateType.COMPUTE),
            ("node_reducer_v1", TemplateType.REDUCER),
        ])
        ```
    """
```

##### get_cache_stats()

```python
def get_cache_stats(self) -> TemplateCacheStats:
    """
    Get cache statistics.

    Returns:
        TemplateCacheStats with hit rate, size, timing

    Example:
        ```python
        stats = manager.get_cache_stats()
        print(f"Hit rate: {stats.hit_rate:.1%}")
        print(f"Cache size: {stats.current_size}/{stats.max_size}")
        print(f"Avg get time: {stats.avg_get_time_ms:.2f}ms")
        ```
    """
```

---

### Template

Template data model with content and metadata.

```python
@dataclass
class Template:
    """
    Template with content and metadata.

    Attributes:
        template_id: Unique template identifier
        template_type: Type of template
        content: Template content (Jinja2 syntax)
        metadata: Optional metadata
        size_bytes: Template size in bytes
    """
    template_id: str
    template_type: TemplateType
    content: str
    metadata: Optional[TemplateMetadata] = None
    size_bytes: int = 0

    # Example
    template = Template(
        template_id="node_effect_v1",
        template_type=TemplateType.EFFECT,
        content="...",  # Jinja2 template content
        size_bytes=5120,
    )
```

---

### TemplateType

Enum for supported template types.

```python
class TemplateType(str, Enum):
    """
    Supported template types for ONEX v2.0 nodes.
    """
    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"
    MODEL = "model"
    VALIDATOR = "validator"
    TEST = "test"
    CONTRACT = "contract"

    # Example usage
    template_type = TemplateType.EFFECT
    if template_type == TemplateType.EFFECT:
        print("Effect node template")
```

---

## Validation Pipeline

### ValidationPipeline

Multi-stage validation pipeline for generated code.

#### Constructor

```python
class ValidationPipeline:
    def __init__(
        self,
        validators: list[Validator],
        ai_quorum: Optional[AIQuorum] = None,
        enable_fast_fail: bool = False,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize validation pipeline.

        Args:
            validators: List of validator instances
            ai_quorum: Optional AI Quorum for AI-powered validation
            enable_fast_fail: Stop on first validation failure
            metrics_collector: Optional metrics collector

        Example:
            ```python
            pipeline = ValidationPipeline(
                validators=[
                    CompletenessValidator(),
                    QualityValidator(),
                    ONEXComplianceValidator(),
                ],
                ai_quorum=quorum,  # Optional
                enable_fast_fail=False,
            )
            ```
        """
```

#### Methods

##### validate()

```python
async def validate(
    self,
    code: str,
    context: ValidationContext,
) -> ValidationResult:
    """
    Run validation pipeline on generated code.

    Args:
        code: Generated code to validate
        context: Validation context with criteria

    Returns:
        ValidationResult with pass/fail and errors

    Performance:
        - Total validation: <200ms target, <150ms actual (without AI Quorum)
        - With AI Quorum: +2-10s (parallel model calls)

    Example:
        ```python
        result = await pipeline.validate(
            code=generated_code,
            context=ValidationContext(
                node_type="effect",
                contract_summary="Effect node for API calls",
                validation_criteria=[
                    "ONEX v2.0 compliance",
                    "Code quality",
                    "Security",
                ],
            ),
        )

        if result.passed:
            print("✅ Validation passed")
        else:
            for error in result.errors:
                print(f"❌ {error}")
        ```
    """
```

---

### ValidationContext

Context for code validation.

```python
@dataclass
class ValidationContext:
    """
    Context for code validation.

    Attributes:
        node_type: Type of node being validated
        contract_summary: Summary of contract requirements
        validation_criteria: List of validation criteria
    """
    node_type: str
    contract_summary: str
    validation_criteria: list[str] = field(default_factory=list)

    # Example
    context = ValidationContext(
        node_type="effect",
        contract_summary="Effect node for database operations",
        validation_criteria=[
            "ONEX v2.0 compliance",
            "Code quality",
            "Security",
            "Performance",
        ],
    )
```

---

### ValidationResult

Result of validation pipeline execution.

```python
@dataclass
class ValidationResult:
    """
    Result of validation pipeline.

    Attributes:
        passed: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        suggestions: List of fix suggestions
        duration_ms: Validation duration
    """
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    # Example usage
    if not result.passed:
        print("Validation failed:")
        for error in result.errors:
            print(f"  - {error}")
        print("\nSuggestions:")
        for suggestion in result.suggestions:
            print(f"  💡 {suggestion}")
```

---

## AI Quorum

### AIQuorum

4-model consensus validation system.

#### Constructor

```python
class AIQuorum:
    def __init__(
        self,
        model_configs: list[ModelConfig],
        pass_threshold: float = 0.6,
        min_participating_weight: float = 3.0,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize AI Quorum.

        Args:
            model_configs: List of model configurations
            pass_threshold: Consensus threshold (0.0-1.0, default: 0.6)
            min_participating_weight: Minimum total weight required (default: 3.0)
            metrics_collector: Optional metrics collector

        Example:
            ```python
            from omninode_bridge.agents.workflows import DEFAULT_QUORUM_MODELS

            quorum = AIQuorum(
                model_configs=DEFAULT_QUORUM_MODELS,
                pass_threshold=0.6,  # 60% consensus
                min_participating_weight=3.0,
            )
            ```
        """
```

#### Methods

##### register_client()

```python
def register_client(
    self,
    model_id: str,
    client: LLMClient,
) -> None:
    """
    Register LLM client for a model.

    Args:
        model_id: Model identifier (must match ModelConfig.model_id)
        client: LLM client instance

    Example:
        ```python
        quorum.register_client("gemini", GeminiClient(api_key=key))
        quorum.register_client("glm-4.5", GLMClient(api_key=key, model="glm-4.5"))
        ```
    """
```

##### initialize()

```python
async def initialize(self) -> None:
    """
    Initialize all registered LLM clients.

    Example:
        ```python
        await quorum.initialize()
        ```
    """
```

##### validate_code()

```python
async def validate_code(
    self,
    code: str,
    context: ValidationContext,
    correlation_id: Optional[str] = None,
) -> QuorumResult:
    """
    Validate code using 4-model consensus.

    Args:
        code: Generated code to validate
        context: Validation context
        correlation_id: Optional correlation ID

    Returns:
        QuorumResult with consensus decision

    Performance:
        - Fast models (100ms): ~150ms total
        - Standard models (1s): ~1.2s total
        - Slow models (2.5s): ~2.6s total
        - Parallel speedup: 2-4x vs sequential

    Example:
        ```python
        result = await quorum.validate_code(
            code=generated_code,
            context=ValidationContext(
                node_type="effect",
                validation_criteria=["ONEX v2.0 compliance"],
            ),
        )

        if result.passed:
            print(f"✅ Passed with {result.consensus_score:.1%} consensus")
        else:
            print(f"❌ Failed with {result.consensus_score:.1%} consensus")

        # Inspect individual votes
        for vote in result.votes:
            status = "PASS" if vote.vote else "FAIL"
            print(f"  {vote.model_id}: {status} ({vote.confidence:.1%})")
        ```
    """
```

##### get_statistics()

```python
def get_statistics(self) -> dict[str, Any]:
    """
    Get quorum validation statistics.

    Returns:
        Dictionary with statistics:
        - total_validations: Total validations performed
        - pass_rate: Percentage of passed validations
        - avg_consensus_score: Average consensus score
        - avg_duration_ms: Average validation duration
        - avg_cost_per_validation: Average cost

    Example:
        ```python
        stats = quorum.get_statistics()
        print(f"Pass rate: {stats['pass_rate']:.1%}")
        print(f"Avg consensus: {stats['avg_consensus_score']:.1%}")
        print(f"Avg cost: ${stats['avg_cost_per_validation']:.4f}")
        ```
    """
```

---

### ModelConfig

Configuration for an LLM model in the quorum.

```python
@dataclass
class ModelConfig:
    """
    Configuration for LLM model in quorum.

    Attributes:
        model_id: Unique model identifier
        model_name: Model name for API calls
        weight: Model weight (0-10, higher = more important)
        endpoint: API endpoint URL
        api_key_env: Environment variable name for API key
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    """
    model_id: str
    model_name: str
    weight: float
    endpoint: str
    api_key_env: str
    timeout: int = 30
    max_retries: int = 2

    # Example
    config = ModelConfig(
        model_id="gemini",
        model_name="gemini-1.5-pro",
        weight=2.0,  # Highest weight
        endpoint="https://api.google.com/...",
        api_key_env="GEMINI_API_KEY",  # pragma: allowlist secret
        timeout=30,
    )
```

#### DEFAULT_QUORUM_MODELS

Pre-configured model configurations for production use:

```python
DEFAULT_QUORUM_MODELS = [
    ModelConfig(
        model_id="gemini",
        model_name="gemini-1.5-pro",
        weight=2.0,  # Highest weight
        endpoint="...",
        api_key_env="GEMINI_API_KEY",  # pragma: allowlist secret
    ),
    ModelConfig(
        model_id="glm-4.5",
        model_name="glm-4.5",
        weight=2.0,
        endpoint="...",
        api_key_env="GLM_API_KEY",  # pragma: allowlist secret
    ),
    ModelConfig(
        model_id="glm-air",
        model_name="glm-air",
        weight=1.5,
        endpoint="...",
        api_key_env="GLM_API_KEY",  # pragma: allowlist secret
    ),
    ModelConfig(
        model_id="codestral",
        model_name="codestral",
        weight=1.0,
        endpoint="...",
        api_key_env="CODESTRAL_API_KEY",  # pragma: allowlist secret
    ),
]
# Total weight: 6.5
# Pass threshold: 60% (3.9/6.5)
```

---

### QuorumResult

Result of AI Quorum validation.

```python
@dataclass
class QuorumResult:
    """
    Result of AI Quorum validation.

    Attributes:
        passed: Whether code passed quorum validation
        consensus_score: Weighted consensus score (0.0-1.0)
        votes: Individual model votes
        total_weight: Total model weight
        participating_weight: Actual participating weight
        pass_threshold: Required threshold for passing
        duration_ms: Total validation duration
        correlation_id: Optional correlation ID
    """
    passed: bool
    consensus_score: float
    votes: list[QuorumVote]
    total_weight: float
    participating_weight: float
    pass_threshold: float
    duration_ms: float
    correlation_id: Optional[str] = None

    def get_votes_summary(self) -> str:
        """Get formatted summary of votes."""
        pass

    def get_vote_by_model(self, model_id: str) -> Optional[QuorumVote]:
        """Get vote for specific model."""
        pass

    # Example usage
    print(f"Consensus: {result.consensus_score:.1%}")
    print(f"Threshold: {result.pass_threshold:.1%}")
    print(f"Participating: {result.participating_weight}/{result.total_weight}")
    print("\nVotes:")
    for vote in result.votes:
        status = "✅" if vote.vote else "❌"
        print(f"  {status} {vote.model_id}: {vote.confidence:.1%}")
```

---

### QuorumVote

Individual model vote in the quorum.

```python
@dataclass
class QuorumVote:
    """
    Individual model vote in quorum.

    Attributes:
        model_id: Model identifier
        vote: Pass (True) or fail (False)
        confidence: Model confidence (0.0-1.0)
        reasoning: Explanation for vote
        duration_ms: Model call duration
    """
    model_id: str
    vote: bool
    confidence: float
    reasoning: str
    duration_ms: float

    # Example
    vote = QuorumVote(
        model_id="gemini",
        vote=True,  # Pass
        confidence=0.92,
        reasoning="Code meets all ONEX v2.0 requirements",
        duration_ms=1234.5,
    )
```

---

### LLM Clients

#### GeminiClient

Google Gemini 1.5 Pro client.

```python
class GeminiClient(LLMClient):
    def __init__(
        self,
        model_id: str,
        model_name: str,
        api_key: str,
        endpoint: str,
        timeout: int = 30,
    ):
        """Initialize Gemini client."""

    async def validate_code(
        self,
        code: str,
        context: ValidationContext,
    ) -> tuple[bool, float, str]:
        """
        Validate code with Gemini.

        Returns:
            (vote, confidence, reasoning)
        """
```

#### GLMClient

Zhipu AI GLM-4.5 and GLM-Air client.

```python
class GLMClient(LLMClient):
    def __init__(
        self,
        model_id: str,
        model_name: str,  # "glm-4.5" or "glm-air"
        api_key: str,
        endpoint: str,
        timeout: int = 30,
    ):
        """Initialize GLM client."""

    async def validate_code(
        self,
        code: str,
        context: ValidationContext,
    ) -> tuple[bool, float, str]:
        """
        Validate code with GLM.

        Returns:
            (vote, confidence, reasoning)
        """
```

#### CodestralClient

Mistral AI Codestral client.

```python
class CodestralClient(LLMClient):
    def __init__(
        self,
        model_id: str,
        model_name: str,
        api_key: str,
        endpoint: str,
        timeout: int = 30,
    ):
        """Initialize Codestral client."""

    async def validate_code(
        self,
        code: str,
        context: ValidationContext,
    ) -> tuple[bool, float, str]:
        """
        Validate code with Codestral.

        Returns:
            (vote, confidence, reasoning)
        """
```

---

## Exceptions

### WorkflowException

Base exception for workflow errors.

```python
class WorkflowException(Exception):
    """Base exception for workflow errors."""
    pass
```

### StageTimeoutError

Stage execution timeout.

```python
class StageTimeoutError(WorkflowException):
    """Raised when stage execution exceeds timeout."""

    def __init__(self, stage_id: str, timeout_seconds: int):
        super().__init__(
            f"Stage '{stage_id}' exceeded timeout of {timeout_seconds}s"
        )
```

### ValidationException

Validation pipeline error.

```python
class ValidationException(Exception):
    """Base exception for validation errors."""
    pass
```

### QuorumException

AI Quorum error.

```python
class QuorumException(Exception):
    """Base exception for quorum errors."""
    pass

class InsufficientParticipationError(QuorumException):
    """Raised when too many models fail to participate."""

    def __init__(self, participating: float, minimum: float):
        super().__init__(
            f"Insufficient participation: {participating} < {minimum}"
        )
```

---

## Usage Examples

### Complete Workflow Example

```python
from omninode_bridge.agents.workflows import (
    StagedParallelExecutor,
    TemplateManager,
    ValidationPipeline,
    AIQuorum,
    WorkflowStage,
    WorkflowStep,
    EnumStepType,
    DEFAULT_QUORUM_MODELS,
)

# 1. Initialize components
executor = StagedParallelExecutor(
    scheduler=scheduler,
    metrics_collector=metrics,
)

template_manager = TemplateManager(template_dir="./templates")
await template_manager.start()

quorum = AIQuorum(model_configs=DEFAULT_QUORUM_MODELS)
await quorum.initialize()

validation_pipeline = ValidationPipeline(
    validators=[...],
    ai_quorum=quorum,
)

# 2. Define handlers
async def generate_model_handler(step, context):
    template = await template_manager.load_template(...)
    rendered = await template_manager.render_template(...)
    result = await validation_pipeline.validate(...)
    return {"generated_file": "..."}

# 3. Define workflow
stages = [
    WorkflowStage(stage_id="models", stage_number=2, steps=[
        WorkflowStep(
            step_id="gen_1",
            step_type=EnumStepType.PARALLEL,
            handler=generate_model_handler,
        ),
        # More steps...
    ]),
    # More stages...
]

# 4. Execute
result = await executor.execute_workflow(
    workflow_id="session-1",
    stages=stages,
)

print(f"Success: {result.success}")
print(f"Duration: {result.total_duration_ms:.2f}ms")
print(f"Speedup: {result.overall_speedup:.2f}x")
```

### Template Caching Example

```python
manager = TemplateManager(template_dir="./templates", cache_size=100)
await manager.start()

# Preload common templates
await manager.preload_templates([
    ("node_effect_v1", TemplateType.EFFECT),
    ("node_compute_v1", TemplateType.COMPUTE),
])

# First load: ~25ms (from disk)
template1 = await manager.load_template("node_effect_v1", TemplateType.EFFECT)

# Second load: <1ms (from cache)
template2 = await manager.load_template("node_effect_v1", TemplateType.EFFECT)

# Check cache stats
stats = manager.get_cache_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")  # ~87%
```

### AI Quorum Validation Example

```python
quorum = AIQuorum(
    model_configs=DEFAULT_QUORUM_MODELS,
    pass_threshold=0.6,
)

# Register clients
quorum.register_client("gemini", GeminiClient(...))
quorum.register_client("glm-4.5", GLMClient(...))
quorum.register_client("glm-air", GLMClient(...))
quorum.register_client("codestral", CodestralClient(...))

await quorum.initialize()

# Validate code
result = await quorum.validate_code(
    code=generated_code,
    context=ValidationContext(
        node_type="effect",
        validation_criteria=["ONEX v2.0 compliance", "Code quality"],
    ),
)

if result.passed:
    print(f"✅ Passed ({result.consensus_score:.1%} consensus)")
else:
    print(f"❌ Failed ({result.consensus_score:.1%} consensus)")
    for vote in result.votes:
        if not vote.vote:
            print(f"  - {vote.model_id}: {vote.reasoning}")
```

---

## Error Recovery (Weeks 7-8)

### ErrorRecoveryOrchestrator

Centralized error recovery with 5 strategies and 90%+ success rate.

#### Constructor

```python
class ErrorRecoveryOrchestrator:
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        signal_coordinator: Optional[SignalCoordinator] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """
        Initialize error recovery orchestrator.

        Args:
            metrics_collector: Optional metrics collector
            signal_coordinator: Optional signal coordinator
            max_retries: Default maximum retries (default: 3)
            base_delay: Default base delay for exponential backoff (default: 1.0s)
        """
```

#### Methods

##### handle_error()

```python
async def handle_error(
    self,
    context: RecoveryContext,
) -> RecoveryResult:
    """
    Handle error with automatic recovery strategy selection.

    Args:
        context: Recovery context with error details

    Returns:
        RecoveryResult with recovery outcome

    Performance:
        - Error analysis: <100ms
        - Recovery decision: <50ms
        - Total overhead: <500ms target, <300ms actual
        - Success rate: 90%+ for recoverable errors

    Example:
        ```python
        try:
            result = await generate_code(contract)
        except Exception as e:
            context = RecoveryContext(
                workflow_id="codegen-1",
                node_name="generator",
                exception=e,
            )

            recovery_result = await orchestrator.handle_error(context)

            if recovery_result.success:
                result = recovery_result.result
            else:
                raise
        ```
    """
```

##### add_error_pattern()

```python
def add_error_pattern(self, pattern: ErrorPattern) -> None:
    """
    Add custom error pattern for matching.

    Args:
        pattern: ErrorPattern with regex and recovery strategy

    Example:
        ```python
        orchestrator.add_error_pattern(ErrorPattern(
            pattern_id="custom_timeout",
            error_type=ErrorType.TIMEOUT,
            regex_pattern=r"Custom.*timeout",
            recovery_strategy=RecoveryStrategy.RETRY,
            metadata={"max_retries": 5},
        ))
        ```
    """
```

##### get_statistics()

```python
def get_statistics(self) -> RecoveryStatistics:
    """
    Get error recovery statistics.

    Returns:
        RecoveryStatistics with success rates and metrics
    """
```

---

### RecoveryContext

Context for error recovery operations.

```python
@dataclass
class RecoveryContext:
    """
    Context for error recovery.

    Attributes:
        workflow_id: Workflow identifier
        node_name: Node that encountered error
        exception: Original exception
        state: Workflow state at time of error
        step_count: Number of steps executed
        correlation_id: Optional correlation ID
    """
    workflow_id: str
    node_name: str
    exception: Exception
    state: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    correlation_id: Optional[str] = None
```

---

### RecoveryResult

Result of error recovery operation.

```python
@dataclass
class RecoveryResult:
    """
    Result of error recovery.

    Attributes:
        success: Whether recovery succeeded
        strategy_used: Recovery strategy applied
        result: Recovered result (if successful)
        error_message: Error message (if failed)
        duration_ms: Recovery duration
        attempts: Number of attempts made
    """
    success: bool
    strategy_used: RecoveryStrategy
    result: Any = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    attempts: int = 1
```

---

### ErrorPattern

Pattern for error matching and recovery.

```python
@dataclass
class ErrorPattern:
    """
    Error pattern for automatic recovery.

    Attributes:
        pattern_id: Unique pattern identifier
        error_type: Error type (network, validation, etc.)
        regex_pattern: Regex pattern for error message matching
        recovery_strategy: Strategy to use for recovery
        metadata: Strategy-specific configuration
        priority: Pattern matching priority (higher = check first)
    """
    pattern_id: str
    error_type: ErrorType
    regex_pattern: str
    recovery_strategy: RecoveryStrategy
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 50
```

---

## Performance Optimization (Weeks 7-8)

### PerformanceOptimizer

Automatic optimization for 2-3x speedup.

#### Constructor

```python
class PerformanceOptimizer:
    def __init__(
        self,
        profiler: PerformanceProfiler,
        template_manager: Optional[TemplateManager] = None,
        staged_executor: Optional[StagedParallelExecutor] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        auto_optimize: bool = True,
    ):
        """
        Initialize performance optimizer.

        Args:
            profiler: PerformanceProfiler instance
            template_manager: Optional TemplateManager for cache optimization
            staged_executor: Optional StagedParallelExecutor for parallel tuning
            metrics_collector: Optional MetricsCollector
            auto_optimize: Enable automatic optimizations (default: True)
        """
```

#### Methods

##### optimize_workflow()

```python
async def optimize_workflow(
    self,
    workflow_id: str,
    workflow_func: Optional[Callable] = None,
    iterations: int = 5,
    apply_optimizations: bool = None,
    **profile_kwargs,
) -> PerformanceReport:
    """
    Profile and optimize workflow.

    Args:
        workflow_id: Workflow identifier
        workflow_func: Optional workflow function to profile
        iterations: Number of profiling iterations (default: 5)
        apply_optimizations: Apply optimizations (default: auto_optimize)
        **profile_kwargs: Additional profiler kwargs

    Returns:
        PerformanceReport with analysis and recommendations

    Performance:
        - Expected speedup: 2-3x vs baseline
        - Cache hit rate: 95%+ (from 85-95%)
        - Parallel speedup: 3-4x (from 2.25-4.17x)
        - Memory overhead: <50MB

    Example:
        ```python
        report = await optimizer.optimize_workflow(
            workflow_id="session-1",
            workflow_func=my_workflow,
            iterations=5,
            apply_optimizations=True,
        )

        print(f"Estimated speedup: {report.estimated_speedup:.2f}x")
        ```
    """
```

##### optimize_template_cache()

```python
async def optimize_template_cache(
    self,
    target_hit_rate: float = 0.95,
    preload_top_n: int = 20,
) -> None:
    """
    Optimize template cache for target hit rate.

    Args:
        target_hit_rate: Target cache hit rate (default: 0.95)
        preload_top_n: Number of templates to preload (default: 20)

    Impact: 10-15% speedup for template-heavy workflows
    """
```

##### tune_parallel_execution()

```python
def tune_parallel_execution(
    self,
    target_concurrency: int = 12,
    target_speedup: float = 3.5,
) -> None:
    """
    Tune parallel execution parameters.

    Args:
        target_concurrency: Target concurrent tasks (default: 12)
        target_speedup: Target parallelism speedup (default: 3.5x)

    Impact: 2-3x speedup for parallelizable workflows
    """
```

---

### PerformanceProfiler

Performance profiling with <5% overhead.

#### Constructor

```python
class PerformanceProfiler:
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        enable_memory_profiling: bool = False,
        enable_io_profiling: bool = True,
    ):
        """
        Initialize performance profiler.

        Args:
            metrics_collector: Optional MetricsCollector
            enable_memory_profiling: Enable memory profiling (adds ~5% overhead)
            enable_io_profiling: Enable I/O tracking (minimal overhead)
        """
```

#### Methods

##### profile_operation()

```python
@asynccontextmanager
async def profile_operation(
    self,
    operation_name: str,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Context manager for profiling single operation.

    Args:
        operation_name: Operation name
        metadata: Optional metadata

    Performance:
        - Overhead: <5%
        - Timing precision: Microsecond

    Example:
        ```python
        async with profiler.profile_operation("template_render"):
            result = await template_manager.render_template(...)
        ```
    """
```

##### profile_workflow()

```python
async def profile_workflow(
    self,
    workflow_func: Callable,
    iterations: int = 10,
    **kwargs,
) -> ProfileResult:
    """
    Profile workflow execution multiple times.

    Args:
        workflow_func: Workflow function to profile
        iterations: Number of iterations (default: 10)
        **kwargs: Arguments passed to workflow_func

    Returns:
        ProfileResult with timing and resource statistics
    """
```

##### analyze_hot_paths()

```python
def analyze_hot_paths(
    self,
    profile_data: ProfileResult,
    threshold: float = 0.20,
) -> list[HotPath]:
    """
    Identify performance bottlenecks (hot paths).

    Args:
        profile_data: ProfileResult from profile_workflow()
        threshold: Bottleneck threshold (default: 0.20 = 20%)

    Returns:
        List of HotPath objects for operations >threshold of total time

    Example:
        ```python
        hot_paths = profiler.analyze_hot_paths(profile_data)
        for hp in hot_paths:
            print(f"{hp.operation_name}: {hp.bottleneck_score:.1%}")
        ```
    """
```

##### get_timing_stats()

```python
def get_timing_stats(self, operation_name: str) -> TimingStats:
    """
    Get timing statistics for operation.

    Returns:
        TimingStats with p50, p95, p99, min, max, average
    """
```

---

## Performance Models (Weeks 7-8)

### PerformanceReport

```python
@dataclass
class PerformanceReport:
    """
    Performance optimization report.

    Attributes:
        workflow_id: Workflow identifier
        estimated_speedup: Estimated speedup factor
        baseline_time_ms: Baseline execution time
        optimized_time_ms: Estimated optimized time
        recommendations: List of optimization recommendations
        applied_optimizations: List of applied optimizations
        profiling_overhead_ms: Profiling overhead
    """
```

### OptimizationRecommendation

```python
@dataclass
class OptimizationRecommendation:
    """
    Optimization recommendation.

    Attributes:
        area: Optimization area (cache, parallel, memory, io)
        priority: Priority level (HIGH, MEDIUM, LOW)
        description: Human-readable description
        estimated_impact: Estimated performance impact
        confidence: Confidence in estimate (0.0-1.0)
        implementation: Implementation instructions
    """
```

---

**Document Version**: 1.1
**Last Updated**: 2025-11-06 (Phase 4 Weeks 7-8)
**Status**: ✅ Production-Ready

**See Also**:
- [Phase 4 Optimization Guide](../guides/PHASE_4_OPTIMIZATION_GUIDE.md) - Complete optimization system
- [Error Recovery Guide](../guides/ERROR_RECOVERY_GUIDE.md) - Error recovery strategies
- [Production Deployment Guide](../guides/PRODUCTION_DEPLOYMENT_GUIDE.md) - Production deployment
- [Quick Start Guide](../guides/CODE_GENERATION_WORKFLOW_QUICK_START.md)
- [Architecture Guide](../architecture/PHASE_4_WORKFLOWS_ARCHITECTURE.md)
- [Integration Guide](../guides/WORKFLOW_INTEGRATION_GUIDE.md)
- [Performance Tuning](../guides/WORKFLOW_PERFORMANCE_TUNING.md)
