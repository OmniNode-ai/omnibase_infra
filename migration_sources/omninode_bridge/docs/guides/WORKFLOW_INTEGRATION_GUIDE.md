# Workflow Integration Guide

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
**Reading Time**: 15 minutes

---

## Overview

This guide shows you how to integrate Phase 4 Workflows components into your code generation system. Learn step-by-step integration patterns, configuration options, and best practices.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Integration](#quick-integration)
3. [Component Integration](#component-integration)
4. [Configuration](#configuration)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

**Required**:
- Python 3.11+
- omninode_bridge Phase 4 Weeks 1-4 (Foundation & Coordination)
- PostgreSQL 15+ (for metrics)
- Kafka/Redpanda (for events)

**Optional** (for AI Quorum):
- Gemini API key
- GLM API key
- Codestral API key

```bash
# Install
poetry add omninode_bridge

# Environment variables
export POSTGRES_URL="postgresql://..."
export KAFKA_BOOTSTRAP_SERVERS="192.168.86.200:29092"
export GEMINI_API_KEY="your-key"  # Optional
export GLM_API_KEY="your-key"     # Optional
export CODESTRAL_API_KEY="your-key"  # Optional
```

---

## Quick Integration

### Step 1: Initialize Foundation Components

```python
from omninode_bridge.agents.scheduler import DependencyAwareScheduler
from omninode_bridge.agents.metrics import MetricsCollector
from omninode_bridge.agents.coordination import SignalCoordinator

# Initialize foundation
scheduler = DependencyAwareScheduler()
metrics = MetricsCollector()
coordinator = SignalCoordinator(metrics_collector=metrics)

# Start metrics collector
await metrics.start(
    kafka_bootstrap_servers="192.168.86.200:29092",
    postgres_url="postgresql://postgres:password@192.168.86.200:5436/omninode_bridge"
)
```

### Step 2: Initialize Workflow Components

```python
from omninode_bridge.agents.workflows import (
    StagedParallelExecutor,
    TemplateManager,
    ValidationPipeline,
    AIQuorum,
    DEFAULT_QUORUM_MODELS,
)

# Staged Parallel Executor
executor = StagedParallelExecutor(
    scheduler=scheduler,
    metrics_collector=metrics,
    signal_coordinator=coordinator,
)

# Template Manager
template_manager = TemplateManager(
    template_dir="./templates",
    cache_size=100,
    metrics_collector=metrics,
)
await template_manager.start()

# Validation Pipeline
validation_pipeline = ValidationPipeline(
    validators=[
        CompletenessValidator(),
        QualityValidator(),
        ONEXComplianceValidator(),
    ],
)

# AI Quorum (optional)
quorum = AIQuorum(
    model_configs=DEFAULT_QUORUM_MODELS,
    pass_threshold=0.6,
    metrics_collector=metrics,
)
# Register clients and initialize
# (see AI Quorum section)
```

### Step 3: Define Workflow

```python
from omninode_bridge.agents.workflows import WorkflowStage, WorkflowStep, EnumStepType

stages = [
    WorkflowStage(
        stage_id="models",
        stage_number=2,
        steps=[
            WorkflowStep(
                step_id="gen_model_1",
                step_type=EnumStepType.PARALLEL,
                handler=your_handler_function,
                context={"contract_id": "c1"},
            ),
            # More steps...
        ],
    ),
    # More stages...
]

result = await executor.execute_workflow(
    workflow_id="session-1",
    stages=stages,
)
```

---

## Component Integration

### Staged Parallel Executor Integration

#### Into Existing Code Generation Pipeline

```python
class CodeGenerationPipeline:
    def __init__(self):
        self.executor = StagedParallelExecutor(
            scheduler=scheduler,
            metrics_collector=metrics,
        )

    async def generate_nodes(
        self,
        contracts: list[Contract]
    ) -> GenerationResult:
        """Generate nodes from contracts using staged execution."""

        # Define stages
        stages = self._build_stages(contracts)

        # Execute workflow
        result = await self.executor.execute_workflow(
            workflow_id=f"gen-{uuid.uuid4()}",
            stages=stages,
            context={"contracts": contracts},
        )

        return GenerationResult(
            success=result.success,
            generated_files=result.generated_files,
            duration_ms=result.total_duration_ms,
        )

    def _build_stages(self, contracts):
        """Build 6-phase workflow stages."""
        return [
            self._build_parse_stage(),
            self._build_model_stage(contracts),
            self._build_validator_stage(contracts),
            self._build_test_stage(contracts),
            self._build_quality_stage(),
            self._build_package_stage(),
        ]
```

---

### Template Manager Integration

#### With Existing Template System

```python
class MyTemplateSystem:
    def __init__(self):
        self.template_manager = TemplateManager(
            template_dir="./templates",
            cache_size=100,
        )

    async def start(self):
        """Initialize template manager."""
        await self.template_manager.start()

        # Preload common templates
        await self.template_manager.preload_templates([
            ("node_effect_v1", TemplateType.EFFECT),
            ("node_compute_v1", TemplateType.COMPUTE),
        ])

    async def render_node(
        self,
        node_type: str,
        context: dict
    ) -> str:
        """Render node template with caching."""
        template_id = f"node_{node_type}_v1"

        return await self.template_manager.render_template(
            template_id=template_id,
            context=context,
        )
```

---

### Validation Pipeline Integration

#### As Quality Gate

```python
class QualityGate:
    def __init__(self):
        self.validation_pipeline = ValidationPipeline(
            validators=[
                CompletenessValidator(),
                QualityValidator(min_score=0.7),
                ONEXComplianceValidator(),
            ],
            enable_fast_fail=False,
        )

    async def validate_generated_code(
        self,
        code: str,
        node_type: str
    ) -> bool:
        """Validate generated code before acceptance."""

        result = await self.validation_pipeline.validate(
            code=code,
            context=ValidationContext(
                node_type=node_type,
                validation_criteria=[
                    "ONEX v2.0 compliance",
                    "Code quality",
                ],
            ),
        )

        if not result.passed:
            logger.warning(f"Validation failed: {result.errors}")

        return result.passed
```

---

### AI Quorum Integration

#### As Optional Quality Enhancement

```python
class EnhancedQualityGate:
    def __init__(self, enable_ai_quorum: bool = False):
        # Setup quorum if enabled
        self.quorum = None
        if enable_ai_quorum:
            self.quorum = self._setup_quorum()

        self.validation_pipeline = ValidationPipeline(
            validators=[...],
            ai_quorum=self.quorum,  # Optional
        )

    def _setup_quorum(self) -> AIQuorum:
        """Setup AI Quorum with API keys."""
        quorum = AIQuorum(
            model_configs=DEFAULT_QUORUM_MODELS,
            pass_threshold=0.6,
        )

        # Load API keys from environment
        gemini_key = os.getenv("GEMINI_API_KEY")
        glm_key = os.getenv("GLM_API_KEY")
        codestral_key = os.getenv("CODESTRAL_API_KEY")

        # Register clients
        if gemini_key:
            quorum.register_client("gemini", GeminiClient(...))
        if glm_key:
            quorum.register_client("glm-4.5", GLMClient(...))
            quorum.register_client("glm-air", GLMClient(...))
        if codestral_key:
            quorum.register_client("codestral", CodestralClient(...))

        return quorum

    async def validate(self, code: str) -> bool:
        """Validate with optional AI Quorum."""
        result = await self.validation_pipeline.validate(code, ...)
        return result.passed
```

---

## Configuration

### Production Configuration

```python
# config/production.py
WORKFLOW_CONFIG = {
    "staged_executor": {
        "max_parallel_tasks": 10,
        "stage_timeout_seconds": 300,
        "enable_fast_fail": True,
    },
    "template_manager": {
        "cache_size": 200,  # Larger cache for production
        "template_dir": "/opt/templates",
    },
    "validation_pipeline": {
        "enable_fast_fail": False,  # Collect all errors
        "quality_threshold": 0.8,
    },
    "ai_quorum": {
        "enabled": True,
        "pass_threshold": 0.7,  # Stricter in production
        "min_participating_weight": 4.0,
    },
}
```

### Development Configuration

```python
# config/development.py
WORKFLOW_CONFIG = {
    "staged_executor": {
        "max_parallel_tasks": 3,
        "stage_timeout_seconds": 60,
        "enable_fast_fail": True,
    },
    "template_manager": {
        "cache_size": 50,
        "template_dir": "./templates",
    },
    "validation_pipeline": {
        "enable_fast_fail": True,  # Faster iteration
        "quality_threshold": 0.6,
    },
    "ai_quorum": {
        "enabled": False,  # Skip expensive AI validation
    },
}
```

---

## Best Practices

### 1. Cache Template Preloading

```python
# Preload at startup for better performance
await template_manager.preload_templates([
    ("node_effect_v1", TemplateType.EFFECT),
    ("node_compute_v1", TemplateType.COMPUTE),
    ("node_reducer_v1", TemplateType.REDUCER),
    ("node_orchestrator_v1", TemplateType.ORCHESTRATOR),
])
```

### 2. Metrics Monitoring

```python
# Monitor workflow performance
stats = executor.get_statistics()
logger.info(f"Avg workflow time: {stats['avg_workflow_duration_ms']:.2f}ms")
logger.info(f"Avg speedup: {stats['avg_speedup']:.2f}x")

# Monitor cache performance
cache_stats = template_manager.get_cache_stats()
if cache_stats.hit_rate < 0.80:
    logger.warning(f"Low cache hit rate: {cache_stats.hit_rate:.1%}")
```

### 3. Error Handling

```python
try:
    result = await executor.execute_workflow(...)
except StageTimeoutError as e:
    logger.error(f"Stage timeout: {e}")
    # Handle timeout
except ValidationException as e:
    logger.error(f"Validation failed: {e}")
    # Handle validation failure
except QuorumException as e:
    logger.error(f"Quorum failed: {e}")
    # Handle quorum failure
```

### 4. Resource Management

```python
# Cleanup on shutdown
async def shutdown():
    await template_manager.stop()
    await metrics.stop()
    if quorum:
        await quorum.close()
```

---

## Troubleshooting

### Issue: Low Template Cache Hit Rate

**Symptoms**: Cache hit rate <70%

**Solutions**:
1. Increase cache size
2. Preload common templates
3. Verify template IDs are consistent

### Issue: High AI Quorum Latency

**Symptoms**: Validation taking >15s

**Solutions**:
1. Disable AI Quorum for development
2. Reduce timeout per model
3. Check API rate limits

### Issue: Workflow Speedup <2x

**Symptoms**: Parallel execution not much faster

**Solutions**:
1. Verify steps are marked PARALLEL
2. Increase max_parallel_tasks
3. Check CPU/memory limits

---

**See Also**:
- [Quick Start Guide](./CODE_GENERATION_WORKFLOW_QUICK_START.md)
- [API Reference](../api/WORKFLOWS_API_REFERENCE.md)
- [Performance Tuning](./WORKFLOW_PERFORMANCE_TUNING.md)
