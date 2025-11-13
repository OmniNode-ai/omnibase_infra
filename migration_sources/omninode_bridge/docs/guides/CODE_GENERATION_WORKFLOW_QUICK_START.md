# Code Generation Workflow - Quick Start Guide

**Version**: 1.0
**Status**: ✅ Production-Ready
**Last Updated**: 2025-11-06
**Reading Time**: 5 minutes

---

## Overview

Get started with the Phase 4 Workflows system in under 5 minutes. This guide shows you how to generate ONEX v2.0 nodes using the complete code generation pipeline.

**What You'll Learn**:
- How to set up the workflow components
- How to generate code from contracts
- How to validate generated code
- Common patterns and troubleshooting

---

## Prerequisites

```bash
# Required
Python 3.11+
poetry install

# Optional (for AI Quorum)
export GEMINI_API_KEY="your-key"
export GLM_API_KEY="your-key"
export CODESTRAL_API_KEY="your-key"
```

---

## Quick Example: Generate a Node in 3 Steps

### Step 1: Setup Components

```python
from omninode_bridge.agents.workflows import (
    StagedParallelExecutor,
    TemplateManager,
    ValidationPipeline,
    AIQuorum,
    DEFAULT_QUORUM_MODELS,
)
from omninode_bridge.agents.scheduler import DependencyAwareScheduler
from omninode_bridge.agents.metrics import MetricsCollector
from omninode_bridge.agents.coordination import SignalCoordinator

# Initialize foundation components
scheduler = DependencyAwareScheduler()
metrics = MetricsCollector()
coordinator = SignalCoordinator(metrics_collector=metrics)

await metrics.start(
    kafka_bootstrap_servers="192.168.86.200:29092",
    postgres_url="postgresql://postgres:password@192.168.86.200:5436/omninode_bridge"
)

# Initialize workflow components
executor = StagedParallelExecutor(
    scheduler=scheduler,
    metrics_collector=metrics,
    signal_coordinator=coordinator
)

template_manager = TemplateManager(
    template_dir="./templates",
    cache_size=100
)
await template_manager.start()

validation_pipeline = ValidationPipeline(
    validators=[...]  # See below for validator setup
)
```

### Step 2: Define Workflow Stages

```python
from omninode_bridge.agents.workflows import (
    WorkflowStage,
    WorkflowStep,
    EnumStepType,
)

# Define 6-phase pipeline
stages = [
    # Stage 1: Parse contracts
    WorkflowStage(
        stage_id="parse",
        stage_number=1,
        steps=[
            WorkflowStep(
                step_id="parse_contracts",
                step_type=EnumStepType.SEQUENTIAL,
                handler=parse_contracts_handler,
            )
        ],
    ),

    # Stage 2: Generate models (parallel)
    WorkflowStage(
        stage_id="models",
        stage_number=2,
        dependencies=["parse"],  # Depends on Stage 1
        steps=[
            WorkflowStep(
                step_id="generate_model_1",
                step_type=EnumStepType.PARALLEL,
                handler=generate_model_handler,
                context={"contract_id": "contract_1"},
            ),
            WorkflowStep(
                step_id="generate_model_2",
                step_type=EnumStepType.PARALLEL,
                handler=generate_model_handler,
                context={"contract_id": "contract_2"},
            ),
            WorkflowStep(
                step_id="generate_model_3",
                step_type=EnumStepType.PARALLEL,
                handler=generate_model_handler,
                context={"contract_id": "contract_3"},
            ),
        ],
    ),

    # Stage 3: Generate validators (parallel)
    WorkflowStage(
        stage_id="validators",
        stage_number=3,
        dependencies=["models"],
        steps=[...],  # Similar to Stage 2
    ),

    # Stage 4: Generate tests (parallel)
    WorkflowStage(
        stage_id="tests",
        stage_number=4,
        dependencies=["validators"],
        steps=[...],  # Similar to Stage 2
    ),

    # Stage 5: Quality validation (sequential)
    WorkflowStage(
        stage_id="quality",
        stage_number=5,
        dependencies=["tests"],
        steps=[
            WorkflowStep(
                step_id="validate_quality",
                step_type=EnumStepType.SEQUENTIAL,
                handler=validate_quality_handler,
            ),
        ],
    ),

    # Stage 6: Package (sequential)
    WorkflowStage(
        stage_id="package",
        stage_number=6,
        dependencies=["quality"],
        steps=[
            WorkflowStep(
                step_id="package_code",
                step_type=EnumStepType.SEQUENTIAL,
                handler=package_handler,
            ),
        ],
    ),
]
```

### Step 3: Execute Workflow

```python
# Execute complete pipeline
result = await executor.execute_workflow(
    workflow_id="codegen-session-1",
    stages=stages
)

# Check results
if result.success:
    print(f"✅ Code generation complete!")
    print(f"   Duration: {result.total_duration_ms:.2f}ms")
    print(f"   Speedup: {result.overall_speedup:.2f}x")
    print(f"   Files generated: {len(result.generated_files)}")
else:
    print(f"❌ Code generation failed at stage {result.failed_stage}")
    for error in result.errors:
        print(f"   - {error}")
```

---

## Step-by-Step Handler Examples

### Parse Contracts Handler

```python
async def parse_contracts_handler(
    step: WorkflowStep,
    context: dict
) -> dict:
    """Parse YAML contracts into structured objects."""
    import yaml

    contracts = []
    contract_files = context.get("contract_files", [])

    for file_path in contract_files:
        with open(file_path, "r") as f:
            contract_data = yaml.safe_load(f)
            contracts.append(contract_data)

    return {"parsed_contracts": contracts}
```

### Generate Model Handler

```python
async def generate_model_handler(
    step: WorkflowStep,
    context: dict
) -> dict:
    """Generate model code from parsed contract."""

    # Get contract from context
    contract_id = context["contract_id"]
    parsed_contracts = context["parsed_contracts"]
    contract = next(c for c in parsed_contracts if c["id"] == contract_id)

    # Load and render template
    template = await template_manager.load_template(
        template_id="node_effect_v1",
        template_type=TemplateType.EFFECT
    )

    rendered_code = await template_manager.render_template(
        template_id="node_effect_v1",
        context={
            "node_name": contract["name"],
            "description": contract["description"],
            "mixins": contract.get("mixins", []),
        }
    )

    # Validate generated code
    validation_result = await validation_pipeline.validate(
        code=rendered_code,
        context=ValidationContext(
            node_type="effect",
            contract_summary=contract["description"],
        )
    )

    if not validation_result.passed:
        raise ValueError(f"Validation failed: {validation_result.errors}")

    # Write file
    output_path = f"./generated/{contract['name']}.py"
    with open(output_path, "w") as f:
        f.write(rendered_code)

    return {
        "generated_file": output_path,
        "validation_passed": True,
    }
```

### Validate Quality Handler

```python
async def validate_quality_handler(
    step: WorkflowStep,
    context: dict
) -> dict:
    """Validate all generated files with AI Quorum."""

    generated_files = context.get("generated_files", [])
    validation_results = []

    for file_path in generated_files:
        with open(file_path, "r") as f:
            code = f.read()

        # Run validation pipeline
        pipeline_result = await validation_pipeline.validate(
            code=code,
            context=ValidationContext(node_type="effect")
        )

        # Optional: AI Quorum validation
        if ai_quorum_enabled:
            quorum_result = await quorum.validate_code(
                code=code,
                context=ValidationContext(
                    node_type="effect",
                    validation_criteria=[
                        "ONEX v2.0 compliance",
                        "Code quality",
                        "Security",
                    ],
                )
            )

            if not quorum_result.passed:
                raise ValueError(
                    f"AI Quorum validation failed for {file_path}: "
                    f"consensus {quorum_result.consensus_score:.2f}"
                )

        validation_results.append({
            "file": file_path,
            "passed": pipeline_result.passed,
        })

    return {"validation_results": validation_results}
```

---

## Common Patterns

### Pattern 1: Template with Caching

```python
# First load: ~25ms (from disk)
template1 = await template_manager.load_template("node_effect_v1", TemplateType.EFFECT)

# Subsequent loads: <1ms (from cache)
template2 = await template_manager.load_template("node_effect_v1", TemplateType.EFFECT)
template3 = await template_manager.load_template("node_effect_v1", TemplateType.EFFECT)

# Check cache statistics
stats = template_manager.get_cache_stats()
print(f"Cache hit rate: {stats.hit_rate:.1%}")  # ~87%
```

### Pattern 2: Parallel Contract Processing

```python
# Process 3 contracts in parallel (Stage 2-4)
contracts = ["contract_1", "contract_2", "contract_3"]

steps = [
    WorkflowStep(
        step_id=f"generate_{contract}",
        step_type=EnumStepType.PARALLEL,
        handler=generate_model_handler,
        context={"contract_id": contract},
    )
    for contract in contracts
]

stage = WorkflowStage(
    stage_id="models",
    stage_number=2,
    steps=steps,  # All execute in parallel
)
```

### Pattern 3: Optional AI Quorum

```python
# Development: Skip AI Quorum (fast)
pipeline_dev = ValidationPipeline(
    validators=[CompletenessValidator(), QualityValidator()],
    ai_quorum=None,  # No AI validation
)

# Production: Enable AI Quorum (thorough)
quorum = AIQuorum(
    model_configs=DEFAULT_QUORUM_MODELS,
    pass_threshold=0.6,
)
await quorum.initialize()

pipeline_prod = ValidationPipeline(
    validators=[CompletenessValidator(), QualityValidator()],
    ai_quorum=quorum,  # AI-powered validation
)
```

### Pattern 4: Fast-Fail vs Collect-All Errors

```python
# Fast-fail: Stop on first error
executor_fast = StagedParallelExecutor(
    scheduler=scheduler,
    enable_fast_fail=True,  # Stop on first stage failure
)

# Collect all: Continue to collect all errors
executor_all = StagedParallelExecutor(
    scheduler=scheduler,
    enable_fast_fail=False,  # Collect all errors
)
```

---

## Validation Setup

### Basic Validation Pipeline

```python
from omninode_bridge.agents.workflows import ValidationPipeline

pipeline = ValidationPipeline(
    validators=[
        CompletenessValidator(),     # <50ms
        QualityValidator(),           # <100ms
        ONEXComplianceValidator(),    # <100ms
    ],
    enable_fast_fail=False,  # Collect all errors
)

result = await pipeline.validate(code, context)
```

### With AI Quorum

```python
from omninode_bridge.agents.workflows import AIQuorum, DEFAULT_QUORUM_MODELS
from omninode_bridge.agents.workflows.llm_client import (
    GeminiClient,
    GLMClient,
    CodestralClient,
)

# Create quorum
quorum = AIQuorum(
    model_configs=DEFAULT_QUORUM_MODELS,
    pass_threshold=0.6,  # 60% consensus required
)

# Register LLM clients
quorum.register_client("gemini", GeminiClient(api_key=gemini_key))
quorum.register_client("glm-4.5", GLMClient(api_key=glm_key, model="glm-4.5"))
quorum.register_client("glm-air", GLMClient(api_key=glm_key, model="glm-air"))
quorum.register_client("codestral", CodestralClient(api_key=codestral_key))

await quorum.initialize()

# Add to pipeline
pipeline = ValidationPipeline(
    validators=[...],
    ai_quorum=quorum,
)
```

---

## Performance Tips

### 1. Preload Templates at Startup

```python
# Preload common templates to warm cache
await template_manager.preload_templates([
    ("node_effect_v1", TemplateType.EFFECT),
    ("node_compute_v1", TemplateType.COMPUTE),
    ("node_reducer_v1", TemplateType.REDUCER),
])

# Now all subsequent loads are <1ms from cache
```

### 2. Optimize Parallel Execution

```python
# Process contracts in batches to avoid overwhelming CPU
contracts = [...] # 12 contracts

# Split into batches of 3
batches = [contracts[i:i+3] for i in range(0, len(contracts), 3)]

for batch in batches:
    steps = [
        WorkflowStep(..., context={"contract_id": c})
        for c in batch
    ]
    stage = WorkflowStage(steps=steps)
    await executor.execute_stage(stage)
```

### 3. Disable AI Quorum for Development

```python
# Development: Fast iteration
if environment == "development":
    ai_quorum = None  # Skip expensive AI validation

# Production: Thorough validation
else:
    ai_quorum = AIQuorum(...)
```

---

## Troubleshooting

### Issue: "Template not found"

**Cause**: Template file missing or incorrect path

**Solution**:
```python
# Verify template directory
import os
print(os.listdir(template_dir))

# Check template ID matches filename
# Template ID: "node_effect_v1"
# Expected file: "./templates/node_effect_v1.jinja2"
```

### Issue: "Cache hit rate <50%"

**Cause**: Cache size too small or templates not reused

**Solution**:
```python
# Increase cache size
manager = TemplateManager(cache_size=200)  # Default: 100

# Preload common templates
await manager.preload_templates([...])

# Check cache stats
stats = manager.get_cache_stats()
print(f"Current size: {stats.current_size}/{stats.max_size}")
print(f"Hit rate: {stats.hit_rate:.1%}")
```

### Issue: "AI Quorum timeout"

**Cause**: Models taking too long to respond

**Solution**:
```python
# Increase model timeout
from omninode_bridge.agents.workflows import ModelConfig

config = ModelConfig(
    model_id="gemini",
    timeout=60,  # Increase from 30s to 60s
    max_retries=3,
)

quorum = AIQuorum(
    model_configs=[config, ...],
)
```

### Issue: "Workflow too slow (>10s)"

**Cause**: Sequential execution or AI Quorum enabled

**Solution**:
```python
# Check parallelism
result = await executor.execute_workflow(...)
print(f"Speedup: {result.overall_speedup:.2f}x")  # Should be 2-4x

# If speedup is low, check step types
for stage in stages:
    for step in stage.steps:
        print(f"{step.step_id}: {step.step_type}")
        # Should be PARALLEL for Stages 2-4

# Disable AI Quorum for development
pipeline = ValidationPipeline(ai_quorum=None)
```

---

## Next Steps

### Learn More

- **[Architecture Guide](../architecture/PHASE_4_WORKFLOWS_ARCHITECTURE.md)** - Deep dive into system design
- **[API Reference](../api/WORKFLOWS_API_REFERENCE.md)** - Complete API documentation
- **[Integration Guide](./WORKFLOW_INTEGRATION_GUIDE.md)** - Integration patterns
- **[Performance Tuning](./WORKFLOW_PERFORMANCE_TUNING.md)** - Optimization guide

### Example Projects

- **Basic Node Generation**: `examples/basic_node_generation.py`
- **Batch Processing**: `examples/batch_contract_processing.py`
- **Custom Validation**: `examples/custom_validation_pipeline.py`
- **AI Quorum Integration**: `examples/ai_quorum_validation.py`

### Community

- **GitHub**: [OmniNode Bridge Repository](https://github.com/OmniNode-ai/omninode_bridge)
- **Documentation**: [Complete Documentation Index](../INDEX.md)
- **Issues**: [Report Issues](https://github.com/OmniNode-ai/omninode_bridge/issues)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-06
**Status**: ✅ Production-Ready
