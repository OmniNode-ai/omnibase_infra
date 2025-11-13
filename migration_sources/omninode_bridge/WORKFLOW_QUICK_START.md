# Code Generation Workflow - Quick Start Guide

**Quick reference for using the integrated code generation workflow.**

## Installation

```bash
# From omninode_bridge directory
poetry install

# Verify installation
python -c "from omninode_bridge.agents.workflows import CodeGenerationWorkflow; print('✓ Ready')"
```

## Basic Usage (3 Steps)

### 1. Import and Initialize

```python
from omninode_bridge.agents.workflows import CodeGenerationWorkflow

workflow = CodeGenerationWorkflow(
    template_dir="/path/to/templates",  # Optional
    quality_threshold=0.8,               # Validation threshold
    enable_ai_quorum=False,             # Disable for speed
)

await workflow.initialize()
```

### 2. Generate Code

```python
result = await workflow.generate_code(
    contracts=["path/to/contract.yaml"],
    workflow_id="my-session-1",
    output_dir="/path/to/output"  # Optional
)
```

### 3. Check Results and Cleanup

```python
if result.status == "completed":
    print(f"✓ Success! Duration: {result.total_duration_ms:.2f}ms")
else:
    print(f"✗ Failed: {result.failed_stages} stages failed")

await workflow.shutdown()
```

## Complete Example

```python
import asyncio
from omninode_bridge.agents.workflows import CodeGenerationWorkflow

async def main():
    # Initialize
    workflow = CodeGenerationWorkflow()
    await workflow.initialize()

    try:
        # Generate code
        result = await workflow.generate_code(
            contracts=["contract1.yaml", "contract2.yaml"],
            workflow_id="session-1"
        )

        # Show results
        print(f"Status: {result.status}")
        print(f"Duration: {result.total_duration_ms:.2f}ms")
        print(f"Speedup: {result.overall_speedup:.2f}x")
        print(f"Stages: {result.successful_stages}/{result.total_stages}")
        print(f"Steps: {result.successful_steps}/{result.total_steps}")

        # Get statistics
        stats = workflow.get_statistics()
        print(f"\nTemplate hit rate: {stats['template_hit_rate']:.2%}")

    finally:
        await workflow.shutdown()

asyncio.run(main())
```

## Run Example Script

```bash
# Basic usage
python examples/code_generation_workflow_example.py

# With AI quorum (slower, higher quality)
python examples/code_generation_workflow_example.py --enable-quorum

# Performance benchmark
python examples/code_generation_workflow_example.py --benchmark --iterations 10
```

## Configuration Options

### CodeGenerationWorkflow Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `template_dir` | str | None | Template directory path |
| `metrics_collector` | MetricsCollector | None | Metrics collector instance |
| `signal_coordinator` | SignalCoordinator | None | Signal coordinator instance |
| `state` | ThreadSafeState | None | Shared state instance |
| `quality_threshold` | float | 0.8 | Quality validation threshold (0.0-1.0) |
| `enable_ai_quorum` | bool | False | Enable AI quorum validation |
| `quorum_threshold` | float | 0.6 | Quorum consensus threshold (0.0-1.0) |
| `quorum_models` | list | None | Custom model configurations |
| `cache_size` | int | 100 | Template cache size |

### Performance Modes

**Fast Mode** (Default):
```python
workflow = CodeGenerationWorkflow(
    enable_ai_quorum=False,  # Skip AI quorum
    quality_threshold=0.7,   # Lower threshold
)
```
- Duration: ~2-3s per contract
- Quality: Good for development

**Quality Mode**:
```python
workflow = CodeGenerationWorkflow(
    enable_ai_quorum=True,   # Enable AI quorum
    quorum_threshold=0.6,    # 60% consensus
    quality_threshold=0.8,   # Higher threshold
)
```
- Duration: ~10-15s per contract
- Quality: Best for production

## Workflow Stages

The workflow executes 6 stages sequentially, with parallel steps within each stage:

1. **Parse Contracts** (parallel per contract)
   - Parse YAML contract files
   - Extract node metadata

2. **Generate Models** (parallel per contract, uses templates)
   - Load model template
   - Render with contract data
   - Output: Python model code

3. **Generate Validators** (parallel per contract, uses templates)
   - Load validator template
   - Render with contract data
   - Output: Python validator code

4. **Generate Tests** (parallel per contract, uses templates)
   - Load test template
   - Render with contract data
   - Output: Python test code

5. **Validate Code** (parallel per contract, uses pipeline + quorum)
   - Run validation pipeline (3 validators)
   - Optional: Run AI quorum (4 models)
   - Output: Validation results

6. **Package Nodes** (parallel per contract)
   - Package generated code
   - Create deployment artifacts
   - Output: Package metadata

## Statistics and Metrics

### Get Workflow Statistics

```python
stats = workflow.get_statistics()

print(f"Total generations: {stats['generation_count']}")
print(f"Average duration: {stats['avg_duration_ms']:.2f}ms")
print(f"Template hit rate: {stats['template_hit_rate']:.2%}")

# If AI quorum enabled
if 'quorum_validations' in stats:
    print(f"Quorum validations: {stats['quorum_validations']}")
    print(f"Quorum pass rate: {stats['quorum_pass_rate']:.2%}")
```

### Available Statistics

**Workflow Stats**:
- `generation_count` - Total workflow executions
- `total_duration_ms` - Total execution time
- `avg_duration_ms` - Average execution time

**Template Manager Stats**:
- `template_hit_rate` - Cache hit rate (target: 85-95%)
- `template_cache_size` - Current cache size
- `template_avg_load_ms` - Average template load time

**AI Quorum Stats** (if enabled):
- `quorum_validations` - Total quorum validations
- `quorum_pass_rate` - Quorum pass rate

## Troubleshooting

### Import Error

```python
# Error: ModuleNotFoundError
# Solution: Install dependencies
poetry install

# Verify installation
from omninode_bridge.agents.workflows import CodeGenerationWorkflow
```

### Template Not Found

```python
# Error: FileNotFoundError: Template not found
# Solution: Specify template directory
workflow = CodeGenerationWorkflow(
    template_dir="/path/to/templates"
)
```

### Workflow Timeout

```python
# Error: Task timeout
# Solution: Increase timeout in config
config = WorkflowConfig(
    step_timeout_seconds=600.0  # Increase from default 300s
)
```

### Low Template Hit Rate

```python
# Problem: template_hit_rate < 85%
# Solution: Preload templates
await workflow.template_manager.preload_templates([
    ("node_effect_v1", TemplateType.EFFECT),
    ("node_compute_v1", TemplateType.COMPUTE),
    ("node_reducer_v1", TemplateType.REDUCER),
])
```

## Performance Tuning

### Optimize for Speed

```python
workflow = CodeGenerationWorkflow(
    enable_ai_quorum=False,      # Disable quorum
    quality_threshold=0.7,       # Lower threshold
    cache_size=200,              # Larger cache
)

# Preload templates
await workflow.initialize()
await workflow.template_manager.preload_templates([...])
```

### Optimize for Quality

```python
workflow = CodeGenerationWorkflow(
    enable_ai_quorum=True,       # Enable quorum
    quorum_threshold=0.7,        # Higher consensus
    quality_threshold=0.85,      # Higher threshold
)
```

### Optimize for Memory

```python
workflow = CodeGenerationWorkflow(
    cache_size=50,               # Smaller cache
)
```

## Advanced Usage

### Custom Metrics Collector

```python
from omninode_bridge.agents.metrics import MetricsCollector

metrics = MetricsCollector()
workflow = CodeGenerationWorkflow(
    metrics_collector=metrics
)
```

### Custom Signal Coordinator

```python
from omninode_bridge.agents.coordination import SignalCoordinator

coordinator = SignalCoordinator()
workflow = CodeGenerationWorkflow(
    signal_coordinator=coordinator
)
```

### Custom Thread-Safe State

```python
from omninode_bridge.agents.coordination import ThreadSafeState

state = ThreadSafeState()
workflow = CodeGenerationWorkflow(
    state=state
)
```

## Performance Targets

| Metric | Target | How to Check |
|--------|--------|--------------|
| Workflow duration | <5s | `result.total_duration_ms < 5000` |
| Template hit rate | 85-95% | `stats['template_hit_rate'] > 0.85` |
| Validation time | <800ms | Stage 5 duration in results |
| Quorum time | 2-10s | `stats['quorum_duration_ms']` |
| Overall speedup | 2.25x-4.17x | `result.overall_speedup > 2.25` |

## Next Steps

1. **Read Integration Summary**: `PHASE4_WORKFLOW_INTEGRATION_SUMMARY.md`
2. **Run Example**: `python examples/code_generation_workflow_example.py`
3. **Review Architecture**: See summary document for detailed architecture
4. **Create Templates**: Add your own templates to template directory
5. **Test with Real Contracts**: Try with your actual contract files

## Support

- **Documentation**: See `PHASE4_WORKFLOW_INTEGRATION_SUMMARY.md`
- **Example**: `examples/code_generation_workflow_example.py`
- **Architecture**: `docs/architecture/` directory
- **Issues**: Check GitHub issues or create new one
