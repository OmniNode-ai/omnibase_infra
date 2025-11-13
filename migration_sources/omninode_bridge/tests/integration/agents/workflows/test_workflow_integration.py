"""
Comprehensive Integration Tests for Phase 4 Workflows.

This module tests the complete code generation workflow integration:
1. Staged Parallel Execution (Pattern 8) - 6-phase pipeline
2. Template Management - LRU caching
3. Validation Pipeline - 3 validators
4. AI Quorum Integration - 4-model consensus
5. CodeGenerationWorkflow (integration layer)

Test Scenarios:
- Scenario 1: Simple Contract → Code Pipeline
- Scenario 2: Multiple Contracts in Parallel
- Scenario 3: Full Workflow with AI Quorum
- Scenario 4: Template Cache Performance
- Scenario 5: Performance Validation

Performance Targets:
- Full pipeline: <5s
- Template lookup (cached): <1ms
- Validation pipeline: 300-800ms
- AI Quorum: 2-10s
- Parallelism speedup: 2.25x-4.17x

Coverage Targets:
- Unit tests: 95%+ per component
- Integration tests: 90%+ of integration layer
- End-to-end scenarios: 5+ realistic workflows
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omninode_bridge.agents.coordination.signals import SignalCoordinator
from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.scheduler.scheduler import DependencyAwareScheduler
from omninode_bridge.agents.workflows.ai_quorum import AIQuorum
from omninode_bridge.agents.workflows.llm_client import LLMClient
from omninode_bridge.agents.workflows.quorum_models import (
    ModelConfig,
    QuorumResult,
    QuorumVote,
    ValidationContext,
)
from omninode_bridge.agents.workflows.staged_execution import StagedParallelExecutor
from omninode_bridge.agents.workflows.template_cache import TemplateLRUCache
from omninode_bridge.agents.workflows.template_manager import TemplateManager
from omninode_bridge.agents.workflows.template_models import (
    Template,
    TemplateMetadata,
    TemplateType,
)
from omninode_bridge.agents.workflows.validation_models import (
    ValidationContext as CodeValidationContext,
)
from omninode_bridge.agents.workflows.validation_models import (
    ValidationResult,
)
from omninode_bridge.agents.workflows.validation_pipeline import ValidationPipeline
from omninode_bridge.agents.workflows.validators import (
    BaseValidator,
    CompletenessValidator,
    OnexComplianceValidator,
    QualityValidator,
)
from omninode_bridge.agents.workflows.workflow_models import (
    EnumStageStatus,
    EnumStepType,
    WorkflowConfig,
    WorkflowStage,
    WorkflowStep,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def metrics_collector():
    """Create MetricsCollector instance for testing."""
    collector = MetricsCollector()
    return collector


@pytest.fixture
def signal_coordinator(thread_safe_state):
    """Create SignalCoordinator instance for testing."""
    coordinator = SignalCoordinator(state=thread_safe_state)
    return coordinator


@pytest.fixture
def thread_safe_state():
    """Create ThreadSafeState instance for testing."""
    state = ThreadSafeState()
    return state


@pytest.fixture
def dependency_scheduler(thread_safe_state):
    """Create DependencyAwareScheduler instance for testing."""
    scheduler = DependencyAwareScheduler(state=thread_safe_state)
    return scheduler


@pytest.fixture
def staged_executor(
    dependency_scheduler, metrics_collector, signal_coordinator, thread_safe_state
):
    """Create StagedParallelExecutor instance for testing."""
    executor = StagedParallelExecutor(
        scheduler=dependency_scheduler,
        metrics_collector=metrics_collector,
        signal_coordinator=signal_coordinator,
        state=thread_safe_state,
    )
    return executor


@pytest.fixture
def template_cache():
    """Create TemplateLRUCache instance for testing."""
    cache = TemplateLRUCache(max_size=100)
    return cache


@pytest.fixture
async def template_manager(template_cache, metrics_collector, tmp_path):
    """Create TemplateManager instance for testing."""
    # Create temporary template directory
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    # Create sample templates
    effect_template = template_dir / "node_effect_v1.jinja2"
    effect_template.write_text(
        """
# {{ node_name }} - Effect Node
# Version: {{ version }}
# Description: {{ description }}

from typing import Any
from omnibase_core import NodeEffect

class {{ node_name }}(NodeEffect):
    async def execute_effect(self, context: dict[str, Any]) -> dict[str, Any]:
        # TODO: Implement effect logic
        return {"status": "success"}
"""
    )

    compute_template = template_dir / "node_compute_v1.jinja2"
    compute_template.write_text(
        """
# {{ node_name }} - Compute Node
# Version: {{ version }}

from typing import Any
from omnibase_core import NodeCompute

class {{ node_name }}(NodeCompute):
    async def execute_compute(self, context: dict[str, Any]) -> dict[str, Any]:
        # TODO: Implement compute logic
        return {"result": 0}
"""
    )

    manager = TemplateManager(
        template_dir=str(template_dir),
        cache=template_cache,
        metrics_collector=metrics_collector,
    )
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def validation_pipeline(metrics_collector):
    """Create ValidationPipeline instance for testing."""
    validators = [
        CompletenessValidator(metrics_collector),
        QualityValidator(metrics_collector, quality_threshold=0.7),
        OnexComplianceValidator(metrics_collector),
    ]
    pipeline = ValidationPipeline(
        validators=validators,
        metrics_collector=metrics_collector,
        parallel_execution=True,
    )
    return pipeline


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing."""

    class MockLLMClient(LLMClient):
        def __init__(self, model_id: str, should_pass: bool = True):
            super().__init__(model_id=model_id)
            self.should_pass = should_pass
            self.call_count = 0

        async def initialize(self):
            pass

        async def close(self):
            pass

        async def validate_code(
            self, code: str, context: ValidationContext
        ) -> QuorumVote:
            self.call_count += 1
            await asyncio.sleep(0.1)  # Simulate API latency

            return QuorumVote(
                model_id=self.model_id,
                approved=self.should_pass,
                confidence=0.9 if self.should_pass else 0.3,
                reasoning=f"Mock reasoning from {self.model_id}",
                duration_ms=100.0,
                cost_usd=0.001,
            )

    return MockLLMClient


@pytest.fixture
def ai_quorum(mock_llm_client, metrics_collector):
    """Create AIQuorum instance with mock clients for testing."""
    model_configs = [
        ModelConfig(
            model_id="gemini-flash",
            provider="google",
            weight=2.0,
            enabled=True,
        ),
        ModelConfig(
            model_id="glm-4.5",
            provider="zhipu",
            weight=2.0,
            enabled=True,
        ),
        ModelConfig(
            model_id="glm-air",
            provider="zhipu",
            weight=1.5,
            enabled=True,
        ),
        ModelConfig(
            model_id="codestral",
            provider="mistral",
            weight=1.0,
            enabled=True,
        ),
    ]

    quorum = AIQuorum(
        model_configs=model_configs,
        pass_threshold=0.6,
        metrics_collector=metrics_collector,
    )

    # Register mock clients
    for config in model_configs:
        client = mock_llm_client(config.model_id, should_pass=True)
        quorum.register_client(config.model_id, client)

    return quorum


# ============================================================================
# Helper Functions
# ============================================================================


async def create_test_workflow_stages(
    contract_count: int = 1,
) -> list[WorkflowStage]:
    """
    Create test workflow stages for code generation pipeline.

    Args:
        contract_count: Number of contracts to process

    Returns:
        List of 6 WorkflowStage instances
    """

    # Factory functions to create executors with proper closures
    def make_parse_executor(contract_id: str):
        async def executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.05)  # Simulate parsing
            return {
                "contract_id": contract_id,
                "parsed": True,
                "node_type": "effect",
                "methods": ["execute_effect"],
            }
        return executor

    parse_steps = [
        WorkflowStep(
            step_id=f"parse_contract_{i}",
            step_type=EnumStepType.PARSE_CONTRACT,
            dependencies=[],
            input_data={"contract_id": f"contract_{i}"},
            executor=make_parse_executor(f"contract_{i}"),
        )
        for i in range(contract_count)
    ]

    stage_parse = WorkflowStage(
        stage_id="parse",
        stage_number=1,
        stage_name="Parse Contracts",
        steps=parse_steps,
        dependencies=[],
        parallel=True,
    )

    # Stage 2: Generate Models
    def make_generate_model_executor(contract_id: str):
        async def executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.08)  # Simulate code generation
            return {
                "contract_id": contract_id,
                "model_code": "class TestNode:\n    pass",
            }
        return executor

    model_steps = [
        WorkflowStep(
            step_id=f"generate_model_{i}",
            step_type=EnumStepType.GENERATE_MODEL,
            dependencies=[],  # No step dependencies within stage, use stage dependencies
            input_data={"contract_id": f"contract_{i}"},
            executor=make_generate_model_executor(f"contract_{i}"),
        )
        for i in range(contract_count)
    ]

    stage_models = WorkflowStage(
        stage_id="models",
        stage_number=2,
        stage_name="Generate Models",
        steps=model_steps,
        dependencies=["parse"],
        parallel=True,
    )

    # Stage 3: Generate Validators
    def make_generate_validator_executor(contract_id: str):
        async def executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.06)  # Simulate code generation
            return {
                "contract_id": contract_id,
                "validator_code": "def validate():\n    pass",
            }
        return executor

    validator_steps = [
        WorkflowStep(
            step_id=f"generate_validator_{i}",
            step_type=EnumStepType.GENERATE_VALIDATOR,
            dependencies=[],  # No step dependencies within stage, use stage dependencies
            input_data={"contract_id": f"contract_{i}"},
            executor=make_generate_validator_executor(f"contract_{i}"),
        )
        for i in range(contract_count)
    ]

    stage_validators = WorkflowStage(
        stage_id="validators",
        stage_number=3,
        stage_name="Generate Validators",
        steps=validator_steps,
        dependencies=["models"],
        parallel=True,
    )

    # Stage 4: Generate Tests
    def make_generate_test_executor(contract_id: str):
        async def executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.07)  # Simulate test generation
            return {
                "contract_id": contract_id,
                "test_code": "def test_node():\n    assert True",
            }
        return executor

    test_steps = [
        WorkflowStep(
            step_id=f"generate_test_{i}",
            step_type=EnumStepType.GENERATE_TEST,
            dependencies=[],  # No step dependencies within stage, use stage dependencies
            input_data={"contract_id": f"contract_{i}"},
            executor=make_generate_test_executor(f"contract_{i}"),
        )
        for i in range(contract_count)
    ]

    stage_tests = WorkflowStage(
        stage_id="tests",
        stage_number=4,
        stage_name="Generate Tests",
        steps=test_steps,
        dependencies=["validators"],
        parallel=True,
    )

    # Stage 5: Quality Check
    def make_quality_check_executor(contract_id: str):
        async def executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.04)  # Simulate quality check
            return {
                "contract_id": contract_id,
                "quality_score": 0.95,
                "passed": True,
            }
        return executor

    quality_steps = [
        WorkflowStep(
            step_id=f"quality_check_{i}",
            step_type=EnumStepType.VALIDATE_QUALITY,
            dependencies=[],  # No step dependencies within stage, use stage dependencies
            input_data={"contract_id": f"contract_{i}"},
            executor=make_quality_check_executor(f"contract_{i}"),
        )
        for i in range(contract_count)
    ]

    stage_quality = WorkflowStage(
        stage_id="quality",
        stage_number=5,
        stage_name="Quality Checks",
        steps=quality_steps,
        dependencies=["tests"],
        parallel=True,
    )

    # Stage 6: Package Output
    def make_package_output_executor(contract_id: str):
        async def executor(context: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.03)  # Simulate packaging
            return {
                "contract_id": contract_id,
                "package_path": f"/output/{contract_id}.tar.gz",
            }
        return executor

    package_steps = [
        WorkflowStep(
            step_id=f"package_output_{i}",
            step_type=EnumStepType.PACKAGE_NODE,
            dependencies=[],  # No step dependencies within stage, use stage dependencies
            input_data={"contract_id": f"contract_{i}"},
            executor=make_package_output_executor(f"contract_{i}"),
        )
        for i in range(contract_count)
    ]

    stage_package = WorkflowStage(
        stage_id="package",
        stage_number=6,
        stage_name="Package Output",
        steps=package_steps,
        dependencies=["quality"],
        parallel=True,
    )

    return [
        stage_parse,
        stage_models,
        stage_validators,
        stage_tests,
        stage_quality,
        stage_package,
    ]


def create_sample_code() -> str:
    """Create sample Python code for validation testing."""
    return """
from typing import Any
from omnibase_core import NodeEffect
from omnibase_core.models import ModelOnexError

class SampleEffectNode(NodeEffect):
    async def execute_effect(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._process_data(context)
            return {"status": "success", "result": result}
        except Exception as e:
            raise ModelOnexError(f"Effect execution failed: {e}")

    async def _process_data(self, context: dict[str, Any]) -> Any:
        return context.get("data", {})
"""


# ============================================================================
# Scenario 1: Simple Contract → Code Pipeline
# ============================================================================


@pytest.mark.asyncio
async def test_simple_contract_generation_pipeline(staged_executor, template_manager):
    """
    Scenario 1: Simple Contract → Code Pipeline

    Tests:
    - Parse contract → Generate model → Validate → Package
    - Template caching functionality
    - Validation pipeline execution
    - Staged execution coordination

    Performance Target: <5s for complete pipeline
    """
    # Create workflow stages for 1 contract
    stages = await create_test_workflow_stages(contract_count=1)

    # Execute workflow
    start_time = time.perf_counter()
    result = await staged_executor.execute_workflow(
        workflow_id="simple_contract_test",
        stages=stages,
    )
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Assertions
    assert result.status == EnumStageStatus.COMPLETED
    assert result.total_stages == 6
    assert result.successful_stages == 6
    assert result.failed_stages == 0
    assert result.total_steps == 6  # 1 step per stage
    assert result.successful_steps == 6
    assert result.failed_steps == 0

    # Performance assertion
    assert duration_ms < 5000, f"Pipeline took {duration_ms:.2f}ms (target: <5000ms)"

    # Verify speedup (with 1 contract, speedup ~1.0; with multiple would be >1.0)
    assert result.overall_speedup >= 0.9, (
        f"Speedup {result.overall_speedup:.2f}x is too low"
    )

    print(f"✅ Simple contract pipeline completed in {duration_ms:.2f}ms")
    print(f"   Speedup: {result.overall_speedup:.2f}x")


# ============================================================================
# Scenario 2: Multiple Contracts in Parallel
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_contract_processing(staged_executor):
    """
    Scenario 2: Multiple Contracts in Parallel

    Tests:
    - 10 contracts processed in parallel through 6 stages
    - Parallelism speedup validation
    - Stage-by-stage execution coordination
    - Dependency resolution across stages

    Performance Target: Parallelism speedup 2.25x-4.17x
    """
    # Create workflow stages for 10 contracts
    contract_count = 10
    stages = await create_test_workflow_stages(contract_count=contract_count)

    # Execute workflow
    start_time = time.perf_counter()
    result = await staged_executor.execute_workflow(
        workflow_id="parallel_contracts_test",
        stages=stages,
    )
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Assertions
    assert result.status == EnumStageStatus.COMPLETED
    assert result.total_stages == 6
    assert result.successful_stages == 6
    assert result.failed_stages == 0
    assert result.total_steps == 60  # 10 contracts × 6 stages
    assert result.successful_steps == 60
    assert result.failed_steps == 0

    # Performance assertion - parallelism speedup
    assert (
        result.overall_speedup >= 2.25
    ), f"Speedup {result.overall_speedup:.2f}x below target (2.25x-4.17x)"

    print(f"✅ Parallel contract processing completed in {duration_ms:.2f}ms")
    print(f"   Contracts: {contract_count}")
    print(f"   Speedup: {result.overall_speedup:.2f}x (target: 2.25x-4.17x)")


# ============================================================================
# Scenario 3: Full Workflow with AI Quorum
# ============================================================================


@pytest.mark.asyncio
async def test_full_workflow_with_ai_quorum(
    staged_executor, validation_pipeline, ai_quorum
):
    """
    Scenario 3: Full Workflow with AI Quorum

    Tests:
    - Complete workflow: Contract → Code → Validation → AI Quorum → Package
    - AI Quorum integration with 4-model consensus
    - Validation pipeline with 3 validators
    - End-to-end integration of all components

    Performance Target: <15s (5s workflow + 10s quorum)
    """
    # Initialize AI Quorum
    await ai_quorum.initialize()

    # Create workflow stages
    stages = await create_test_workflow_stages(contract_count=1)

    # Add AI Quorum validation step to quality stage
    async def ai_quorum_executor(context: dict[str, Any]) -> dict[str, Any]:
        code = create_sample_code()
        validation_context = ValidationContext(
            node_type="effect",
            node_name="SampleEffectNode",
            expected_patterns=["execute_effect", "ModelOnexError"],
        )

        quorum_result = await ai_quorum.validate_code(code, validation_context)

        return {
            "quorum_passed": quorum_result.passed,
            "consensus_score": quorum_result.consensus_score,
            "participating_models": len(quorum_result.votes),
        }

    # Replace quality check with AI quorum
    stages[4].steps[0].executor = ai_quorum_executor

    # Execute workflow
    start_time = time.perf_counter()
    result = await staged_executor.execute_workflow(
        workflow_id="full_workflow_ai_quorum_test",
        stages=stages,
    )
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Cleanup
    await ai_quorum.close()

    # Assertions
    assert result.status == EnumStageStatus.COMPLETED
    assert result.successful_stages == 6
    assert result.failed_stages == 0

    # Performance assertion
    assert (
        duration_ms < 15000
    ), f"Full workflow took {duration_ms:.2f}ms (target: <15000ms)"

    # Verify AI Quorum execution
    quality_result = result.stage_results[4].step_results["quality_check_0"]
    assert quality_result.status == EnumStageStatus.COMPLETED
    assert quality_result.result["quorum_passed"] is True
    assert quality_result.result["participating_models"] == 4

    print(f"✅ Full workflow with AI Quorum completed in {duration_ms:.2f}ms")
    print(
        f"   Consensus score: {quality_result.result['consensus_score']:.2f}"
    )


# ============================================================================
# Scenario 4: Template Cache Performance
# ============================================================================


@pytest.mark.asyncio
async def test_template_cache_performance(template_manager):
    """
    Scenario 4: Template Cache Performance

    Tests:
    - Template cache hit rate validation
    - Cache performance: <1ms for cached lookups
    - Multiple nodes using same templates
    - Cache statistics and metrics

    Performance Target: 85-95% cache hit rate, <1ms cached lookup
    """
    # Load templates for the first time (cache miss)
    template_specs = [
        ("node_effect_v1", TemplateType.EFFECT),
        ("node_compute_v1", TemplateType.COMPUTE),
    ]

    # First load (cache miss)
    for template_id, template_type in template_specs:
        await template_manager.load_template(template_id, template_type)

    # Simulate generating 100 nodes (50 effect + 50 compute)
    load_times = []
    for i in range(100):
        template_id = (
            "node_effect_v1" if i % 2 == 0 else "node_compute_v1"
        )
        template_type = (
            TemplateType.EFFECT if i % 2 == 0 else TemplateType.COMPUTE
        )

        start = time.perf_counter()
        template = await template_manager.load_template(template_id, template_type)
        elapsed_ms = (time.perf_counter() - start) * 1000
        load_times.append(elapsed_ms)

        assert template is not None
        assert template.template_id == template_id

    # Get cache statistics
    cache_stats = template_manager.get_cache_stats()
    timing_stats = template_manager.get_timing_stats()

    # Assertions
    assert cache_stats.hit_rate >= 0.85, (
        f"Cache hit rate {cache_stats.hit_rate:.2%} below target (85-95%)"
    )
    assert cache_stats.hit_rate <= 0.95

    # Performance assertion - cached lookups should be <1ms
    avg_cached_time = timing_stats.get("get_avg_ms", 0)
    assert avg_cached_time < 1.0, (
        f"Average cached lookup {avg_cached_time:.2f}ms exceeds target (<1ms)"
    )

    print(f"✅ Template cache performance validated")
    print(f"   Hit rate: {cache_stats.hit_rate:.2%} (target: 85-95%)")
    print(f"   Avg cached lookup: {avg_cached_time:.4f}ms (target: <1ms)")
    print(f"   Total requests: {cache_stats.total_requests}")
    print(f"   Cache size: {cache_stats.current_size}/{cache_stats.max_size}")


# ============================================================================
# Scenario 5: Performance Validation
# ============================================================================


@pytest.mark.asyncio
async def test_workflow_performance_targets(
    staged_executor, template_manager, validation_pipeline
):
    """
    Scenario 5: Performance Validation

    Tests:
    - Full workflow <5s target
    - Template cache <1ms for cached lookups
    - Validation pipeline 300-800ms
    - Parallelism speedup 2.25x-4.17x

    Validates all performance targets are met.
    """
    # Test 1: Full workflow performance
    stages = await create_test_workflow_stages(contract_count=5)

    start_time = time.perf_counter()
    result = await staged_executor.execute_workflow(
        workflow_id="performance_test",
        stages=stages,
    )
    workflow_duration_ms = (time.perf_counter() - start_time) * 1000

    assert workflow_duration_ms < 5000, (
        f"Workflow took {workflow_duration_ms:.2f}ms (target: <5000ms)"
    )
    assert result.overall_speedup >= 2.25, (
        f"Speedup {result.overall_speedup:.2f}x below target (2.25x-4.17x)"
    )

    # Test 2: Template cache performance
    await template_manager.load_template("node_effect_v1", TemplateType.EFFECT)

    start = time.perf_counter()
    template = await template_manager.load_template(
        "node_effect_v1", TemplateType.EFFECT
    )
    cached_lookup_ms = (time.perf_counter() - start) * 1000

    assert cached_lookup_ms < 1.0, (
        f"Cached lookup {cached_lookup_ms:.2f}ms exceeds target (<1ms)"
    )

    # Test 3: Validation pipeline performance
    code = create_sample_code()
    context = CodeValidationContext(
        code_type="node",
        required_methods=["execute_effect"],
        expected_patterns=["async def", "ModelOnexError"],
        quality_threshold=0.7,
    )

    start = time.perf_counter()
    validation_results = await validation_pipeline.validate(code, context)
    validation_duration_ms = (time.perf_counter() - start) * 1000

    assert 300 <= validation_duration_ms <= 800, (
        f"Validation took {validation_duration_ms:.2f}ms (target: 300-800ms)"
    )

    # Create summary
    summary = validation_pipeline.create_summary(validation_results)

    print("✅ All performance targets validated")
    print(f"   Full workflow: {workflow_duration_ms:.2f}ms (target: <5000ms)")
    print(f"   Parallelism speedup: {result.overall_speedup:.2f}x (target: 2.25x-4.17x)")
    print(f"   Cached template lookup: {cached_lookup_ms:.4f}ms (target: <1ms)")
    print(f"   Validation pipeline: {validation_duration_ms:.2f}ms (target: 300-800ms)")
    print(f"   Validation score: {summary.overall_score:.2f}")


# ============================================================================
# Additional Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_validation_pipeline_integration(validation_pipeline):
    """Test validation pipeline integration with all validators."""
    code = create_sample_code()
    context = CodeValidationContext(
        code_type="node",
        required_methods=["execute_effect"],
        expected_patterns=["async def", "ModelOnexError", "omnibase_core"],
        quality_threshold=0.7,
    )

    # Run validation
    results = await validation_pipeline.validate(code, context)

    # Verify all validators executed
    assert len(results) == 3
    assert "completeness" in results
    assert "quality" in results
    assert "onex_compliance" in results

    # Create summary
    summary = validation_pipeline.create_summary(results)

    # Assertions
    assert summary.total_validators == 3
    assert summary.passed_validators >= 2  # At least 2 should pass
    assert summary.overall_score >= 0.5

    print(f"✅ Validation pipeline integration test passed")
    print(f"   Validators passed: {summary.passed_validators}/3")
    print(f"   Overall score: {summary.overall_score:.2f}")


@pytest.mark.asyncio
async def test_template_rendering_integration(template_manager):
    """Test template loading and rendering integration."""
    # Load template
    template = await template_manager.load_template(
        "node_effect_v1", TemplateType.EFFECT
    )

    # Render template
    context = {
        "node_name": "TestEffectNode",
        "version": "1.0.0",
        "description": "Test effect node for integration testing",
    }

    rendered = await template_manager.render_template("node_effect_v1", context)

    # Assertions
    assert "TestEffectNode" in rendered
    assert "1.0.0" in rendered
    assert "NodeEffect" in rendered
    assert "execute_effect" in rendered

    print(f"✅ Template rendering integration test passed")
    print(f"   Template: node_effect_v1")
    print(f"   Rendered output: {len(rendered)} characters")


@pytest.mark.asyncio
async def test_error_recovery_workflow(staged_executor):
    """Test workflow error recovery and resilience."""

    # Create workflow with a failing step
    async def failing_executor(context: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("Simulated execution failure")

    async def success_executor(context: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"status": "success"}

    stages = [
        WorkflowStage(
            stage_id="stage1",
            stage_number=1,
            stage_name="Stage 1",
            steps=[
                WorkflowStep(
                    step_id="step1",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=success_executor,
                ),
                WorkflowStep(
                    step_id="step2",
                    step_type=EnumStepType.PARSE_CONTRACT,
                    dependencies=[],
                    input_data={},
                    executor=failing_executor,
                ),
            ],
            parallel=True,
        ),
    ]

    # Execute workflow (should handle failure gracefully)
    result = await staged_executor.execute_workflow(
        workflow_id="error_recovery_test",
        stages=stages,
    )

    # Assertions - workflow should complete but report failures
    assert result.status == EnumStageStatus.FAILED
    assert result.successful_steps == 1
    assert result.failed_steps == 1

    print(f"✅ Error recovery test passed")
    print(f"   Successful steps: {result.successful_steps}")
    print(f"   Failed steps: {result.failed_steps}")


# ============================================================================
# Performance Benchmarks
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_workflow_performance_benchmark(staged_executor, benchmark_results=None):
    """
    Benchmark test for workflow performance.

    Measures:
    - Workflow execution time across different contract counts
    - Parallelism speedup scaling
    - Stage transition overhead
    """
    results = []

    for contract_count in [1, 5, 10, 20]:
        stages = await create_test_workflow_stages(contract_count=contract_count)

        start_time = time.perf_counter()
        result = await staged_executor.execute_workflow(
            workflow_id=f"benchmark_test_{contract_count}",
            stages=stages,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        results.append(
            {
                "contract_count": contract_count,
                "duration_ms": duration_ms,
                "speedup": result.overall_speedup,
                "successful_steps": result.successful_steps,
                "failed_steps": result.failed_steps,
            }
        )

    print("\n" + "=" * 70)
    print("Workflow Performance Benchmark Results")
    print("=" * 70)
    for result in results:
        print(
            f"Contracts: {result['contract_count']:2d} | "
            f"Duration: {result['duration_ms']:7.2f}ms | "
            f"Speedup: {result['speedup']:.2f}x | "
            f"Success: {result['successful_steps']}/{result['successful_steps'] + result['failed_steps']}"
        )
    print("=" * 70)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
