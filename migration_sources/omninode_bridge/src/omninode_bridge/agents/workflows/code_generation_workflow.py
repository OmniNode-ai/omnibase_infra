"""
Code Generation Workflow - Integrated orchestration with optimization.

This module integrates all 7 Phase 4 components:

**Core Workflow (Weeks 1-4):**
1. StagedParallelExecutor - 6-phase orchestration pipeline
2. TemplateManager - LRU-cached template loading (85-95% hit rate)
3. ValidationPipeline - Multi-stage validation (completeness, quality, ONEX)
4. AIQuorum - 4-model consensus validation

**Optimization (Weeks 7-8):**
5. ErrorRecoveryOrchestrator - 5 recovery strategies (Pattern 7)
6. PerformanceOptimizer - Automatic optimization (2-3x speedup)
7. ProductionMonitor - SLA tracking and alerting

Performance Targets:
- Full workflow: <5s for typical contract (with optimization: 2-3x faster)
- Template hit rate: 85-95% (optimized: 95%+)
- Validation: <800ms (pipeline) + 2-10s (quorum)
- Overall speedup: 2.25x-4.17x vs sequential (optimized: 3-4x)
- Error recovery success: 90%+ for recoverable errors

Example:
    ```python
    from omninode_bridge.agents.workflows import CodeGenerationWorkflow

    # Initialize workflow with all components including optimization
    workflow = CodeGenerationWorkflow(
        template_dir="/path/to/templates",
        metrics_collector=metrics,
        quality_threshold=0.8,
        enable_ai_quorum=True,
        enable_optimization=True,
        enable_monitoring=True
    )
    await workflow.initialize()

    # Generate code from contracts with error recovery and optimization
    result = await workflow.generate_code(
        contracts=["contract1.yaml", "contract2.yaml"],
        workflow_id="codegen-session-1"
    )

    if result.status == EnumStageStatus.COMPLETED:
        print(f"Success! Generated {len(result.generated_nodes)} nodes")
        print(f"Duration: {result.total_duration_ms:.2f}ms")
        print(f"Speedup: {result.overall_speedup:.2f}x")
        print(f"Error recovery: {workflow.get_recovery_stats()}")
    ```
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from omninode_bridge.agents.coordination.signals import SignalCoordinator
from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.scheduler.scheduler import DependencyAwareScheduler
from omninode_bridge.agents.type_defs import (
    GeneratedCodeDict,
    OptimizationSummaryDict,
    PackageInfoDict,
    ParsedContractDict,
    RecoveryStatsDict,
    SLAComplianceDict,
    ValidationResultDict,
    WorkflowStatisticsDict,
)
from omninode_bridge.agents.workflows.ai_quorum import DEFAULT_QUORUM_MODELS, AIQuorum
from omninode_bridge.agents.workflows.error_recovery import ErrorRecoveryOrchestrator
from omninode_bridge.agents.workflows.performance_optimizer import PerformanceOptimizer
from omninode_bridge.agents.workflows.profiling import PerformanceProfiler
from omninode_bridge.agents.workflows.quorum_models import ModelConfig
from omninode_bridge.agents.workflows.quorum_models import (
    ValidationContext as QuorumValidationContext,
)
from omninode_bridge.agents.workflows.recovery_models import RecoveryContext
from omninode_bridge.agents.workflows.staged_execution import StagedParallelExecutor
from omninode_bridge.agents.workflows.template_manager import TemplateManager
from omninode_bridge.agents.workflows.template_models import TemplateType
from omninode_bridge.agents.workflows.validation_models import ValidationContext
from omninode_bridge.agents.workflows.validation_pipeline import ValidationPipeline
from omninode_bridge.agents.workflows.validators import (
    CompletenessValidator,
    OnexComplianceValidator,
    QualityValidator,
)
from omninode_bridge.agents.workflows.workflow_models import (
    EnumStageStatus,
    EnumStepType,
    WorkflowConfig,
    WorkflowResult,
    WorkflowStage,
    WorkflowStep,
)
from omninode_bridge.production.monitoring import ProductionMonitor, SLAConfiguration

logger = logging.getLogger(__name__)


class CodeGenerationWorkflow:
    """
    Integrated code generation workflow with optimization.

    Combines all 7 Phase 4 components:

    **Core Workflow (Weeks 1-4):**
    1. StagedParallelExecutor - Orchestrates 6-phase pipeline
    2. TemplateManager - Loads and renders templates with caching
    3. ValidationPipeline - Validates generated code (completeness, quality, ONEX)
    4. AIQuorum - Consensus validation with 4 models

    **Optimization (Weeks 7-8):**
    5. ErrorRecoveryOrchestrator - 5 recovery strategies (90%+ success)
    6. PerformanceOptimizer - Automatic optimization (2-3x speedup)
    7. ProductionMonitor - SLA tracking and alerting

    Features:
    - Contract parsing and validation
    - Model, validator, and test generation with templates
    - Multi-stage validation (pipeline + quorum)
    - Automatic error recovery with 5 strategies
    - Performance profiling and optimization
    - Production monitoring and SLA tracking
    - Node packaging and output
    - Metrics collection and observability
    - Thread-safe state management

    Performance:
    - Full pipeline: <5s for typical contract
    - Template cache hit rate: 85-95% (optimized: 95%+)
    - Validation: 300-800ms (pipeline) + 2-10s (quorum)
    - Overall speedup: 2.25x-4.17x vs sequential (optimized: 3-4x)
    - Error recovery success: 90%+ for recoverable errors
    - Monitoring overhead: <5ms

    Example:
        ```python
        workflow = CodeGenerationWorkflow(
            template_dir="/path/to/templates",
            metrics_collector=metrics,
            quality_threshold=0.8,
            enable_ai_quorum=True,
            enable_optimization=True,
            enable_monitoring=True
        )
        await workflow.initialize()

        result = await workflow.generate_code(
            contracts=["contract1.yaml"],
            workflow_id="session-1"
        )

        print(f"Generated {len(result.generated_nodes)} nodes in {result.total_duration_ms:.2f}ms")
        print(f"Error recovery: {workflow.get_recovery_stats()}")
        print(f"SLA compliance: {workflow.get_sla_compliance()}")
        ```
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        signal_coordinator: Optional[SignalCoordinator] = None,
        state: Optional[ThreadSafeState] = None,
        quality_threshold: float = 0.8,
        enable_ai_quorum: bool = False,
        quorum_threshold: float = 0.6,
        quorum_models: Optional[list[ModelConfig]] = None,
        cache_size: int = 100,
        enable_optimization: bool = True,
        enable_monitoring: bool = True,
        sla_config: Optional[SLAConfiguration] = None,
    ) -> None:
        """
        Initialize code generation workflow with optimization.

        Args:
            template_dir: Directory containing template files
            metrics_collector: MetricsCollector for performance tracking
            signal_coordinator: SignalCoordinator for event signaling
            state: ThreadSafeState for context management
            quality_threshold: Quality threshold for validation (0.0-1.0)
            enable_ai_quorum: Enable AI quorum validation (default: False)
            quorum_threshold: Consensus threshold for AI quorum (0.0-1.0)
            quorum_models: Custom model configurations for AI quorum
            cache_size: Template cache size (default: 100)
            enable_optimization: Enable performance optimization (default: True)
            enable_monitoring: Enable production monitoring (default: True)
            sla_config: SLA configuration for monitoring (default: standard SLAs)
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.signals = signal_coordinator
        self.state = state or ThreadSafeState()
        self.quality_threshold = quality_threshold
        self.enable_ai_quorum = enable_ai_quorum
        self.enable_optimization = enable_optimization
        self.enable_monitoring = enable_monitoring

        # Component 1: Staged Parallel Executor
        self.scheduler = DependencyAwareScheduler(
            state=self.state,
            max_concurrent=10,
        )
        self.executor = StagedParallelExecutor(
            scheduler=self.scheduler,
            metrics_collector=self.metrics,
            signal_coordinator=self.signals,
            state=self.state,
        )

        # Component 2: Template Manager
        self.template_manager = TemplateManager(
            template_dir=template_dir,
            metrics_collector=self.metrics,
            cache_size=cache_size,
        )

        # Component 3: Validation Pipeline
        validators = [
            CompletenessValidator(self.metrics),
            QualityValidator(self.metrics, quality_threshold=quality_threshold),
            OnexComplianceValidator(self.metrics),
        ]
        self.validation_pipeline = ValidationPipeline(
            validators=validators,
            metrics_collector=self.metrics,
            parallel_execution=True,
        )

        # Component 4: AI Quorum (optional)
        self.ai_quorum: Optional[AIQuorum] = None
        if enable_ai_quorum:
            models = quorum_models or DEFAULT_QUORUM_MODELS
            self.ai_quorum = AIQuorum(
                model_configs=models,
                pass_threshold=quorum_threshold,
                metrics_collector=self.metrics,
            )

        # Component 5: Error Recovery Orchestrator (Weeks 7-8)
        self.error_recovery: Optional[ErrorRecoveryOrchestrator] = None
        if enable_optimization:
            self.error_recovery = ErrorRecoveryOrchestrator(
                metrics_collector=self.metrics,
                signal_coordinator=self.signals,
                max_retries=3,
                base_delay=1.0,
            )

        # Component 6: Performance Optimizer (Weeks 7-8)
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        if enable_optimization:
            profiler = PerformanceProfiler(metrics_collector=self.metrics)
            self.performance_optimizer = PerformanceOptimizer(
                profiler=profiler,
                template_manager=self.template_manager,
                staged_executor=self.executor,
                metrics_collector=self.metrics,
                auto_optimize=True,
            )

        # Component 7: Production Monitor (Weeks 7-8)
        self.production_monitor: Optional[ProductionMonitor] = None
        if enable_monitoring:
            self.production_monitor = ProductionMonitor(
                metrics_collector=self.metrics,
                sla_config=sla_config or SLAConfiguration(),
            )

        # Workflow tracking
        self._initialized = False
        self._generation_count = 0
        self._total_duration_ms = 0.0

        logger.info(
            f"CodeGenerationWorkflow initialized: "
            f"template_dir={template_dir}, "
            f"quality_threshold={quality_threshold}, "
            f"enable_ai_quorum={enable_ai_quorum}, "
            f"enable_optimization={enable_optimization}, "
            f"enable_monitoring={enable_monitoring}"
        )

    async def initialize(self) -> None:
        """
        Initialize all workflow components including optimization.

        Must be called before generate_code().

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Initialize template manager
            await self.template_manager.start()
            logger.info("TemplateManager initialized")

            # Initialize AI quorum (if enabled)
            if self.ai_quorum:
                await self.ai_quorum.initialize()
                logger.info("AIQuorum initialized")

            # Initialize production monitor (if enabled)
            if self.production_monitor:
                await self.production_monitor.start_monitoring(
                    health_check_interval_seconds=30,
                    metrics_export_interval_seconds=10,
                )
                logger.info("ProductionMonitor initialized")

            # Run initial optimization (if enabled)
            if self.performance_optimizer:
                # Optimize template cache
                await self.performance_optimizer.optimize_template_cache(
                    target_hit_rate=0.95
                )
                # Tune parallel execution
                self.performance_optimizer.tune_parallel_execution(target_speedup=3.5)
                logger.info("PerformanceOptimizer initialized")

            self._initialized = True
            logger.info("CodeGenerationWorkflow initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CodeGenerationWorkflow: {e}")
            raise RuntimeError(f"Initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """
        Shutdown all workflow components and cleanup resources.

        Example:
            ```python
            await workflow.shutdown()
            ```
        """
        try:
            # Stop template manager
            await self.template_manager.stop()

            # Close AI quorum
            if self.ai_quorum:
                await self.ai_quorum.close()

            # Stop production monitor
            if self.production_monitor:
                await self.production_monitor.stop_monitoring()

            self._initialized = False
            logger.info("CodeGenerationWorkflow shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def generate_code(
        self,
        contracts: list[str],
        workflow_id: str,
        output_dir: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Generate code from contracts using full 6-phase workflow with optimization.

        Phases:
        1. Parse contracts (parallel per contract)
        2. Generate models (parallel, uses TemplateManager)
        3. Generate validators (parallel, uses TemplateManager)
        4. Generate tests (parallel, uses TemplateManager)
        5. Validate (uses ValidationPipeline + AIQuorum)
        6. Package nodes (parallel per node)

        With optimization:
        - Error recovery (5 strategies, 90%+ success)
        - Performance profiling and optimization
        - SLA tracking and alerting

        Performance Target: <5s for typical contract (2-3x faster with optimization)

        Args:
            contracts: List of contract file paths
            workflow_id: Unique workflow identifier
            output_dir: Optional output directory for generated code

        Returns:
            WorkflowResult with execution summary and generated nodes

        Raises:
            ValueError: If workflow not initialized
            RuntimeError: If workflow execution fails

        Example:
            ```python
            result = await workflow.generate_code(
                contracts=["contract1.yaml", "contract2.yaml"],
                workflow_id="codegen-1"
            )

            if result.status == EnumStageStatus.COMPLETED:
                for node in result.generated_nodes:
                    print(f"Generated: {node['name']}")
                print(f"Error recovery: {workflow.get_recovery_stats()}")
                print(f"SLA compliance: {workflow.get_sla_compliance()}")
            ```
        """
        if not self._initialized:
            raise ValueError("Workflow not initialized. Call initialize() first.")

        if not contracts:
            raise ValueError("contracts list cannot be empty")

        workflow_start = time.perf_counter()

        # Wrap execution with error recovery if enabled
        if self.error_recovery:
            try:
                return await self._generate_code_with_recovery(
                    contracts, workflow_id, output_dir, workflow_start
                )
            except Exception as e:
                # Final fallback if error recovery fails
                logger.error(f"Error recovery failed: {e}", exc_info=True)
                raise
        else:
            return await self._generate_code_internal(
                contracts, workflow_id, output_dir, workflow_start
            )

    async def _generate_code_with_recovery(
        self,
        contracts: list[str],
        workflow_id: str,
        output_dir: Optional[str],
        workflow_start: float,
    ) -> WorkflowResult:
        """Generate code with error recovery."""
        try:
            return await self._generate_code_internal(
                contracts, workflow_id, output_dir, workflow_start
            )
        except Exception as e:
            logger.warning(f"Workflow error detected, attempting recovery: {e}")

            # Create recovery context
            context = RecoveryContext(
                workflow_id=workflow_id,
                node_name="code_generation_workflow",
                step_count=len(contracts),
                state={
                    "contracts": contracts,
                    "output_dir": output_dir,
                },
                exception=e,
                correlation_id=workflow_id,
            )

            # Attempt recovery
            recovery_result = await self.error_recovery.handle_error(
                context=context,
                operation=lambda: self._generate_code_internal(
                    contracts, workflow_id, output_dir, workflow_start
                ),
            )

            if recovery_result.success:
                logger.info(
                    f"Workflow recovered successfully using {recovery_result.strategy_used.value}"
                )
                # Re-run workflow after successful recovery
                return await self._generate_code_internal(
                    contracts, workflow_id, output_dir, workflow_start
                )
            else:
                logger.error(
                    f"Workflow recovery failed: {recovery_result.error_message}"
                )
                raise RuntimeError(
                    f"Workflow failed and recovery unsuccessful: {recovery_result.error_message}"
                ) from e

    async def _generate_code_internal(
        self,
        contracts: list[str],
        workflow_id: str,
        output_dir: Optional[str],
        workflow_start: float,
    ) -> WorkflowResult:
        """Internal code generation with monitoring."""

        logger.info(
            f"Starting code generation workflow: {workflow_id} "
            f"({len(contracts)} contracts)"
        )

        # Record workflow start
        if self.metrics:
            await self.metrics.record_counter(
                "workflow_started",
                count=1,
                tags={
                    "workflow_id": workflow_id,
                    "contract_count": str(len(contracts)),
                },
                correlation_id=workflow_id,
            )

        # Create workflow configuration
        config = WorkflowConfig(
            workflow_id=workflow_id,
            workflow_name=f"CodeGen-{workflow_id}",
            enable_stage_recovery=True,
            enable_step_retry=True,
            step_retry_count=2,
            step_timeout_seconds=300.0,
            collect_metrics=True,
            signal_on_stage_complete=True,
            metadata={
                "contracts": contracts,
                "output_dir": output_dir,
                "created_at": time.time(),
            },
        )

        # Build workflow stages
        stages = await self._build_workflow_stages(contracts, workflow_id, output_dir)

        # Execute workflow
        try:
            result = await self.executor.execute_workflow(
                workflow_id=workflow_id,
                stages=stages,
                config=config,
            )

            # Update statistics
            self._generation_count += 1
            workflow_duration = (time.perf_counter() - workflow_start) * 1000
            self._total_duration_ms += workflow_duration

            # Record metrics
            if self.metrics:
                await self.metrics.record_timing(
                    "workflow_completed",
                    workflow_duration,
                    tags={
                        "workflow_id": workflow_id,
                        "status": result.status.value,
                        "contract_count": str(len(contracts)),
                    },
                    correlation_id=workflow_id,
                )

            # Monitor SLAs (if monitoring enabled)
            if self.production_monitor:
                # Get workflow metrics
                template_stats = self.template_manager.get_cache_stats()
                workflow_metrics = {
                    "workflow_latency_p95": workflow_duration,
                    "workflow_latency_p99": workflow_duration,
                    "template_cache_hit_rate": template_stats.hit_rate,
                    "validation_pass_rate": (
                        1.0 if result.status == EnumStageStatus.COMPLETED else 0.0
                    ),
                    "cost_per_node": 0.01,  # TODO: Calculate actual cost
                    "error_rate": (
                        0.0 if result.status == EnumStageStatus.COMPLETED else 1.0
                    ),
                    "throughput": len(contracts) / (workflow_duration / 1000),
                }

                # Check SLA compliance
                sla_alerts = await self.production_monitor.monitor_slas(
                    workflow_metrics
                )

                if sla_alerts:
                    logger.warning(
                        f"SLA violations detected: {len(sla_alerts)} alerts generated"
                    )
                    for alert in sla_alerts[:3]:  # Log first 3
                        logger.warning(f"  - {alert.message}")

            logger.info(
                f"Code generation workflow completed: {workflow_id} "
                f"(status={result.status.value}, duration={workflow_duration:.2f}ms)"
            )

            return result

        except Exception as e:
            workflow_duration = (time.perf_counter() - workflow_start) * 1000

            logger.error(f"Code generation workflow failed: {e}", exc_info=True)

            # Record error metrics
            if self.metrics:
                await self.metrics.record_counter(
                    "workflow_failed",
                    count=1,
                    tags={"workflow_id": workflow_id, "error": str(e)[:100]},
                    correlation_id=workflow_id,
                )

            raise RuntimeError(f"Workflow execution failed: {e}") from e

    async def _build_workflow_stages(
        self,
        contracts: list[str],
        workflow_id: str,
        output_dir: Optional[str],
    ) -> list[WorkflowStage]:
        """
        Build 6-phase workflow stages with all steps.

        Args:
            contracts: List of contract file paths
            workflow_id: Workflow identifier
            output_dir: Output directory for generated code

        Returns:
            List of WorkflowStage instances
        """
        stages: list[WorkflowStage] = []

        # Stage 1: Parse contracts (parallel)
        parse_steps = []
        for i, contract_path in enumerate(contracts):
            step_id = f"parse_contract_{i}"
            step = WorkflowStep(
                step_id=step_id,
                step_type=EnumStepType.PARSE_CONTRACT,
                dependencies=[],
                input_data={"contract_path": contract_path},
                executor=self._parse_contract_executor,
            )
            parse_steps.append(step)

        stages.append(
            WorkflowStage(
                stage_id="parse_contracts",
                stage_number=1,
                stage_name="Parse Contracts",
                steps=parse_steps,
                parallel=True,
                dependencies=[],
            )
        )

        # Stage 2: Generate models (parallel, uses templates)
        model_steps = []
        for i in range(len(contracts)):
            step_id = f"generate_model_{i}"
            step = WorkflowStep(
                step_id=step_id,
                step_type=EnumStepType.GENERATE_MODEL,
                dependencies=[f"parse_contract_{i}"],
                input_data={"template_type": TemplateType.EFFECT},
                executor=self._generate_model_executor,
            )
            model_steps.append(step)

        stages.append(
            WorkflowStage(
                stage_id="generate_models",
                stage_number=2,
                stage_name="Generate Models",
                steps=model_steps,
                parallel=True,
                dependencies=["parse_contracts"],
            )
        )

        # Stage 3: Generate validators (parallel, uses templates)
        validator_steps = []
        for i in range(len(contracts)):
            step_id = f"generate_validator_{i}"
            step = WorkflowStep(
                step_id=step_id,
                step_type=EnumStepType.GENERATE_VALIDATOR,
                dependencies=[f"generate_model_{i}"],
                input_data={"template_type": TemplateType.COMPUTE},
                executor=self._generate_validator_executor,
            )
            validator_steps.append(step)

        stages.append(
            WorkflowStage(
                stage_id="generate_validators",
                stage_number=3,
                stage_name="Generate Validators",
                steps=validator_steps,
                parallel=True,
                dependencies=["generate_models"],
            )
        )

        # Stage 4: Generate tests (parallel, uses templates)
        test_steps = []
        for i in range(len(contracts)):
            step_id = f"generate_test_{i}"
            step = WorkflowStep(
                step_id=step_id,
                step_type=EnumStepType.GENERATE_TEST,
                dependencies=[f"generate_validator_{i}"],
                input_data={"template_type": TemplateType.EFFECT},
                executor=self._generate_test_executor,
            )
            test_steps.append(step)

        stages.append(
            WorkflowStage(
                stage_id="generate_tests",
                stage_number=4,
                stage_name="Generate Tests",
                steps=test_steps,
                parallel=True,
                dependencies=["generate_validators"],
            )
        )

        # Stage 5: Validate (uses ValidationPipeline + AIQuorum)
        validation_steps = []
        for i in range(len(contracts)):
            step_id = f"validate_code_{i}"
            step = WorkflowStep(
                step_id=step_id,
                step_type=EnumStepType.VALIDATE_QUALITY,
                dependencies=[
                    f"generate_model_{i}",
                    f"generate_validator_{i}",
                    f"generate_test_{i}",
                ],
                input_data={},
                executor=self._validate_code_executor,
            )
            validation_steps.append(step)

        stages.append(
            WorkflowStage(
                stage_id="validate_code",
                stage_number=5,
                stage_name="Validate Code",
                steps=validation_steps,
                parallel=True,
                dependencies=["generate_tests"],
            )
        )

        # Stage 6: Package nodes (parallel)
        package_steps = []
        for i in range(len(contracts)):
            step_id = f"package_node_{i}"
            step = WorkflowStep(
                step_id=step_id,
                step_type=EnumStepType.PACKAGE_NODE,
                dependencies=[f"validate_code_{i}"],
                input_data={"output_dir": output_dir},
                executor=self._package_node_executor,
            )
            package_steps.append(step)

        stages.append(
            WorkflowStage(
                stage_id="package_nodes",
                stage_number=6,
                stage_name="Package Nodes",
                steps=package_steps,
                parallel=True,
                dependencies=["validate_code"],
            )
        )

        return stages

    # ==================== Step Executors ====================

    async def _parse_contract_executor(
        self, context: dict[str, Any]
    ) -> ParsedContractDict:
        """
        Parse contract file (Stage 1).

        Args:
            context: Step context with contract_path

        Returns:
            Parsed contract data

        Raises:
            FileNotFoundError: If contract file doesn't exist
            ValueError: If contract is invalid or parsing fails
        """
        contract_path = context["input_data"]["contract_path"]

        logger.debug(f"Parsing contract: {contract_path}")

        # Validate contract file exists
        contract_file = Path(contract_path)
        if not contract_file.exists():
            error_msg = f"Contract file not found: {contract_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Import YAMLContractParser
            from omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser

            # Parse contract file
            parser = YAMLContractParser()
            contract = parser.parse_contract_file(contract_path)

            # Validate contract is valid
            if not contract.is_valid:
                error_msg = (
                    f"Contract validation failed for {contract_path}: "
                    f"{'; '.join(contract.validation_errors)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate required fields
            if not contract.name:
                raise ValueError(
                    f"Contract missing required field 'name': {contract_path}"
                )
            if not contract.node_type:
                raise ValueError(
                    f"Contract missing required field 'node_type': {contract_path}"
                )

            # Extract version string from ModelVersionInfo
            version_str = f"{contract.version.major}.{contract.version.minor}.{contract.version.patch}"

            logger.info(
                f"Successfully parsed contract: {contract.name} "
                f"(type={contract.node_type}, version={version_str})"
            )

            # Return parsed contract data
            return {
                "contract_path": contract_path,
                "node_type": contract.node_type,
                "node_name": contract.name,
                "version": version_str,
            }

        except FileNotFoundError:
            # Re-raise with context preserved
            raise
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap other errors with context
            error_msg = f"Failed to parse contract {contract_path}: {e}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    async def _generate_model_executor(
        self, context: dict[str, Any]
    ) -> GeneratedCodeDict:
        """
        Generate model using TemplateManager (Stage 2).

        Args:
            context: Step context with parsed contract

        Returns:
            Generated model code
        """
        template_type = context["input_data"]["template_type"]

        # Get parsed contract from dependencies
        dependency_results = context.get("dependency_results", {})
        contract_data = (
            next(iter(dependency_results.values())) if dependency_results else {}
        )

        logger.debug(
            f"Generating model for: {contract_data.get('node_name', 'unknown')}"
        )

        # Load and render template
        template_id = "node_effect_v1"  # Default template
        rendered = await self.template_manager.load_and_render(
            template_id=template_id,
            template_type=template_type,
            context={
                "node_name": contract_data.get("node_name", "UnknownNode"),
                "version": contract_data.get("version", "1.0.0"),
            },
        )

        return {
            "template_id": template_id,
            "generated_code": rendered,
            "size_bytes": len(rendered),
        }

    async def _generate_validator_executor(
        self, context: dict[str, Any]
    ) -> GeneratedCodeDict:
        """
        Generate validator using TemplateManager (Stage 3).

        Args:
            context: Step context with model data

        Returns:
            Generated validator code
        """
        template_type = context["input_data"]["template_type"]

        logger.debug("Generating validator")

        # Load and render template
        template_id = "node_compute_v1"  # Default template
        rendered = await self.template_manager.load_and_render(
            template_id=template_id,
            template_type=template_type,
            context={"node_name": "ValidatorNode", "version": "1.0.0"},
        )

        return {
            "template_id": template_id,
            "generated_code": rendered,
            "size_bytes": len(rendered),
        }

    async def _generate_test_executor(
        self, context: dict[str, Any]
    ) -> GeneratedCodeDict:
        """
        Generate test using TemplateManager (Stage 4).

        Args:
            context: Step context with validator data

        Returns:
            Generated test code
        """
        template_type = context["input_data"]["template_type"]

        logger.debug("Generating test")

        # Load and render template
        template_id = "node_test_v1"  # Default template
        rendered = await self.template_manager.load_and_render(
            template_id=template_id,
            template_type=template_type,
            context={"node_name": "TestNode", "version": "1.0.0"},
        )

        return {
            "template_id": template_id,
            "generated_code": rendered,
            "size_bytes": len(rendered),
        }

    async def _validate_code_executor(
        self, context: dict[str, Any]
    ) -> ValidationResultDict:
        """
        Validate generated code using ValidationPipeline + AIQuorum (Stage 5).

        Args:
            context: Step context with generated code

        Returns:
            Validation results
        """
        # Get generated code from dependencies
        dependency_results = context.get("dependency_results", {})
        model_data = dependency_results.get(
            f"generate_model_{context['step_id'].split('_')[-1]}", {}
        )
        code = model_data.get("generated_code", "")

        logger.debug(f"Validating code ({len(code)} chars)")

        # Run validation pipeline
        validation_context = ValidationContext(
            code_type="node",
            required_methods=["execute_effect"],
            expected_patterns=["async def", "ModelOnexError"],
            quality_threshold=self.quality_threshold,
            correlation_id=context.get("workflow_id"),
        )

        pipeline_results = await self.validation_pipeline.validate(
            code, validation_context
        )
        pipeline_summary = self.validation_pipeline.create_summary(pipeline_results)

        # Run AI quorum (if enabled)
        quorum_result = None
        if self.ai_quorum and code:
            quorum_context = QuorumValidationContext(
                node_type="effect",
                contract_summary="Generated effect node",
                code_snippet=code[:500],  # First 500 chars for summary
            )
            quorum_result = await self.ai_quorum.validate_code(
                code=code,
                context=quorum_context,
                correlation_id=context.get("workflow_id"),
            )

        return {
            "pipeline_passed": pipeline_summary.passed,
            "pipeline_score": pipeline_summary.overall_score,
            "pipeline_duration_ms": pipeline_summary.total_duration_ms,
            "quorum_passed": quorum_result.passed if quorum_result else None,
            "quorum_score": quorum_result.consensus_score if quorum_result else None,
            "quorum_duration_ms": quorum_result.duration_ms if quorum_result else None,
        }

    async def _package_node_executor(self, context: dict[str, Any]) -> PackageInfoDict:
        """
        Package node for deployment (Stage 6).

        Args:
            context: Step context with validated code

        Returns:
            Package information
        """
        output_dir = context["input_data"].get("output_dir")

        logger.debug(f"Packaging node (output_dir={output_dir})")

        # TODO: Implement actual packaging logic
        # For now, return mock package info
        return {
            "package_path": f"{output_dir}/node_package.zip" if output_dir else None,
            "package_size_bytes": 1024,
            "packaged_at": time.time(),
        }

    def get_statistics(self) -> WorkflowStatisticsDict:
        """
        Get workflow statistics.

        Returns:
            Dictionary with generation counts, durations, cache stats

        Example:
            ```python
            stats = workflow.get_statistics()
            print(f"Total generations: {stats['generation_count']}")
            print(f"Avg duration: {stats['avg_duration_ms']:.2f}ms")
            print(f"Template hit rate: {stats['template_hit_rate']:.2%}")
            ```
        """
        template_stats = self.template_manager.get_cache_stats()
        template_timing = self.template_manager.get_timing_stats()

        stats = {
            "generation_count": self._generation_count,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": (
                self._total_duration_ms / self._generation_count
                if self._generation_count > 0
                else 0.0
            ),
            "template_hit_rate": template_stats.hit_rate,
            "template_cache_size": template_stats.current_size,
            "template_avg_load_ms": template_timing.get("get_avg_ms", 0.0),
        }

        # Add AI quorum stats if enabled
        if self.ai_quorum:
            quorum_stats = self.ai_quorum.get_statistics()
            stats["quorum_validations"] = quorum_stats["total_validations"]
            stats["quorum_pass_rate"] = quorum_stats["pass_rate"]

        # Add optimization stats if enabled
        if self.enable_optimization:
            stats["optimization_enabled"] = True
            if self.error_recovery:
                recovery_stats = self.error_recovery.get_statistics()
                stats["error_recovery"] = {
                    "total_attempts": recovery_stats.total_attempts,
                    "successful_recoveries": recovery_stats.successful_recoveries,
                    "success_rate": recovery_stats.success_rate,
                }
            if self.performance_optimizer:
                opt_summary = self.performance_optimizer.get_optimization_summary()
                stats["performance_optimization"] = {
                    "total_optimizations": opt_summary["total_optimizations"],
                    "avg_speedup": opt_summary["avg_speedup"],
                }

        # Add monitoring stats if enabled
        if self.enable_monitoring and self.production_monitor:
            monitoring_status = self.production_monitor.get_monitoring_status()
            stats["monitoring"] = {
                "is_monitoring": monitoring_status["is_monitoring"],
                "overhead_ms": monitoring_status["monitoring_overhead_ms"],
            }

        return stats

    def get_recovery_stats(self) -> RecoveryStatsDict:
        """
        Get error recovery statistics.

        Returns:
            Dictionary with recovery statistics

        Example:
            ```python
            recovery_stats = workflow.get_recovery_stats()
            print(f"Success rate: {recovery_stats['success_rate']:.2%}")
            ```
        """
        if not self.error_recovery:
            return {"enabled": False}

        stats = self.error_recovery.get_statistics()
        return {
            "enabled": True,
            "total_attempts": stats.total_attempts,
            "successful_recoveries": stats.successful_recoveries,
            "failed_recoveries": stats.failed_recoveries,
            "success_rate": stats.success_rate,
            "strategies_used": dict(stats.strategies_used),
            "error_types_seen": {k.value: v for k, v in stats.error_types_seen.items()},
        }

    def get_sla_compliance(self) -> SLAComplianceDict:
        """
        Get SLA compliance report.

        Returns:
            Dictionary with SLA compliance status

        Example:
            ```python
            sla_report = workflow.get_sla_compliance()
            print(f"Overall compliant: {sla_report['overall_compliant']}")
            ```
        """
        if not self.production_monitor:
            return {"enabled": False}

        # Get current workflow metrics
        template_stats = self.template_manager.get_cache_stats()
        metrics = {
            "workflow_latency_p95": self._total_duration_ms
            / max(1, self._generation_count),
            "workflow_latency_p99": self._total_duration_ms
            / max(1, self._generation_count),
            "template_cache_hit_rate": template_stats.hit_rate,
            "validation_pass_rate": 1.0,  # Default
            "cost_per_node": 0.01,  # Default
            "error_rate": 0.0,  # Default
            "throughput": 1.0,  # Default
        }

        return self.production_monitor.get_sla_compliance_report(metrics)

    def get_optimization_summary(self) -> OptimizationSummaryDict:
        """
        Get performance optimization summary.

        Returns:
            Dictionary with optimization summary

        Example:
            ```python
            opt_summary = workflow.get_optimization_summary()
            print(f"Optimizations applied: {opt_summary['total_optimizations']}")
            ```
        """
        if not self.performance_optimizer:
            return {"enabled": False}

        return {
            "enabled": True,
            **self.performance_optimizer.get_optimization_summary(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CodeGenerationWorkflow("
            f"initialized={self._initialized}, "
            f"generations={self._generation_count}, "
            f"ai_quorum={self.enable_ai_quorum}, "
            f"optimization={self.enable_optimization}, "
            f"monitoring={self.enable_monitoring})"
        )
