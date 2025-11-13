"""
Performance optimizer for code generation workflows.

Applies automatic optimizations based on profiling analysis to achieve
2-3x speedup vs Phase 3. Integrates with TemplateManager, StagedParallelExecutor,
and MetricsCollector.

Performance Targets:
- Overall speedup: 2-3x vs Phase 3 (measured)
- Template cache hit rate: 95%+ (from 85-95%)
- Parallel execution speedup: 3-4x (from 2.25x-4.17x)
- Memory overhead: <50MB
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Optional

from omninode_bridge.agents.metrics.collector import MetricsCollector

from .optimization_models import (
    DryRunChange,
    DryRunReport,
    OptimizationArea,
    OptimizationPriority,
    OptimizationRecommendation,
    PerformanceReport,
)
from .profiling import PerformanceProfiler
from .staged_execution import StagedParallelExecutor
from .template_manager import TemplateManager
from .template_models import TemplateType

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Performance optimizer for code generation workflows.

    Analyzes performance profiles and automatically applies optimizations:
    - Template cache preloading and tuning
    - Parallel execution parameter tuning
    - Memory usage optimization
    - I/O batching and async conversion

    Features:
    - Automatic optimization based on profiling
    - Integration with TemplateManager, StagedParallelExecutor
    - Performance tracking and validation
    - 2-3x speedup target achievement
    - Dry-run mode for simulating optimizations without applying changes

    Example:
        ```python
        optimizer = PerformanceOptimizer(
            profiler=profiler,
            template_manager=template_manager,
            staged_executor=executor,
            metrics_collector=metrics
        )

        # Profile and optimize workflow
        report = await optimizer.optimize_workflow(
            workflow_id="codegen-session-1",
            workflow_func=my_workflow
        )

        print(f"Estimated speedup: {report.estimated_speedup:.2f}x")

        # Apply specific optimizations
        await optimizer.optimize_template_cache(target_hit_rate=0.95)
        optimizer.tune_parallel_execution(target_concurrency=12)

        # Use dry-run mode to simulate optimizations
        optimizer_dry = PerformanceOptimizer(
            profiler=profiler,
            dry_run=True
        )

        await optimizer_dry.optimize_workflow(
            workflow_id="session-1",
            apply_optimizations=True
        )

        # Get dry-run report
        dry_report = optimizer_dry.get_dry_run_report()
        if dry_report:
            print(dry_report.get_summary())
        ```
    """

    def __init__(
        self,
        profiler: PerformanceProfiler,
        template_manager: Optional[TemplateManager] = None,
        staged_executor: Optional[StagedParallelExecutor] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        auto_optimize: bool = True,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize performance optimizer.

        Args:
            profiler: PerformanceProfiler for profiling analysis
            template_manager: Optional TemplateManager for cache optimization
            staged_executor: Optional StagedParallelExecutor for parallel tuning
            metrics_collector: Optional MetricsCollector for tracking
            auto_optimize: Whether to automatically apply safe optimizations
            dry_run: Whether to run in dry-run mode (simulation only, no changes)
        """
        self.profiler = profiler
        self.template_manager = template_manager
        self.staged_executor = staged_executor
        self.metrics = metrics_collector
        self.auto_optimize = auto_optimize
        self.dry_run = dry_run

        # Optimization tracking
        self._applied_optimizations: list[OptimizationRecommendation] = []
        self._optimization_history: list[PerformanceReport] = []

        # Dry-run tracking
        self._dry_run_changes: list[DryRunChange] = []
        self._current_dry_run_report: Optional[DryRunReport] = None

        # Default optimization targets
        self.target_cache_hit_rate = 0.95
        self.target_parallel_speedup = 3.5
        self.target_memory_overhead_mb = 50.0
        self.target_io_async_ratio = 0.8

        logger.info(
            f"PerformanceOptimizer initialized: auto_optimize={auto_optimize}, "
            f"dry_run={dry_run}, "
            f"targets: cache_hit_rate={self.target_cache_hit_rate:.1%}, "
            f"parallel_speedup={self.target_parallel_speedup:.1f}x, "
            f"memory_overhead={self.target_memory_overhead_mb:.0f}MB"
        )

    async def optimize_workflow(
        self,
        workflow_id: str,
        workflow_func: Optional[Callable] = None,
        iterations: int = 5,
        apply_optimizations: bool = None,
        dry_run: bool = None,
        **profile_kwargs,
    ) -> PerformanceReport:
        """
        Profile and optimize a workflow.

        Profiles the workflow, generates recommendations, and optionally
        applies automatic optimizations.

        Args:
            workflow_id: Workflow identifier
            workflow_func: Optional workflow function to profile
            iterations: Number of profiling iterations
            apply_optimizations: Whether to apply optimizations (default: auto_optimize)
            dry_run: Override instance dry_run setting (None = use instance setting)
            **profile_kwargs: Additional kwargs for profiler

        Returns:
            PerformanceReport with analysis and recommendations

        Example:
            ```python
            # Run in dry-run mode to see proposed changes
            report = await optimizer.optimize_workflow(
                workflow_id="session-1",
                workflow_func=my_workflow,
                iterations=10,
                dry_run=True
            )

            # Apply optimizations for real
            report = await optimizer.optimize_workflow(
                workflow_id="session-1",
                workflow_func=my_workflow,
                iterations=10,
                apply_optimizations=True,
                dry_run=False
            )
            ```
        """
        # Determine dry-run mode
        effective_dry_run = dry_run if dry_run is not None else self.dry_run

        # Clear previous dry-run changes
        self._dry_run_changes.clear()

        logger.info(f"Optimizing workflow: {workflow_id} (dry_run={effective_dry_run})")

        # Profile workflow if function provided
        profile_data = {}
        if workflow_func:
            logger.info(f"Profiling workflow for {iterations} iterations")
            profile_data = await self.profiler.profile_workflow(
                workflow_func=workflow_func, iterations=iterations
            )

        # Generate performance report
        report = self.profiler.generate_performance_report(
            workflow_id=workflow_id, profile_data=profile_data, **profile_kwargs
        )

        # Store report
        self._optimization_history.append(report)

        # Log report summary
        logger.info(f"Performance Report:\n{report.get_summary()}")

        # Log recommendations
        if report.recommendations:
            logger.info(f"Generated {len(report.recommendations)} recommendations:")
            for rec in report.recommendations[:5]:  # Log top 5
                logger.info(
                    f"  [{rec.priority.value.upper()}] {rec.area.value}: {rec.issue}"
                )

        # Apply optimizations if requested (and not in dry-run mode)
        should_apply = (
            apply_optimizations
            if apply_optimizations is not None
            else self.auto_optimize
        )

        if should_apply:
            await self._apply_automatic_optimizations(report, dry_run=effective_dry_run)

        # Generate dry-run report if in dry-run mode
        if effective_dry_run and self._dry_run_changes:
            self._current_dry_run_report = self._generate_dry_run_report(
                workflow_id=workflow_id,
                estimated_speedup=report.estimated_speedup,
            )
            logger.info(f"\n{self._current_dry_run_report.get_summary()}")

        # Record metrics
        if self.metrics:
            await self.metrics.record_gauge(
                "optimizer.estimated_speedup", report.estimated_speedup
            )
            await self.metrics.record_counter(
                "optimizer.recommendations_generated", len(report.recommendations)
            )
            await self.metrics.record_counter(
                "optimizer.critical_recommendations",
                len(report.critical_recommendations),
            )

        return report

    async def _apply_automatic_optimizations(
        self, report: PerformanceReport, dry_run: bool = False
    ) -> None:
        """
        Apply automatic optimizations from performance report.

        Only applies safe, non-destructive optimizations.

        Args:
            report: Performance report with recommendations
            dry_run: If True, simulate changes without applying them
        """
        mode_str = "Simulating" if dry_run else "Applying"
        logger.info(f"{mode_str} automatic optimizations")

        applied_count = 0

        for rec in report.recommendations:
            # Only apply critical and high priority optimizations automatically
            if rec.priority not in [
                OptimizationPriority.CRITICAL,
                OptimizationPriority.HIGH,
            ]:
                continue

            try:
                if rec.area == OptimizationArea.TEMPLATE_CACHE:
                    await self._apply_cache_optimization(rec, dry_run=dry_run)
                    applied_count += 1

                elif rec.area == OptimizationArea.PARALLEL_EXECUTION:
                    self._apply_parallel_optimization(rec, dry_run=dry_run)
                    applied_count += 1

                elif rec.area == OptimizationArea.MEMORY:
                    self._apply_memory_optimization(rec, dry_run=dry_run)
                    applied_count += 1

                elif rec.area == OptimizationArea.IO:
                    self._apply_io_optimization(rec, dry_run=dry_run)
                    applied_count += 1

                # Track applied optimization (not in dry-run mode)
                if not dry_run:
                    self._applied_optimizations.append(rec)

            except Exception as e:
                logger.warning(
                    f"Failed to {mode_str.lower()} optimization for {rec.area.value}: {e}"
                )

        action = "Simulated" if dry_run else "Applied"
        logger.info(f"{action} {applied_count} automatic optimizations")

        if self.metrics and not dry_run:
            await self.metrics.record_counter(
                "optimizer.optimizations_applied", applied_count
            )

    async def _apply_cache_optimization(
        self, rec: OptimizationRecommendation, dry_run: bool = False
    ) -> None:
        """Apply template cache optimization."""
        if not self.template_manager:
            logger.warning("TemplateManager not available for cache optimization")
            return

        mode = "Simulating" if dry_run else "Applying"
        logger.info(f"{mode} cache optimization: {rec.recommendation}")

        # Preload frequently used templates
        await self.optimize_template_cache(
            target_hit_rate=rec.target_value or self.target_cache_hit_rate,
            dry_run=dry_run,
        )

    def _apply_parallel_optimization(
        self, rec: OptimizationRecommendation, dry_run: bool = False
    ) -> None:
        """Apply parallel execution optimization."""
        if not self.staged_executor:
            logger.warning(
                "StagedParallelExecutor not available for parallel optimization"
            )
            return

        mode = "Simulating" if dry_run else "Applying"
        logger.info(f"{mode} parallel optimization: {rec.recommendation}")

        # Tune parallel execution parameters
        self.tune_parallel_execution(
            target_speedup=rec.target_value or self.target_parallel_speedup,
            dry_run=dry_run,
        )

    def _apply_memory_optimization(
        self, rec: OptimizationRecommendation, dry_run: bool = False
    ) -> None:
        """Apply memory optimization."""
        mode = "Simulating" if dry_run else "Applying"
        logger.info(f"{mode} memory optimization: {rec.recommendation}")

        # Memory optimizations are mostly code-level changes
        # Log recommendation for manual implementation
        logger.info(f"Memory optimization recommendation: {rec.implementation_notes}")

        # Track dry-run change
        if dry_run:
            self._dry_run_changes.append(
                DryRunChange(
                    area=OptimizationArea.MEMORY,
                    change_type="code_refactoring",
                    description=rec.recommendation,
                    current_value=(
                        f"{rec.current_value}MB" if rec.current_value else "N/A"
                    ),
                    proposed_value=(
                        f"{rec.target_value}MB" if rec.target_value else "N/A"
                    ),
                    expected_impact=rec.expected_improvement,
                    affected_components=rec.related_operations,
                    recommendation=rec,
                )
            )

    def _apply_io_optimization(
        self, rec: OptimizationRecommendation, dry_run: bool = False
    ) -> None:
        """Apply I/O optimization."""
        mode = "Simulating" if dry_run else "Applying"
        logger.info(f"{mode} I/O optimization: {rec.recommendation}")

        # I/O optimizations are mostly code-level changes
        # Log recommendation for manual implementation
        logger.info(f"I/O optimization recommendation: {rec.implementation_notes}")

        # Track dry-run change
        if dry_run:
            self._dry_run_changes.append(
                DryRunChange(
                    area=OptimizationArea.IO,
                    change_type="async_conversion",
                    description=rec.recommendation,
                    current_value=(
                        f"{rec.current_value:.1%}" if rec.current_value else "N/A"
                    ),
                    proposed_value=(
                        f"{rec.target_value:.1%}" if rec.target_value else "N/A"
                    ),
                    expected_impact=rec.expected_improvement,
                    affected_components=rec.related_operations,
                    recommendation=rec,
                )
            )

    async def optimize_template_cache(
        self, target_hit_rate: float = 0.95, dry_run: bool = False
    ) -> None:
        """
        Optimize template cache for target hit rate.

        Preloads frequently used templates based on usage patterns.

        Args:
            target_hit_rate: Target cache hit rate (default: 0.95)
            dry_run: If True, simulate changes without applying them

        Example:
            ```python
            # Simulate optimization
            await optimizer.optimize_template_cache(target_hit_rate=0.95, dry_run=True)

            # Actually apply optimization
            await optimizer.optimize_template_cache(target_hit_rate=0.95, dry_run=False)
            ```
        """
        if not self.template_manager:
            logger.warning("TemplateManager not available")
            return

        mode = "Simulating" if dry_run else "Optimizing"
        logger.info(f"{mode} template cache (target hit rate: {target_hit_rate:.1%})")

        # Get current cache stats
        cache_stats = self.template_manager.get_cache_stats()

        if cache_stats.hit_rate >= target_hit_rate:
            logger.info(f"Cache already optimized: {cache_stats.hit_rate:.1%} hit rate")
            return

        # Preload standard templates for all node types
        templates_to_preload = [
            ("node_effect_v1", TemplateType.EFFECT),
            ("node_compute_v1", TemplateType.COMPUTE),
            ("node_reducer_v1", TemplateType.REDUCER),
            ("node_orchestrator_v1", TemplateType.ORCHESTRATOR),
            ("validator_v1", TemplateType.VALIDATOR),
            ("test_v1", TemplateType.TEST),
            ("contract_v1", TemplateType.CONTRACT),
        ]

        preloaded = 0
        if dry_run:
            # Simulate preloading - just count what we would preload
            preloaded = len(templates_to_preload)
            logger.info(
                f"[DRY-RUN] Would preload {preloaded} templates to improve hit rate "
                f"from {cache_stats.hit_rate:.1%} to ~{target_hit_rate:.1%}"
            )

            # Track dry-run change
            self._dry_run_changes.append(
                DryRunChange(
                    area=OptimizationArea.TEMPLATE_CACHE,
                    change_type="preload_templates",
                    description=f"Preload {preloaded} standard templates",
                    current_value=f"{cache_stats.hit_rate:.1%}",
                    proposed_value=f"~{target_hit_rate:.1%}",
                    expected_impact=f"Reduce template load time by {(1 - cache_stats.hit_rate) * 100:.0f}%",
                    affected_components=[
                        f"{t[0]} ({t[1].value})" for t in templates_to_preload
                    ],
                )
            )
        else:
            # Actually preload templates
            for template_id, template_type in templates_to_preload:
                try:
                    # Load template to cache it
                    await self.template_manager.load_template(
                        template_id=template_id, template_type=template_type
                    )
                    preloaded += 1
                except Exception as e:
                    logger.debug(f"Could not preload {template_id}: {e}")

            logger.info(f"Preloaded {preloaded} templates")

        # Record optimization
        if self.metrics and not dry_run:
            await self.metrics.record_counter(
                "optimizer.templates_preloaded", preloaded
            )
            await self.metrics.record_gauge(
                "optimizer.cache_hit_rate_target", target_hit_rate
            )

    def tune_parallel_execution(
        self,
        target_speedup: float = None,
        target_concurrency: int = None,
        dry_run: bool = False,
    ) -> None:
        """
        Tune parallel execution parameters.

        Adjusts max_concurrent_tasks and batching strategy.

        Args:
            target_speedup: Target speedup factor (default: 3.5x)
            target_concurrency: Target max concurrent tasks (auto if None)
            dry_run: If True, simulate changes without applying them

        Example:
            ```python
            # Simulate tuning
            optimizer.tune_parallel_execution(target_speedup=4.0, dry_run=True)

            # Actually apply tuning
            optimizer.tune_parallel_execution(target_concurrency=16, dry_run=False)
            ```
        """
        if not self.staged_executor:
            logger.warning("StagedParallelExecutor not available")
            return

        target_speedup = target_speedup or self.target_parallel_speedup

        mode = "Simulating" if dry_run else "Tuning"
        logger.info(
            f"{mode} parallel execution (target speedup: {target_speedup:.1f}x)"
        )

        # Calculate optimal concurrency if not provided
        if target_concurrency is None:
            # Use CPU count as baseline
            import os

            cpu_count = os.cpu_count() or 4
            # Target concurrency = CPU count * 1.5 for I/O-bound tasks
            target_concurrency = int(cpu_count * 1.5)

        # Update executor configuration (or simulate)
        if hasattr(self.staged_executor, "max_concurrent_tasks"):
            old_value = getattr(self.staged_executor, "max_concurrent_tasks", 10)

            if dry_run:
                logger.info(
                    f"[DRY-RUN] Would update max_concurrent_tasks: {old_value} -> {target_concurrency}"
                )

                # Track dry-run change
                self._dry_run_changes.append(
                    DryRunChange(
                        area=OptimizationArea.PARALLEL_EXECUTION,
                        change_type="concurrency_tuning",
                        description=f"Increase max concurrent tasks to {target_concurrency}",
                        current_value=old_value,
                        proposed_value=target_concurrency,
                        expected_impact=f"Improve parallelization speedup to ~{target_speedup:.1f}x",
                        affected_components=["StagedParallelExecutor"],
                    )
                )
            else:
                self.staged_executor.max_concurrent_tasks = target_concurrency
                logger.info(
                    f"Updated max_concurrent_tasks: {old_value} -> {target_concurrency}"
                )

        # Record optimization
        if self.metrics and not dry_run:
            asyncio.create_task(
                self.metrics.record_gauge(
                    "optimizer.max_concurrent_tasks", target_concurrency
                )
            )
            asyncio.create_task(
                self.metrics.record_gauge(
                    "optimizer.parallel_speedup_target", target_speedup
                )
            )

    def optimize_memory_usage(self, target_overhead_mb: float = None) -> list[str]:
        """
        Generate memory optimization recommendations.

        Returns list of actionable recommendations for memory reduction.

        Args:
            target_overhead_mb: Target memory overhead (default: 50MB)

        Returns:
            List of memory optimization recommendations

        Example:
            ```python
            recommendations = optimizer.optimize_memory_usage(target_overhead_mb=50.0)
            for rec in recommendations:
                print(rec)
            ```
        """
        target_overhead_mb = target_overhead_mb or self.target_memory_overhead_mb

        logger.info(
            f"Generating memory optimization recommendations "
            f"(target overhead: {target_overhead_mb:.0f}MB)"
        )

        recommendations = [
            "Use generators instead of lists for large datasets",
            "Implement object pooling for frequently created objects",
            "Enable streaming for large file operations",
            "Reduce data copying by using references where possible",
            "Clear intermediate results after processing",
            "Use __slots__ in frequently instantiated classes",
            "Profile with tracemalloc to identify memory leaks",
        ]

        return recommendations

    def optimize_io_operations(self, target_async_ratio: float = None) -> list[str]:
        """
        Generate I/O optimization recommendations.

        Returns list of actionable recommendations for I/O optimization.

        Args:
            target_async_ratio: Target async I/O ratio (default: 0.8)

        Returns:
            List of I/O optimization recommendations

        Example:
            ```python
            recommendations = optimizer.optimize_io_operations(target_async_ratio=0.8)
            for rec in recommendations:
                print(rec)
            ```
        """
        target_async_ratio = target_async_ratio or self.target_io_async_ratio

        logger.info(
            f"Generating I/O optimization recommendations "
            f"(target async ratio: {target_async_ratio:.1%})"
        )

        recommendations = [
            "Convert synchronous file I/O to async using aiofiles",
            "Batch small I/O operations to reduce syscall overhead",
            "Use async HTTP client for network requests",
            "Implement connection pooling for database operations",
            "Cache frequently read files in memory",
            "Use memory-mapped files for large read-only data",
            "Parallelize independent I/O operations",
        ]

        return recommendations

    def get_optimization_summary(self) -> dict[str, Any]:
        """
        Get summary of applied optimizations.

        Returns:
            Dictionary with optimization summary statistics

        Example:
            ```python
            summary = optimizer.get_optimization_summary()
            print(f"Applied {summary['total_optimizations']} optimizations")
            print(f"Average speedup: {summary['avg_speedup']:.2f}x")
            ```
        """
        total_optimizations = len(self._applied_optimizations)

        # Count by area
        by_area = {}
        for opt in self._applied_optimizations:
            area = opt.area.value
            by_area[area] = by_area.get(area, 0) + 1

        # Calculate average estimated speedup from reports
        avg_speedup = 1.0
        if self._optimization_history:
            avg_speedup = sum(
                r.estimated_speedup for r in self._optimization_history
            ) / len(self._optimization_history)

        return {
            "total_optimizations": total_optimizations,
            "optimizations_by_area": by_area,
            "total_reports": len(self._optimization_history),
            "avg_speedup": avg_speedup,
            "latest_report": (
                self._optimization_history[-1] if self._optimization_history else None
            ),
        }

    def reset(self) -> None:
        """Reset optimization history."""
        self._applied_optimizations.clear()
        self._optimization_history.clear()
        self._dry_run_changes.clear()
        self._current_dry_run_report = None
        logger.info("Optimizer history reset")

    def _generate_dry_run_report(
        self, workflow_id: str, estimated_speedup: float
    ) -> DryRunReport:
        """
        Generate dry-run report from collected changes.

        Args:
            workflow_id: Workflow identifier
            estimated_speedup: Estimated speedup from optimizations

        Returns:
            DryRunReport with detailed proposed changes
        """
        # Separate changes by priority
        critical_changes = [
            change
            for change in self._dry_run_changes
            if change.recommendation
            and change.recommendation.priority == OptimizationPriority.CRITICAL
        ]

        high_priority_changes = [
            change
            for change in self._dry_run_changes
            if change.recommendation
            and change.recommendation.priority == OptimizationPriority.HIGH
        ]

        return DryRunReport(
            workflow_id=workflow_id,
            total_changes=len(self._dry_run_changes),
            changes=self._dry_run_changes.copy(),
            critical_changes=critical_changes,
            high_priority_changes=high_priority_changes,
            estimated_speedup=estimated_speedup,
            metadata={
                "auto_optimize": self.auto_optimize,
                "dry_run": self.dry_run,
                "target_cache_hit_rate": self.target_cache_hit_rate,
                "target_parallel_speedup": self.target_parallel_speedup,
                "target_memory_overhead_mb": self.target_memory_overhead_mb,
                "target_io_async_ratio": self.target_io_async_ratio,
            },
        )

    def get_dry_run_report(self) -> Optional[DryRunReport]:
        """
        Get the most recent dry-run report.

        Returns:
            DryRunReport if available, None otherwise

        Example:
            ```python
            # Run in dry-run mode
            await optimizer.optimize_workflow(
                workflow_id="test",
                dry_run=True,
                apply_optimizations=True
            )

            # Get the dry-run report
            report = optimizer.get_dry_run_report()
            if report:
                print(report.get_summary())
                # Export to JSON
                json_data = report.export_json()
            ```
        """
        return self._current_dry_run_report
