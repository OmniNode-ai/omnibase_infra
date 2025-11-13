"""
Performance profiling for code generation workflows.

Provides hot path identification, cache analysis, and bottleneck detection
to guide optimization efforts. Designed for <5% profiling overhead.

Performance Targets:
- Profiling overhead: <5%
- Hot path identification: Bottlenecks >20% of total time
- Percentile calculation: p50, p95, p99
- Memory profiling: Peak and average usage
"""

import asyncio
import gc
import logging
import statistics
import time
import tracemalloc
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Optional

from omninode_bridge.agents.metrics.collector import MetricsCollector

from .optimization_models import (
    IOPerformanceStats,
    MemoryUsageStats,
    ParallelExecutionStats,
    PerformanceReport,
    ProfileResult,
    TemplateCacheStats,
)

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Performance profiler for code generation workflows.

    Provides comprehensive profiling with minimal overhead (<5%):
    - Hot path identification (bottlenecks >20% of total time)
    - Timing analysis with percentiles (p50, p95, p99)
    - Memory profiling (peak and average usage)
    - I/O operation tracking
    - Cache efficiency analysis

    Features:
    - Context manager for automatic profiling
    - Async-friendly profiling
    - Low overhead timing (<5% impact)
    - Bottleneck score calculation
    - Integration with MetricsCollector

    Example:
        ```python
        profiler = PerformanceProfiler(metrics_collector=metrics)

        # Profile single operation
        async with profiler.profile_operation("template_render"):
            result = await render_template(template_id)

        # Profile full workflow
        profile_data = await profiler.profile_workflow(
            workflow_func=execute_codegen_workflow,
            iterations=10
        )

        # Identify bottlenecks
        hot_paths = profiler.analyze_hot_paths(profile_data)
        for hp in hot_paths:
            print(f"{hp.operation_name}: {hp.bottleneck_score:.1%} of total time")
        ```
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        enable_memory_profiling: bool = False,
        enable_io_profiling: bool = True,
    ) -> None:
        """
        Initialize performance profiler.

        Args:
            metrics_collector: Optional metrics collector for integration
            enable_memory_profiling: Enable memory profiling (adds overhead)
            enable_io_profiling: Enable I/O operation tracking
        """
        self.metrics = metrics_collector
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_io_profiling = enable_io_profiling

        # Profile data storage
        self.profile_data: dict[str, list[ProfileResult]] = {}
        self._operation_timings: dict[str, list[float]] = {}
        self._operation_memory: dict[str, list[float]] = {}

        # I/O tracking
        self._io_operations: int = 0
        self._sync_io_count: int = 0
        self._async_io_count: int = 0
        self._total_io_time_ms: float = 0.0

        # Cache tracking
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_cache_requests: int = 0

        # Parallel execution tracking
        self._parallel_operations: int = 0
        self._total_operations: int = 0
        self._sequential_time_ms: float = 0.0
        self._parallel_time_ms: float = 0.0

        logger.info(
            f"PerformanceProfiler initialized: "
            f"memory_profiling={enable_memory_profiling}, "
            f"io_profiling={enable_io_profiling}"
        )

    @asynccontextmanager
    async def profile_operation(
        self,
        operation_name: str,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Context manager for profiling a single operation.

        Args:
            operation_name: Name of the operation being profiled
            metadata: Optional metadata to attach to profile result

        Example:
            ```python
            async with profiler.profile_operation("parse_contract"):
                contract = await parse_contract(yaml_content)
            ```
        """
        start_time = time.perf_counter()
        start_memory = 0.0

        # Start memory tracking if enabled
        if self.enable_memory_profiling:
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

        try:
            yield

        finally:
            # Calculate elapsed time
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Track memory usage if enabled
            memory_mb = 0.0
            if self.enable_memory_profiling and tracemalloc.is_tracing():
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                memory_mb = (peak_memory - start_memory * 1024 * 1024) / 1024 / 1024
                tracemalloc.stop()

            # Store timing data
            if operation_name not in self._operation_timings:
                self._operation_timings[operation_name] = []
            self._operation_timings[operation_name].append(elapsed_ms)

            # Store memory data
            if self.enable_memory_profiling:
                if operation_name not in self._operation_memory:
                    self._operation_memory[operation_name] = []
                self._operation_memory[operation_name].append(memory_mb)

            # Record metric
            if self.metrics:
                await self.metrics.record_timing(
                    f"profile.{operation_name}.duration_ms", elapsed_ms
                )
                if self.enable_memory_profiling:
                    await self.metrics.record_gauge(
                        f"profile.{operation_name}.memory_mb", memory_mb
                    )

            self._total_operations += 1

    async def profile_workflow(
        self,
        workflow_func: Callable,
        iterations: int = 10,
        *args,
        **kwargs,
    ) -> dict[str, ProfileResult]:
        """
        Profile a workflow function multiple times and collect statistics.

        Args:
            workflow_func: Async function to profile
            iterations: Number of profiling iterations
            *args: Arguments to pass to workflow function
            **kwargs: Keyword arguments to pass to workflow function

        Returns:
            Dictionary mapping operation names to ProfileResult objects

        Example:
            ```python
            async def my_workflow():
                async with profiler.profile_operation("step1"):
                    await step1()
                async with profiler.profile_operation("step2"):
                    await step2()

            results = await profiler.profile_workflow(my_workflow, iterations=10)
            ```
        """
        logger.info(f"Profiling workflow for {iterations} iterations")

        # Clear previous data
        self._operation_timings.clear()
        self._operation_memory.clear()

        # Run workflow multiple times
        total_start = time.perf_counter()

        for i in range(iterations):
            # Force garbage collection between iterations for consistent results
            gc.collect()

            # Run workflow
            if asyncio.iscoroutinefunction(workflow_func):
                await workflow_func(*args, **kwargs)
            else:
                workflow_func(*args, **kwargs)

        total_elapsed = (time.perf_counter() - total_start) * 1000

        # Calculate statistics for each operation
        profile_results = {}
        total_time_all_ops = 0.0

        for op_name, timings in self._operation_timings.items():
            if not timings:
                continue

            # Calculate timing statistics
            total_time = sum(timings)
            call_count = len(timings)
            avg_time = statistics.mean(timings)
            p50 = statistics.median(timings)

            # Calculate percentiles
            sorted_timings = sorted(timings)
            p95_idx = int(len(sorted_timings) * 0.95)
            p99_idx = int(len(sorted_timings) * 0.99)
            p95 = (
                sorted_timings[p95_idx]
                if p95_idx < len(sorted_timings)
                else max(timings)
            )
            p99 = (
                sorted_timings[p99_idx]
                if p99_idx < len(sorted_timings)
                else max(timings)
            )

            # Memory statistics
            memory_mb = 0.0
            if self._operation_memory.get(op_name):
                memory_mb = max(self._operation_memory[op_name])

            total_time_all_ops += total_time

            profile_results[op_name] = ProfileResult(
                operation_name=op_name,
                total_time_ms=total_time,
                call_count=call_count,
                avg_time_ms=avg_time,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                memory_mb=memory_mb,
                bottleneck_score=0.0,  # Will be calculated below
            )

        # Calculate bottleneck scores (percentage of total time)
        if total_time_all_ops > 0:
            for result in profile_results.values():
                result.bottleneck_score = result.total_time_ms / total_time_all_ops

        # Store results
        self.profile_data[workflow_func.__name__] = list(profile_results.values())

        logger.info(
            f"Profiling complete: {len(profile_results)} operations profiled "
            f"in {total_elapsed:.2f}ms ({iterations} iterations)"
        )

        return profile_results

    def analyze_hot_paths(
        self,
        profile_data: dict[str, ProfileResult],
        threshold: float = 0.2,
    ) -> list[ProfileResult]:
        """
        Identify hot paths (bottlenecks) from profile data.

        Hot paths are operations that consume >threshold of total time.

        Args:
            profile_data: Profile results from profile_workflow()
            threshold: Bottleneck threshold (default: 0.2 = 20%)

        Returns:
            List of ProfileResult objects sorted by bottleneck_score descending

        Example:
            ```python
            hot_paths = profiler.analyze_hot_paths(profile_data, threshold=0.2)
            for hp in hot_paths:
                print(f"{hp.operation_name}: {hp.bottleneck_score:.1%}")
            ```
        """
        # Filter bottlenecks
        bottlenecks = [
            result
            for result in profile_data.values()
            if result.bottleneck_score >= threshold
        ]

        # Sort by bottleneck score descending
        bottlenecks.sort(key=lambda x: x.bottleneck_score, reverse=True)

        logger.info(
            f"Identified {len(bottlenecks)} hot paths (threshold: {threshold:.1%})"
        )

        return bottlenecks

    def track_cache_access(
        self,
        hit: bool,
        access_time_ms: float,
    ) -> None:
        """
        Track template cache access for statistics.

        Args:
            hit: Whether cache hit occurred
            access_time_ms: Time taken for cache access
        """
        self._total_cache_requests += 1

        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def track_io_operation(
        self,
        is_async: bool,
        duration_ms: float,
    ) -> None:
        """
        Track I/O operation for statistics.

        Args:
            is_async: Whether operation was async
            duration_ms: Duration of I/O operation
        """
        if not self.enable_io_profiling:
            return

        self._io_operations += 1
        self._total_io_time_ms += duration_ms

        if is_async:
            self._async_io_count += 1
        else:
            self._sync_io_count += 1

    def track_parallel_execution(
        self,
        sequential_time_ms: float,
        parallel_time_ms: float,
        operation_count: int,
    ) -> None:
        """
        Track parallel execution statistics.

        Args:
            sequential_time_ms: Estimated sequential execution time
            parallel_time_ms: Actual parallel execution time
            operation_count: Number of parallel operations
        """
        self._parallel_operations += operation_count
        self._sequential_time_ms += sequential_time_ms
        self._parallel_time_ms += parallel_time_ms

    def get_template_cache_stats(
        self,
        cache_size: int = 0,
        max_cache_size: int = 100,
        eviction_count: int = 0,
        preloaded_templates: int = 0,
    ) -> TemplateCacheStats:
        """
        Get template cache statistics.

        Args:
            cache_size: Current cache size
            max_cache_size: Maximum cache size
            eviction_count: Number of evictions
            preloaded_templates: Number of preloaded templates

        Returns:
            TemplateCacheStats object
        """
        hit_rate = (
            self._cache_hits / self._total_cache_requests
            if self._total_cache_requests > 0
            else 0.0
        )

        return TemplateCacheStats(
            total_requests=self._total_cache_requests,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            hit_rate=hit_rate,
            cache_size=cache_size,
            max_cache_size=max_cache_size,
            eviction_count=eviction_count,
            preloaded_templates=preloaded_templates,
        )

    def get_parallel_execution_stats(
        self,
        max_concurrent_tasks: int = 10,
        avg_concurrent_tasks: float = 1.0,
        cpu_utilization: float = 0.0,
        coordination_overhead_ms: float = 0.0,
    ) -> ParallelExecutionStats:
        """
        Get parallel execution statistics.

        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            avg_concurrent_tasks: Average concurrent tasks
            cpu_utilization: CPU utilization ratio (0-1)
            coordination_overhead_ms: Coordination overhead

        Returns:
            ParallelExecutionStats object
        """
        speedup = (
            self._sequential_time_ms / self._parallel_time_ms
            if self._parallel_time_ms > 0
            else 1.0
        )

        return ParallelExecutionStats(
            total_operations=self._total_operations,
            parallel_operations=self._parallel_operations,
            sequential_time_ms=self._sequential_time_ms,
            parallel_time_ms=self._parallel_time_ms,
            speedup_factor=speedup,
            max_concurrent_tasks=max_concurrent_tasks,
            avg_concurrent_tasks=avg_concurrent_tasks,
            cpu_utilization=cpu_utilization,
            coordination_overhead_ms=coordination_overhead_ms,
        )

    def get_memory_stats(self) -> MemoryUsageStats:
        """
        Get memory usage statistics.

        Returns:
            MemoryUsageStats object
        """
        # Calculate memory statistics from collected data
        all_memory = []
        for memory_list in self._operation_memory.values():
            all_memory.extend(memory_list)

        peak_memory = max(all_memory) if all_memory else 0.0
        avg_memory = statistics.mean(all_memory) if all_memory else 0.0

        # Count large allocations (>10MB)
        large_allocations = sum(1 for m in all_memory if m > 10.0)

        return MemoryUsageStats(
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            memory_allocations=len(all_memory),
            large_allocations=large_allocations,
            memory_overhead_mb=peak_memory,  # Simplified for now
        )

    def get_io_stats(self) -> IOPerformanceStats:
        """
        Get I/O performance statistics.

        Returns:
            IOPerformanceStats object
        """
        avg_io_time = (
            self._total_io_time_ms / self._io_operations
            if self._io_operations > 0
            else 0.0
        )

        return IOPerformanceStats(
            total_io_operations=self._io_operations,
            sync_io_operations=self._sync_io_count,
            async_io_operations=self._async_io_count,
            total_io_time_ms=self._total_io_time_ms,
            avg_io_time_ms=avg_io_time,
        )

    def generate_performance_report(
        self,
        workflow_id: str,
        profile_data: dict[str, ProfileResult],
        **kwargs,
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            workflow_id: Workflow identifier
            profile_data: Profile results from profile_workflow()
            **kwargs: Additional arguments for stats methods

        Returns:
            PerformanceReport with analysis and recommendations

        Example:
            ```python
            report = profiler.generate_performance_report(
                workflow_id="codegen-session-1",
                profile_data=profile_data,
                cache_size=50,
                max_cache_size=100
            )

            print(report.get_summary())
            for rec in report.critical_recommendations:
                print(f"CRITICAL: {rec.issue}")
            ```
        """
        # Calculate total duration
        total_duration = sum(r.total_time_ms for r in profile_data.values())

        # Identify hot paths
        hot_paths = self.analyze_hot_paths(profile_data)

        # Collect statistics
        cache_stats = self.get_template_cache_stats(
            cache_size=kwargs.get("cache_size", 0),
            max_cache_size=kwargs.get("max_cache_size", 100),
            eviction_count=kwargs.get("eviction_count", 0),
            preloaded_templates=kwargs.get("preloaded_templates", 0),
        )

        parallel_stats = self.get_parallel_execution_stats(
            max_concurrent_tasks=kwargs.get("max_concurrent_tasks", 10),
            avg_concurrent_tasks=kwargs.get("avg_concurrent_tasks", 1.0),
            cpu_utilization=kwargs.get("cpu_utilization", 0.0),
            coordination_overhead_ms=kwargs.get("coordination_overhead_ms", 0.0),
        )

        memory_stats = self.get_memory_stats()
        io_stats = self.get_io_stats()

        # Generate recommendations
        recommendations = []

        # Cache optimization
        cache_rec = cache_stats.get_optimization_recommendation()
        if cache_rec:
            recommendations.append(cache_rec)

        # Parallel execution optimization
        parallel_rec = parallel_stats.get_optimization_recommendation()
        if parallel_rec:
            recommendations.append(parallel_rec)

        # Memory optimization
        memory_rec = memory_stats.get_optimization_recommendation()
        if memory_rec:
            recommendations.append(memory_rec)

        # I/O optimization
        io_rec = io_stats.get_optimization_recommendation()
        if io_rec:
            recommendations.append(io_rec)

        # Sort recommendations by priority
        recommendations.sort(
            key=lambda r: (
                r.priority.value,
                -r.current_value if r.current_value else 0,
            )
        )

        return PerformanceReport(
            workflow_id=workflow_id,
            total_duration_ms=total_duration,
            profile_results=profile_data,
            hot_paths=hot_paths,
            recommendations=recommendations,
            template_cache_stats=cache_stats,
            parallel_execution_stats=parallel_stats,
            memory_stats=memory_stats,
            io_stats=io_stats,
        )

    def reset(self) -> None:
        """Reset all profiling data."""
        self.profile_data.clear()
        self._operation_timings.clear()
        self._operation_memory.clear()
        self._io_operations = 0
        self._sync_io_count = 0
        self._async_io_count = 0
        self._total_io_time_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_cache_requests = 0
        self._parallel_operations = 0
        self._total_operations = 0
        self._sequential_time_ms = 0.0
        self._parallel_time_ms = 0.0

        logger.info("Profiler data reset")
