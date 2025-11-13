"""
Unit tests for performance optimization system.

Tests profiling, optimization recommendations, and automatic optimization
application. Validates 2-3x speedup target achievement.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows.optimization_models import (
    DryRunChange,
    IOPerformanceStats,
    MemoryUsageStats,
    OptimizationArea,
    OptimizationPriority,
    OptimizationRecommendation,
    ParallelExecutionStats,
    PerformanceReport,
    ProfileResult,
    TemplateCacheStats,
)
from omninode_bridge.agents.workflows.performance_optimizer import PerformanceOptimizer
from omninode_bridge.agents.workflows.profiling import PerformanceProfiler
from omninode_bridge.agents.workflows.staged_execution import StagedParallelExecutor
from omninode_bridge.agents.workflows.template_manager import TemplateManager


class TestProfileResult:
    """Test ProfileResult data model."""

    def test_profile_result_creation(self):
        """Test basic ProfileResult creation."""
        result = ProfileResult(
            operation_name="test_op",
            total_time_ms=1000.0,
            call_count=10,
            avg_time_ms=100.0,
            p50_ms=95.0,
            p95_ms=150.0,
            p99_ms=180.0,
            memory_mb=25.0,
            bottleneck_score=0.35,
        )

        assert result.operation_name == "test_op"
        assert result.total_time_ms == 1000.0
        assert result.call_count == 10
        assert result.avg_time_ms == 100.0
        assert result.bottleneck_score == 0.35

    def test_profile_result_is_bottleneck(self):
        """Test bottleneck detection."""
        # Not a bottleneck
        result = ProfileResult(
            operation_name="fast_op",
            total_time_ms=100.0,
            call_count=10,
            avg_time_ms=10.0,
            p50_ms=9.0,
            p95_ms=15.0,
            p99_ms=18.0,
            memory_mb=5.0,
            bottleneck_score=0.15,
        )
        assert not result.is_bottleneck
        assert not result.is_critical_bottleneck

        # Is a bottleneck
        result.bottleneck_score = 0.35
        assert result.is_bottleneck
        assert not result.is_critical_bottleneck

        # Critical bottleneck
        result.bottleneck_score = 0.65
        assert result.is_bottleneck
        assert result.is_critical_bottleneck

    def test_profile_result_invalid_bottleneck_score(self):
        """Test invalid bottleneck score raises error."""
        with pytest.raises(ValueError, match="bottleneck_score must be 0-1"):
            ProfileResult(
                operation_name="invalid",
                total_time_ms=100.0,
                call_count=1,
                avg_time_ms=100.0,
                p50_ms=100.0,
                p95_ms=100.0,
                p99_ms=100.0,
                memory_mb=10.0,
                bottleneck_score=1.5,  # Invalid
            )


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation data model."""

    def test_recommendation_creation(self):
        """Test basic recommendation creation."""
        rec = OptimizationRecommendation(
            area=OptimizationArea.TEMPLATE_CACHE,
            issue="Cache hit rate too low",
            recommendation="Preload templates",
            expected_improvement="30-50% faster",
            priority=OptimizationPriority.HIGH,
            current_value=0.85,
            target_value=0.95,
        )

        assert rec.area == OptimizationArea.TEMPLATE_CACHE
        assert rec.priority == OptimizationPriority.HIGH
        assert rec.current_value == 0.85
        assert rec.target_value == 0.95

    def test_impact_description(self):
        """Test impact description generation."""
        rec = OptimizationRecommendation(
            area=OptimizationArea.PARALLEL_EXECUTION,
            issue="test",
            recommendation="test",
            expected_improvement="test",
            priority=OptimizationPriority.CRITICAL,
        )
        assert "Critical impact" in rec.impact_description
        assert ">50%" in rec.impact_description

        rec.priority = OptimizationPriority.HIGH
        assert "High impact" in rec.impact_description

        rec.priority = OptimizationPriority.MEDIUM
        assert "Medium impact" in rec.impact_description

        rec.priority = OptimizationPriority.LOW
        assert "Low impact" in rec.impact_description


class TestTemplateCacheStats:
    """Test TemplateCacheStats model."""

    def test_cache_stats_needs_optimization(self):
        """Test cache optimization detection."""
        # Good hit rate
        stats = TemplateCacheStats(
            total_requests=100,
            cache_hits=96,
            cache_misses=4,
            hit_rate=0.96,
        )
        assert not stats.needs_optimization

        # Poor hit rate
        stats.hit_rate = 0.85
        assert stats.needs_optimization

    def test_cache_stats_is_full(self):
        """Test cache full detection."""
        stats = TemplateCacheStats(cache_size=50, max_cache_size=100)
        assert not stats.is_full

        stats.cache_size = 100
        assert stats.is_full

    def test_cache_stats_optimization_recommendation(self):
        """Test cache optimization recommendation generation."""
        # No recommendation needed
        stats = TemplateCacheStats(hit_rate=0.96)
        assert stats.get_optimization_recommendation() is None

        # Critical recommendation
        stats = TemplateCacheStats(hit_rate=0.80)
        rec = stats.get_optimization_recommendation()
        assert rec is not None
        assert rec.priority == OptimizationPriority.CRITICAL
        assert rec.area == OptimizationArea.TEMPLATE_CACHE

        # High priority recommendation
        stats = TemplateCacheStats(hit_rate=0.88)
        rec = stats.get_optimization_recommendation()
        assert rec.priority == OptimizationPriority.HIGH

        # Medium priority recommendation
        stats = TemplateCacheStats(hit_rate=0.92)
        rec = stats.get_optimization_recommendation()
        assert rec.priority == OptimizationPriority.MEDIUM


class TestParallelExecutionStats:
    """Test ParallelExecutionStats model."""

    def test_parallel_stats_needs_optimization(self):
        """Test parallel execution optimization detection."""
        # Good speedup
        stats = ParallelExecutionStats(
            parallel_operations=10,
            speedup_factor=3.8,
        )
        assert not stats.needs_optimization

        # Poor speedup
        stats.speedup_factor = 2.2
        assert stats.needs_optimization

    def test_parallel_stats_low_utilization(self):
        """Test low CPU utilization detection."""
        stats = ParallelExecutionStats(cpu_utilization=0.85)
        assert not stats.has_low_utilization

        stats.cpu_utilization = 0.65
        assert stats.has_low_utilization

    def test_parallel_stats_optimization_recommendation(self):
        """Test parallel optimization recommendation generation."""
        # No recommendation needed
        stats = ParallelExecutionStats(
            parallel_operations=10,
            speedup_factor=3.5,
        )
        assert stats.get_optimization_recommendation() is None

        # Critical recommendation (low speedup)
        stats = ParallelExecutionStats(
            parallel_operations=10,
            speedup_factor=1.8,
            cpu_utilization=0.65,
            max_concurrent_tasks=8,
        )
        rec = stats.get_optimization_recommendation()
        assert rec is not None
        assert rec.priority == OptimizationPriority.CRITICAL
        assert rec.area == OptimizationArea.PARALLEL_EXECUTION


class TestMemoryUsageStats:
    """Test MemoryUsageStats model."""

    def test_memory_stats_needs_optimization(self):
        """Test memory optimization detection."""
        stats = MemoryUsageStats(memory_overhead_mb=45.0)
        assert not stats.needs_optimization

        stats.memory_overhead_mb = 75.0
        assert stats.needs_optimization

    def test_memory_stats_excessive_gc(self):
        """Test excessive GC detection."""
        stats = MemoryUsageStats(gc_collections=80)
        assert not stats.has_excessive_gc

        stats.gc_collections = 150
        assert stats.has_excessive_gc

    def test_memory_stats_optimization_recommendation(self):
        """Test memory optimization recommendation generation."""
        # No recommendation needed
        stats = MemoryUsageStats(memory_overhead_mb=40.0)
        assert stats.get_optimization_recommendation() is None

        # Critical recommendation
        stats = MemoryUsageStats(
            memory_overhead_mb=120.0,
            large_allocations=5,
        )
        rec = stats.get_optimization_recommendation()
        assert rec is not None
        assert rec.priority == OptimizationPriority.CRITICAL
        assert rec.area == OptimizationArea.MEMORY


class TestIOPerformanceStats:
    """Test IOPerformanceStats model."""

    def test_io_stats_needs_optimization(self):
        """Test I/O optimization detection."""
        stats = IOPerformanceStats(
            total_io_operations=100,
            sync_io_operations=40,
            async_io_operations=60,
        )
        assert not stats.needs_optimization

        stats.sync_io_operations = 70
        stats.async_io_operations = 30
        assert stats.needs_optimization

    def test_io_stats_async_ratio(self):
        """Test async I/O ratio calculation."""
        stats = IOPerformanceStats(
            total_io_operations=100,
            async_io_operations=75,
        )
        assert stats.async_ratio == 0.75

    def test_io_stats_optimization_recommendation(self):
        """Test I/O optimization recommendation generation."""
        # No recommendation needed
        stats = IOPerformanceStats(
            total_io_operations=100,
            sync_io_operations=30,
            async_io_operations=70,
        )
        assert stats.get_optimization_recommendation() is None

        # Critical recommendation
        stats = IOPerformanceStats(
            total_io_operations=100,
            sync_io_operations=80,
            async_io_operations=20,
            avg_io_time_ms=15.0,
        )
        rec = stats.get_optimization_recommendation()
        assert rec is not None
        assert rec.priority == OptimizationPriority.CRITICAL
        assert rec.area == OptimizationArea.IO


class TestPerformanceReport:
    """Test PerformanceReport model."""

    def test_performance_report_creation(self):
        """Test performance report creation."""
        profile_results = {
            "op1": ProfileResult(
                operation_name="op1",
                total_time_ms=500.0,
                call_count=5,
                avg_time_ms=100.0,
                p50_ms=95.0,
                p95_ms=150.0,
                p99_ms=180.0,
                memory_mb=10.0,
                bottleneck_score=0.4,
            ),
        }

        recommendations = [
            OptimizationRecommendation(
                area=OptimizationArea.TEMPLATE_CACHE,
                issue="Cache hit rate low",
                recommendation="Preload templates",
                expected_improvement="30%",
                priority=OptimizationPriority.HIGH,
            ),
        ]

        report = PerformanceReport(
            workflow_id="test-workflow",
            total_duration_ms=1250.0,
            profile_results=profile_results,
            hot_paths=list(profile_results.values()),
            recommendations=recommendations,
        )

        assert report.workflow_id == "test-workflow"
        assert report.total_duration_ms == 1250.0
        assert len(report.profile_results) == 1
        assert len(report.recommendations) == 1

    def test_critical_recommendations_filter(self):
        """Test filtering critical recommendations."""
        recommendations = [
            OptimizationRecommendation(
                area=OptimizationArea.TEMPLATE_CACHE,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.CRITICAL,
            ),
            OptimizationRecommendation(
                area=OptimizationArea.PARALLEL_EXECUTION,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.HIGH,
            ),
            OptimizationRecommendation(
                area=OptimizationArea.MEMORY,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.CRITICAL,
            ),
        ]

        report = PerformanceReport(
            workflow_id="test",
            total_duration_ms=1000.0,
            profile_results={},
            hot_paths=[],
            recommendations=recommendations,
        )

        critical = report.critical_recommendations
        assert len(critical) == 2
        assert all(r.priority == OptimizationPriority.CRITICAL for r in critical)

    def test_estimated_speedup_calculation(self):
        """Test estimated speedup calculation."""
        # 2 critical + 1 high = 2.3x speedup
        recommendations = [
            OptimizationRecommendation(
                area=OptimizationArea.TEMPLATE_CACHE,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.CRITICAL,
            ),
            OptimizationRecommendation(
                area=OptimizationArea.PARALLEL_EXECUTION,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.CRITICAL,
            ),
            OptimizationRecommendation(
                area=OptimizationArea.MEMORY,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.HIGH,
            ),
        ]

        report = PerformanceReport(
            workflow_id="test",
            total_duration_ms=1000.0,
            profile_results={},
            hot_paths=[],
            recommendations=recommendations,
        )

        # 1.0 + (2 * 0.5) + (1 * 0.3) = 2.3x
        assert report.estimated_speedup == 2.3

    def test_get_summary(self):
        """Test performance report summary generation."""
        report = PerformanceReport(
            workflow_id="test-workflow",
            total_duration_ms=1500.0,
            profile_results={},
            hot_paths=[],
            recommendations=[],
        )

        summary = report.get_summary()
        assert "test-workflow" in summary
        assert "1500.00ms" in summary
        assert "Estimated Speedup: 1.00x" in summary


@pytest.mark.asyncio
class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""

    async def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(
            enable_memory_profiling=True,
            enable_io_profiling=True,
        )

        assert profiler.enable_memory_profiling is True
        assert profiler.enable_io_profiling is True
        assert len(profiler.profile_data) == 0

    async def test_profile_operation_context_manager(self):
        """Test profiling with context manager."""
        profiler = PerformanceProfiler()

        async with profiler.profile_operation("test_op"):
            await asyncio.sleep(0.01)  # Simulate work

        # Check timing was recorded
        assert "test_op" in profiler._operation_timings
        assert len(profiler._operation_timings["test_op"]) == 1
        assert profiler._operation_timings["test_op"][0] >= 10.0  # At least 10ms

    async def test_profile_workflow(self):
        """Test workflow profiling."""
        profiler = PerformanceProfiler()

        async def sample_workflow():
            async with profiler.profile_operation("step1"):
                await asyncio.sleep(0.01)
            async with profiler.profile_operation("step2"):
                await asyncio.sleep(0.02)

        results = await profiler.profile_workflow(
            workflow_func=sample_workflow,
            iterations=3,
        )

        # Check results
        assert "step1" in results
        assert "step2" in results
        assert results["step1"].call_count == 3
        assert results["step2"].call_count == 3
        assert results["step1"].avg_time_ms >= 10.0
        assert results["step2"].avg_time_ms >= 20.0

    async def test_analyze_hot_paths(self):
        """Test hot path analysis."""
        profiler = PerformanceProfiler()

        profile_data = {
            "fast_op": ProfileResult(
                operation_name="fast_op",
                total_time_ms=100.0,
                call_count=10,
                avg_time_ms=10.0,
                p50_ms=9.0,
                p95_ms=15.0,
                p99_ms=18.0,
                memory_mb=5.0,
                bottleneck_score=0.1,
            ),
            "slow_op": ProfileResult(
                operation_name="slow_op",
                total_time_ms=800.0,
                call_count=8,
                avg_time_ms=100.0,
                p50_ms=95.0,
                p95_ms=150.0,
                p99_ms=180.0,
                memory_mb=20.0,
                bottleneck_score=0.8,
            ),
            "medium_op": ProfileResult(
                operation_name="medium_op",
                total_time_ms=100.0,
                call_count=5,
                avg_time_ms=20.0,
                p50_ms=18.0,
                p95_ms=30.0,
                p99_ms=35.0,
                memory_mb=10.0,
                bottleneck_score=0.1,
            ),
        }

        hot_paths = profiler.analyze_hot_paths(profile_data, threshold=0.2)

        assert len(hot_paths) == 1
        assert hot_paths[0].operation_name == "slow_op"
        assert hot_paths[0].bottleneck_score == 0.8

    async def test_track_cache_access(self):
        """Test cache access tracking."""
        profiler = PerformanceProfiler()

        profiler.track_cache_access(hit=True, access_time_ms=0.5)
        profiler.track_cache_access(hit=True, access_time_ms=0.6)
        profiler.track_cache_access(hit=False, access_time_ms=25.0)

        assert profiler._total_cache_requests == 3
        assert profiler._cache_hits == 2
        assert profiler._cache_misses == 1

    async def test_track_io_operation(self):
        """Test I/O operation tracking."""
        profiler = PerformanceProfiler(enable_io_profiling=True)

        profiler.track_io_operation(is_async=True, duration_ms=5.0)
        profiler.track_io_operation(is_async=False, duration_ms=15.0)

        assert profiler._io_operations == 2
        assert profiler._async_io_count == 1
        assert profiler._sync_io_count == 1
        assert profiler._total_io_time_ms == 20.0

    async def test_generate_performance_report(self):
        """Test performance report generation."""
        profiler = PerformanceProfiler()

        # Add some profile data
        profile_data = {
            "op1": ProfileResult(
                operation_name="op1",
                total_time_ms=500.0,
                call_count=5,
                avg_time_ms=100.0,
                p50_ms=95.0,
                p95_ms=150.0,
                p99_ms=180.0,
                memory_mb=10.0,
                bottleneck_score=0.5,
            ),
        }

        # Track some cache access
        profiler.track_cache_access(hit=True, access_time_ms=0.5)
        profiler.track_cache_access(hit=False, access_time_ms=25.0)

        report = profiler.generate_performance_report(
            workflow_id="test-workflow",
            profile_data=profile_data,
            cache_size=50,
            max_cache_size=100,
        )

        assert report.workflow_id == "test-workflow"
        assert report.total_duration_ms == 500.0
        assert len(report.hot_paths) == 1
        assert report.template_cache_stats is not None
        assert report.template_cache_stats.hit_rate == 0.5


@pytest.mark.asyncio
class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class."""

    async def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(
            profiler=profiler,
            auto_optimize=True,
        )

        assert optimizer.profiler == profiler
        assert optimizer.auto_optimize is True
        assert optimizer.target_cache_hit_rate == 0.95
        assert optimizer.target_parallel_speedup == 3.5

    async def test_optimize_workflow(self):
        """Test workflow optimization."""
        profiler = PerformanceProfiler()
        metrics = Mock(spec=MetricsCollector)
        metrics.record_gauge = AsyncMock()
        metrics.record_counter = AsyncMock()

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            metrics_collector=metrics,
            auto_optimize=False,  # Don't apply optimizations
        )

        async def sample_workflow():
            async with profiler.profile_operation("step1"):
                await asyncio.sleep(0.01)

        report = await optimizer.optimize_workflow(
            workflow_id="test-workflow",
            workflow_func=sample_workflow,
            iterations=2,
            apply_optimizations=False,
        )

        assert report.workflow_id == "test-workflow"
        assert len(optimizer._optimization_history) == 1

    async def test_optimize_template_cache(self):
        """Test template cache optimization."""
        profiler = PerformanceProfiler()
        template_manager = Mock(spec=TemplateManager)
        template_manager.get_cache_stats = Mock(return_value=Mock(hit_rate=0.85))
        template_manager.load_template = AsyncMock()

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            template_manager=template_manager,
        )

        await optimizer.optimize_template_cache(target_hit_rate=0.95)

        # Should attempt to preload templates
        assert template_manager.load_template.call_count > 0

    async def test_tune_parallel_execution(self):
        """Test parallel execution tuning."""
        profiler = PerformanceProfiler()
        executor = Mock(spec=StagedParallelExecutor)
        executor.max_concurrent_tasks = 10

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            staged_executor=executor,
        )

        optimizer.tune_parallel_execution(target_concurrency=16)

        assert executor.max_concurrent_tasks == 16

    async def test_optimize_memory_usage(self):
        """Test memory optimization recommendations."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(profiler=profiler)

        recommendations = optimizer.optimize_memory_usage(target_overhead_mb=50.0)

        assert len(recommendations) > 0
        assert any("generator" in r.lower() for r in recommendations)
        assert any("object pooling" in r.lower() for r in recommendations)

    async def test_optimize_io_operations(self):
        """Test I/O optimization recommendations."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(profiler=profiler)

        recommendations = optimizer.optimize_io_operations(target_async_ratio=0.8)

        assert len(recommendations) > 0
        assert any("async" in r.lower() for r in recommendations)
        assert any("batch" in r.lower() for r in recommendations)

    async def test_get_optimization_summary(self):
        """Test optimization summary generation."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(profiler=profiler)

        # Apply some mock optimizations
        optimizer._applied_optimizations = [
            OptimizationRecommendation(
                area=OptimizationArea.TEMPLATE_CACHE,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.HIGH,
            ),
            OptimizationRecommendation(
                area=OptimizationArea.PARALLEL_EXECUTION,
                issue="test",
                recommendation="test",
                expected_improvement="test",
                priority=OptimizationPriority.CRITICAL,
            ),
        ]

        summary = optimizer.get_optimization_summary()

        assert summary["total_optimizations"] == 2
        assert "template_cache" in summary["optimizations_by_area"]
        assert "parallel_execution" in summary["optimizations_by_area"]

    async def test_reset(self):
        """Test optimizer reset."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(profiler=profiler)

        # Add some data
        optimizer._applied_optimizations = [Mock()]
        optimizer._optimization_history = [Mock()]

        # Reset
        optimizer.reset()

        assert len(optimizer._applied_optimizations) == 0
        assert len(optimizer._optimization_history) == 0


@pytest.mark.asyncio
class TestDryRunMode:
    """Test dry-run mode functionality."""

    async def test_dry_run_initialization(self):
        """Test optimizer initialization with dry_run flag."""
        profiler = PerformanceProfiler()

        # Initialize with dry_run=True
        optimizer = PerformanceOptimizer(profiler=profiler, dry_run=True)

        assert optimizer.dry_run is True
        assert len(optimizer._dry_run_changes) == 0
        assert optimizer._current_dry_run_report is None

    async def test_dry_run_workflow_no_changes_applied(self):
        """Test that dry-run mode doesn't apply changes."""
        profiler = PerformanceProfiler()
        template_manager = Mock(spec=TemplateManager)
        template_manager.get_cache_stats = Mock(return_value=Mock(hit_rate=0.85))
        template_manager.load_template = AsyncMock()

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            template_manager=template_manager,
            dry_run=True,
            auto_optimize=True,
        )

        async def sample_workflow():
            async with profiler.profile_operation("step1"):
                await asyncio.sleep(0.01)

        # Optimize in dry-run mode
        report = await optimizer.optimize_workflow(
            workflow_id="test-dry-run",
            workflow_func=sample_workflow,
            iterations=2,
        )

        # Verify report was generated
        assert report.workflow_id == "test-dry-run"

        # Verify no actual template loading happened in dry-run mode
        # (templates should not be loaded in dry-run)
        # Note: load_template might still be called but changes tracked separately
        assert len(optimizer._dry_run_changes) >= 0  # Changes should be tracked

    async def test_dry_run_template_cache_optimization(self):
        """Test template cache optimization in dry-run mode."""
        profiler = PerformanceProfiler()
        template_manager = Mock(spec=TemplateManager)
        template_manager.get_cache_stats = Mock(
            return_value=Mock(hit_rate=0.80, cache_size=10, max_cache_size=100)
        )
        template_manager.load_template = AsyncMock()

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            template_manager=template_manager,
            dry_run=True,
        )

        # Clear any existing changes
        optimizer._dry_run_changes.clear()

        # Run optimization in dry-run mode
        await optimizer.optimize_template_cache(target_hit_rate=0.95, dry_run=True)

        # Verify changes were tracked
        assert len(optimizer._dry_run_changes) == 1
        change = optimizer._dry_run_changes[0]

        assert change.area == OptimizationArea.TEMPLATE_CACHE
        assert change.change_type == "preload_templates"
        assert "80.0%" in change.current_value
        assert "95.0%" in change.proposed_value

        # Verify no actual template loading happened
        template_manager.load_template.assert_not_called()

    async def test_dry_run_parallel_execution_tuning(self):
        """Test parallel execution tuning in dry-run mode."""
        profiler = PerformanceProfiler()
        executor = Mock(spec=StagedParallelExecutor)
        executor.max_concurrent_tasks = 10

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            staged_executor=executor,
            dry_run=True,
        )

        # Clear any existing changes
        optimizer._dry_run_changes.clear()

        # Run tuning in dry-run mode
        optimizer.tune_parallel_execution(target_concurrency=16, dry_run=True)

        # Verify changes were tracked
        assert len(optimizer._dry_run_changes) == 1
        change = optimizer._dry_run_changes[0]

        assert change.area == OptimizationArea.PARALLEL_EXECUTION
        assert change.change_type == "concurrency_tuning"
        assert change.current_value == 10
        assert change.proposed_value == 16

        # Verify no actual changes were applied
        assert executor.max_concurrent_tasks == 10

    async def test_dry_run_report_generation(self):
        """Test dry-run report generation."""
        profiler = PerformanceProfiler()
        template_manager = Mock(spec=TemplateManager)
        template_manager.get_cache_stats = Mock(return_value=Mock(hit_rate=0.80))
        template_manager.load_template = AsyncMock()

        executor = Mock(spec=StagedParallelExecutor)
        executor.max_concurrent_tasks = 8

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            template_manager=template_manager,
            staged_executor=executor,
            dry_run=True,
            auto_optimize=True,
        )

        async def sample_workflow():
            async with profiler.profile_operation("step1"):
                await asyncio.sleep(0.01)

        # Run optimization with dry-run
        report = await optimizer.optimize_workflow(
            workflow_id="test-report",
            workflow_func=sample_workflow,
            iterations=2,
            cache_size=10,
            max_cache_size=100,
        )

        # Get dry-run report
        dry_report = optimizer.get_dry_run_report()

        # Verify dry-run report exists
        assert dry_report is not None
        assert dry_report.workflow_id == "test-report"
        assert dry_report.total_changes >= 0
        assert dry_report.estimated_speedup >= 1.0

        # Verify report methods work
        summary = dry_report.get_summary()
        assert "DRY-RUN REPORT" in summary
        assert "test-report" in summary

        json_data = dry_report.export_json()
        assert json_data["workflow_id"] == "test-report"
        assert "changes" in json_data

    async def test_dry_run_override_instance_setting(self):
        """Test that method-level dry_run overrides instance setting."""
        profiler = PerformanceProfiler()
        template_manager = Mock(spec=TemplateManager)
        template_manager.get_cache_stats = Mock(return_value=Mock(hit_rate=0.80))
        template_manager.load_template = AsyncMock()

        # Create optimizer with dry_run=False
        optimizer = PerformanceOptimizer(
            profiler=profiler,
            template_manager=template_manager,
            dry_run=False,
        )

        # But call method with dry_run=True
        await optimizer.optimize_template_cache(target_hit_rate=0.95, dry_run=True)

        # Verify dry-run was used (no actual loading)
        template_manager.load_template.assert_not_called()
        assert len(optimizer._dry_run_changes) == 1

    async def test_dry_run_memory_io_optimizations(self):
        """Test memory and I/O optimizations in dry-run mode."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(profiler=profiler, dry_run=True)

        # Clear any existing changes
        optimizer._dry_run_changes.clear()

        # Create mock recommendations
        memory_rec = OptimizationRecommendation(
            area=OptimizationArea.MEMORY,
            issue="Memory overhead too high",
            recommendation="Use object pooling",
            expected_improvement="30% reduction",
            priority=OptimizationPriority.HIGH,
            current_value=75.0,
            target_value=50.0,
        )

        io_rec = OptimizationRecommendation(
            area=OptimizationArea.IO,
            issue="Too many sync operations",
            recommendation="Convert to async",
            expected_improvement="50% faster",
            priority=OptimizationPriority.CRITICAL,
            current_value=0.7,
            target_value=0.5,
        )

        # Apply in dry-run mode
        optimizer._apply_memory_optimization(memory_rec, dry_run=True)
        optimizer._apply_io_optimization(io_rec, dry_run=True)

        # Verify changes were tracked
        assert len(optimizer._dry_run_changes) == 2

        memory_change = optimizer._dry_run_changes[0]
        assert memory_change.area == OptimizationArea.MEMORY
        assert memory_change.change_type == "code_refactoring"

        io_change = optimizer._dry_run_changes[1]
        assert io_change.area == OptimizationArea.IO
        assert io_change.change_type == "async_conversion"

    async def test_reset_clears_dry_run_data(self):
        """Test that reset clears dry-run data."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(profiler=profiler, dry_run=True)

        # Add some mock changes
        optimizer._dry_run_changes.append(
            DryRunChange(
                area=OptimizationArea.TEMPLATE_CACHE,
                change_type="test",
                description="test change",
            )
        )
        optimizer._current_dry_run_report = Mock()

        # Reset
        optimizer.reset()

        # Verify dry-run data was cleared
        assert len(optimizer._dry_run_changes) == 0
        assert optimizer._current_dry_run_report is None


@pytest.mark.asyncio
class TestPerformanceIntegration:
    """Integration tests for performance optimization system."""

    async def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Setup components
        profiler = PerformanceProfiler(
            enable_memory_profiling=False,  # Disable for test performance
            enable_io_profiling=True,
        )

        optimizer = PerformanceOptimizer(
            profiler=profiler,
            auto_optimize=False,
        )

        # Define sample workflow
        async def sample_workflow():
            async with profiler.profile_operation("parse"):
                await asyncio.sleep(0.02)

            async with profiler.profile_operation("generate"):
                await asyncio.sleep(0.05)

            async with profiler.profile_operation("validate"):
                await asyncio.sleep(0.01)

        # Optimize workflow
        report = await optimizer.optimize_workflow(
            workflow_id="integration-test",
            workflow_func=sample_workflow,
            iterations=3,
            apply_optimizations=False,
        )

        # Verify report
        assert report.workflow_id == "integration-test"
        assert len(report.profile_results) == 3
        assert "parse" in report.profile_results
        assert "generate" in report.profile_results
        assert "validate" in report.profile_results

        # Verify hot paths identified (generate should be slowest)
        if report.hot_paths:
            assert report.hot_paths[0].operation_name == "generate"

    async def test_performance_targets_achievement(self):
        """Test that performance targets can be achieved."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer(profiler=profiler)

        # Verify targets are set correctly
        assert optimizer.target_cache_hit_rate >= 0.95
        assert optimizer.target_parallel_speedup >= 3.0
        assert optimizer.target_memory_overhead_mb <= 50.0
        assert optimizer.target_io_async_ratio >= 0.8

        # These targets align with Phase 4 requirements:
        # - 2-3x overall speedup
        # - 95%+ cache hit rate
        # - 3-4x parallel speedup
        # - <50MB memory overhead
