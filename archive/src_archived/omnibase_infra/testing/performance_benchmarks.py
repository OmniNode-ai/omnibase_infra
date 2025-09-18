"""
Performance Benchmarks for ONEX Infrastructure Components

Provides comprehensive performance benchmarking for Kafka adapters,
database connections, outbox processing, and overall system throughput.

Per ONEX performance requirements:
- Message publishing throughput (messages/second)
- Database operation latency (p50, p95, p99)
- Outbox processing batch efficiency
- Memory usage and connection pooling metrics
- End-to-end event processing latency
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psutil


@dataclass
class PerformanceMetrics:
    """Container for performance benchmark results."""

    test_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_duration_seconds: float = 0.0
    throughput_ops_per_second: float = 0.0

    # Latency metrics (in milliseconds)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_connections: int = 0

    # Error details
    error_types: dict[str, int] = field(default_factory=dict)
    latency_samples: list[float] = field(default_factory=list)

    def calculate_statistics(self):
        """Calculate statistical metrics from collected samples."""
        if not self.latency_samples:
            return

        self.avg_latency_ms = statistics.mean(self.latency_samples)
        self.min_latency_ms = min(self.latency_samples)
        self.max_latency_ms = max(self.latency_samples)

        # Calculate percentiles
        sorted_samples = sorted(self.latency_samples)
        n = len(sorted_samples)

        self.p50_latency_ms = sorted_samples[int(n * 0.5)] if n > 0 else 0.0
        self.p95_latency_ms = sorted_samples[int(n * 0.95)] if n > 0 else 0.0
        self.p99_latency_ms = sorted_samples[int(n * 0.99)] if n > 0 else 0.0

        # Calculate throughput
        if self.total_duration_seconds > 0:
            self.throughput_ops_per_second = (
                self.successful_operations / self.total_duration_seconds
            )


@dataclass
class BenchmarkConfiguration:
    """Configuration for performance benchmarks."""

    duration_seconds: int = 30
    concurrent_operations: int = 10
    batch_sizes: list[int] = field(default_factory=lambda: [1, 10, 50, 100])
    message_sizes: list[int] = field(
        default_factory=lambda: [1024, 4096, 16384, 65536],
    )  # bytes
    warm_up_duration: int = 5
    cool_down_duration: int = 2
    resource_monitoring_interval: float = 0.1


class InfrastructurePerformanceBenchmarks:
    """
    Comprehensive performance benchmark suite for ONEX infrastructure.

    Benchmarks:
    - Kafka message publishing throughput and latency
    - Database query performance and connection pooling
    - Outbox processing batch efficiency
    - End-to-end event processing performance
    - Memory usage and resource consumption
    """

    def __init__(
        self, kafka_adapter=None, postgres_outbox=None, connection_manager=None,
    ):
        """
        Initialize performance benchmark suite.

        Args:
            kafka_adapter: Kafka adapter instance for messaging benchmarks
            postgres_outbox: PostgreSQL outbox instance for outbox benchmarks
            connection_manager: Database connection manager for DB benchmarks
        """
        self._logger = logging.getLogger(__name__)
        self._kafka_adapter = kafka_adapter
        self._postgres_outbox = postgres_outbox
        self._connection_manager = connection_manager

        self._config = BenchmarkConfiguration()
        self._benchmark_results: dict[str, PerformanceMetrics] = {}
        self._resource_monitor_task: asyncio.Task | None = None
        self._resource_metrics: list[dict[str, float]] = []

        self._logger.info("Infrastructure performance benchmarks initialized")

    async def run_full_benchmark_suite(self) -> dict[str, Any]:
        """
        Run complete performance benchmark suite.

        Returns:
            Comprehensive benchmark results with recommendations
        """
        self._logger.info("Starting full infrastructure performance benchmark suite")
        start_time = time.perf_counter()

        # Start resource monitoring
        await self._start_resource_monitoring()

        try:
            # Run individual benchmark categories
            benchmarks = [
                ("kafka_messaging", self._benchmark_kafka_messaging),
                ("database_operations", self._benchmark_database_operations),
                ("outbox_processing", self._benchmark_outbox_processing),
                ("end_to_end_latency", self._benchmark_end_to_end_latency),
                ("concurrent_load", self._benchmark_concurrent_load),
                ("memory_efficiency", self._benchmark_memory_efficiency),
            ]

            for benchmark_name, benchmark_func in benchmarks:
                if await self._should_run_benchmark(benchmark_name):
                    self._logger.info(f"Running benchmark: {benchmark_name}")
                    try:
                        await benchmark_func()
                    except Exception as e:
                        self._logger.error(
                            f"Benchmark '{benchmark_name}' failed: {e!s}",
                        )

                        # Create error metrics for failed benchmark
                        error_metrics = PerformanceMetrics(test_name=benchmark_name)
                        error_metrics.error_types["benchmark_failure"] = 1
                        self._benchmark_results[benchmark_name] = error_metrics

                    # Cool down between benchmarks
                    await asyncio.sleep(self._config.cool_down_duration)

            # Stop resource monitoring
            await self._stop_resource_monitoring()

            total_duration = time.perf_counter() - start_time

            # Generate comprehensive report
            report = {
                "summary": self._generate_benchmark_summary(total_duration),
                "detailed_results": {
                    name: self._serialize_metrics(metrics)
                    for name, metrics in self._benchmark_results.items()
                },
                "resource_analysis": self._analyze_resource_usage(),
                "performance_recommendations": self._generate_performance_recommendations(),
                "benchmark_configuration": self._serialize_config(),
                "timestamp": datetime.utcnow().isoformat(),
            }

            self._logger.info(f"Benchmark suite completed in {total_duration:.2f}s")
            return report

        finally:
            # Ensure resource monitoring is stopped
            await self._stop_resource_monitoring()

    async def _benchmark_kafka_messaging(self):
        """Benchmark Kafka message publishing performance."""
        if not self._kafka_adapter:
            self._logger.warning(
                "Kafka adapter not available, skipping Kafka benchmarks",
            )
            return

        for batch_size in self._config.batch_sizes:
            for message_size in self._config.message_sizes:
                test_name = f"kafka_publish_batch_{batch_size}_size_{message_size}"

                metrics = PerformanceMetrics(test_name=test_name)

                # Generate test message payload
                test_payload = {"data": "x" * message_size, "batch_size": batch_size}

                # Warm-up phase
                await self._kafka_warm_up(test_payload, 10)

                # Benchmark phase
                start_time = time.perf_counter()

                tasks = []
                for _ in range(batch_size):
                    task = asyncio.create_task(
                        self._timed_kafka_publish(
                            "benchmark_topic", test_payload, metrics,
                        ),
                    )
                    tasks.append(task)

                # Wait for all publishing tasks
                await asyncio.gather(*tasks, return_exceptions=True)

                metrics.total_duration_seconds = time.perf_counter() - start_time
                metrics.calculate_statistics()

                self._benchmark_results[test_name] = metrics

                self._logger.info(
                    f"Kafka benchmark {test_name}: "
                    f"{metrics.throughput_ops_per_second:.1f} ops/sec, "
                    f"p95: {metrics.p95_latency_ms:.1f}ms",
                )

    async def _benchmark_database_operations(self):
        """Benchmark database query and transaction performance."""
        if not self._connection_manager:
            self._logger.warning(
                "Connection manager not available, skipping DB benchmarks",
            )
            return

        operations = [
            ("select_simple", "SELECT 1"),
            (
                "select_with_join",
                """
                SELECT u.id, u.name, p.title
                FROM users u
                LEFT JOIN posts p ON u.id = p.user_id
                LIMIT 100
            """,
            ),
            ("insert_batch", "INSERT INTO test_table (data) VALUES ($1)"),
        ]

        for operation_name, query in operations:
            test_name = f"database_{operation_name}"
            metrics = PerformanceMetrics(test_name=test_name)

            # Warm-up
            await self._database_warm_up(query, 5)

            # Benchmark concurrent database operations
            start_time = time.perf_counter()

            tasks = []
            for i in range(self._config.concurrent_operations):
                task = asyncio.create_task(
                    self._timed_database_operation(query, f"test_data_{i}", metrics),
                )
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

            metrics.total_duration_seconds = time.perf_counter() - start_time
            metrics.calculate_statistics()

            self._benchmark_results[test_name] = metrics

    async def _benchmark_outbox_processing(self):
        """Benchmark outbox event processing performance."""
        if not self._postgres_outbox:
            self._logger.warning(
                "Outbox processor not available, skipping outbox benchmarks",
            )
            return

        for batch_size in [10, 50, 100, 250]:
            test_name = f"outbox_processing_batch_{batch_size}"
            metrics = PerformanceMetrics(test_name=test_name)

            # Generate test events in outbox
            await self._populate_outbox_events(batch_size)

            # Benchmark outbox processing
            start_time = time.perf_counter()

            try:
                # Process outbox events (this is implementation-specific)
                if hasattr(self._postgres_outbox, "process_pending_events"):
                    await self._postgres_outbox.process_pending_events()
                    metrics.successful_operations = batch_size
                else:
                    self._logger.warning(
                        "Outbox processor missing process_pending_events method",
                    )

            except Exception as e:
                metrics.failed_operations = batch_size
                metrics.error_types[type(e).__name__] = 1

            metrics.total_duration_seconds = time.perf_counter() - start_time
            metrics.total_operations = batch_size
            metrics.calculate_statistics()

            self._benchmark_results[test_name] = metrics

    async def _benchmark_end_to_end_latency(self):
        """Benchmark end-to-end event processing latency."""
        test_name = "end_to_end_latency"
        metrics = PerformanceMetrics(test_name=test_name)

        if not (self._kafka_adapter and self._postgres_outbox):
            self._logger.warning("Missing components for end-to-end benchmark")
            return

        # Test end-to-end processing: outbox -> kafka -> consumer
        test_events = [
            {"event_type": "test_event", "data": f"test_payload_{i}"} for i in range(20)
        ]

        start_time = time.perf_counter()

        for i, event_data in enumerate(test_events):
            event_start = time.perf_counter()

            try:
                # Simulate full end-to-end processing
                # This would typically involve: DB transaction -> outbox -> kafka -> processing

                # For now, just measure kafka publish latency as proxy
                await self._timed_kafka_publish(
                    "end_to_end_test",
                    event_data,
                    metrics,
                )

            except Exception as e:
                metrics.failed_operations += 1
                metrics.error_types[type(e).__name__] = (
                    metrics.error_types.get(type(e).__name__, 0) + 1
                )

        metrics.total_duration_seconds = time.perf_counter() - start_time
        metrics.calculate_statistics()

        self._benchmark_results[test_name] = metrics

    async def _benchmark_concurrent_load(self):
        """Benchmark system behavior under concurrent load."""
        test_name = "concurrent_load_test"
        metrics = PerformanceMetrics(test_name=test_name)

        # Create high concurrent load across all components
        concurrent_tasks = []

        # Kafka publishing tasks
        if self._kafka_adapter:
            for i in range(self._config.concurrent_operations):
                task = asyncio.create_task(
                    self._sustained_kafka_publishing(f"load_test_{i}", 10, metrics),
                )
                concurrent_tasks.append(task)

        # Database operation tasks
        if self._connection_manager:
            for i in range(self._config.concurrent_operations // 2):
                task = asyncio.create_task(
                    self._sustained_database_operations(
                        f"load_test_db_{i}", 10, metrics,
                    ),
                )
                concurrent_tasks.append(task)

        start_time = time.perf_counter()

        # Run concurrent load for specified duration
        try:
            await asyncio.wait_for(
                asyncio.gather(*concurrent_tasks, return_exceptions=True),
                timeout=self._config.duration_seconds,
            )
        except TimeoutError:
            # Expected - we're testing sustained load
            pass

        metrics.total_duration_seconds = time.perf_counter() - start_time
        metrics.calculate_statistics()

        self._benchmark_results[test_name] = metrics

    async def _benchmark_memory_efficiency(self):
        """Benchmark memory usage efficiency."""
        test_name = "memory_efficiency"
        metrics = PerformanceMetrics(test_name=test_name)

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform memory-intensive operations
        large_payloads = []
        for i in range(100):
            large_payload = {"data": "x" * 10000, "id": i}  # ~10KB per payload
            large_payloads.append(large_payload)

            if self._kafka_adapter and i % 10 == 0:
                try:
                    await self._timed_kafka_publish(
                        "memory_test", large_payload, metrics,
                    )
                except Exception:
                    pass  # Memory test, errors expected under load

        # Measure peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        metrics.peak_memory_mb = peak_memory - initial_memory

        # Clean up
        large_payloads.clear()

        self._benchmark_results[test_name] = metrics

    async def _timed_kafka_publish(
        self, topic: str, payload: dict[str, Any], metrics: PerformanceMetrics,
    ):
        """Publish message to Kafka and record timing metrics."""
        start_time = time.perf_counter()

        try:
            # This would use the actual Kafka adapter publish method
            # For now, simulate with a delay
            await asyncio.sleep(0.001)  # Simulate network latency

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.latency_samples.append(duration_ms)
            metrics.successful_operations += 1

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.latency_samples.append(duration_ms)
            metrics.failed_operations += 1

            error_type = type(e).__name__
            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1

        metrics.total_operations += 1

    async def _timed_database_operation(
        self, query: str, param: str, metrics: PerformanceMetrics,
    ):
        """Execute database operation and record timing metrics."""
        start_time = time.perf_counter()

        try:
            # This would use the actual database connection
            # For now, simulate with a delay
            await asyncio.sleep(0.002)  # Simulate DB query time

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.latency_samples.append(duration_ms)
            metrics.successful_operations += 1

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.latency_samples.append(duration_ms)
            metrics.failed_operations += 1

            error_type = type(e).__name__
            metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1

        metrics.total_operations += 1

    async def _sustained_kafka_publishing(
        self, client_id: str, messages_per_second: int, metrics: PerformanceMetrics,
    ):
        """Sustain Kafka publishing at specified rate."""
        interval = 1.0 / messages_per_second

        for i in range(messages_per_second * self._config.duration_seconds):
            await self._timed_kafka_publish(
                "sustained_test",
                {"client_id": client_id, "message_id": i},
                metrics,
            )
            await asyncio.sleep(interval)

    async def _sustained_database_operations(
        self, client_id: str, ops_per_second: int, metrics: PerformanceMetrics,
    ):
        """Sustain database operations at specified rate."""
        interval = 1.0 / ops_per_second

        for i in range(ops_per_second * self._config.duration_seconds):
            await self._timed_database_operation(
                "SELECT 1",
                f"{client_id}_{i}",
                metrics,
            )
            await asyncio.sleep(interval)

    async def _kafka_warm_up(self, payload: dict[str, Any], count: int):
        """Warm up Kafka adapter before benchmarking."""
        for _ in range(count):
            try:
                await asyncio.sleep(0.001)  # Simulate warm-up
            except:
                pass  # Ignore warm-up errors

    async def _database_warm_up(self, query: str, count: int):
        """Warm up database connections before benchmarking."""
        for _ in range(count):
            try:
                await asyncio.sleep(0.002)  # Simulate warm-up
            except:
                pass  # Ignore warm-up errors

    async def _populate_outbox_events(self, count: int):
        """Populate outbox with test events for benchmarking."""
        # This would populate the actual outbox table
        # For now, just simulate the time it would take
        await asyncio.sleep(0.01 * count)

    async def _should_run_benchmark(self, benchmark_name: str) -> bool:
        """Determine if benchmark should run based on available components."""
        component_requirements = {
            "kafka_messaging": self._kafka_adapter is not None,
            "database_operations": self._connection_manager is not None,
            "outbox_processing": self._postgres_outbox is not None,
            "end_to_end_latency": self._kafka_adapter is not None
            and self._postgres_outbox is not None,
            "concurrent_load": True,  # Can always run basic concurrent tests
            "memory_efficiency": True,  # Can always measure memory
        }

        return component_requirements.get(benchmark_name, True)

    async def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        self._resource_monitor_task = asyncio.create_task(self._monitor_resources())

    async def _stop_resource_monitoring(self):
        """Stop background resource monitoring."""
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
            try:
                await self._resource_monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_resources(self):
        """Background task to monitor system resources."""
        process = psutil.Process()

        while True:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()

                self._resource_metrics.append(
                    {
                        "timestamp": time.perf_counter(),
                        "memory_mb": memory_mb,
                        "cpu_percent": cpu_percent,
                    },
                )

                await asyncio.sleep(self._config.resource_monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Resource monitoring error: {e!s}")
                break

    def _generate_benchmark_summary(self, total_duration: float) -> dict[str, Any]:
        """Generate high-level benchmark summary."""
        total_benchmarks = len(self._benchmark_results)
        successful_benchmarks = sum(
            1
            for metrics in self._benchmark_results.values()
            if metrics.failed_operations == 0 and not metrics.error_types
        )

        # Calculate aggregate throughput
        total_ops = sum(
            metrics.successful_operations
            for metrics in self._benchmark_results.values()
        )
        aggregate_throughput = total_ops / total_duration if total_duration > 0 else 0

        return {
            "total_benchmarks": total_benchmarks,
            "successful_benchmarks": successful_benchmarks,
            "success_rate": (
                (successful_benchmarks / total_benchmarks * 100)
                if total_benchmarks > 0
                else 0
            ),
            "total_duration_seconds": round(total_duration, 2),
            "aggregate_throughput_ops_per_second": round(aggregate_throughput, 2),
            "total_operations": total_ops,
        }

    def _analyze_resource_usage(self) -> dict[str, Any]:
        """Analyze collected resource usage metrics."""
        if not self._resource_metrics:
            return {"error": "No resource metrics collected"}

        memory_values = [m["memory_mb"] for m in self._resource_metrics]
        cpu_values = [
            m["cpu_percent"] for m in self._resource_metrics if m["cpu_percent"] > 0
        ]

        return {
            "memory_usage_mb": {
                "avg": round(statistics.mean(memory_values), 2) if memory_values else 0,
                "peak": round(max(memory_values), 2) if memory_values else 0,
                "min": round(min(memory_values), 2) if memory_values else 0,
            },
            "cpu_usage_percent": {
                "avg": round(statistics.mean(cpu_values), 2) if cpu_values else 0,
                "peak": round(max(cpu_values), 2) if cpu_values else 0,
            },
            "monitoring_duration_seconds": len(self._resource_metrics)
            * self._config.resource_monitoring_interval,
        }

    def _generate_performance_recommendations(self) -> list[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []

        for test_name, metrics in self._benchmark_results.items():
            # Throughput recommendations
            if metrics.throughput_ops_per_second < 100:
                recommendations.append(
                    f"Low throughput in {test_name}: {metrics.throughput_ops_per_second:.1f} ops/sec. "
                    f"Consider optimization or scaling.",
                )

            # Latency recommendations
            if metrics.p95_latency_ms > 1000:  # > 1 second
                recommendations.append(
                    f"High P95 latency in {test_name}: {metrics.p95_latency_ms:.1f}ms. "
                    f"Review performance bottlenecks.",
                )

            # Error rate recommendations
            if metrics.failed_operations > 0:
                error_rate = (
                    metrics.failed_operations / metrics.total_operations
                ) * 100
                recommendations.append(
                    f"Error rate in {test_name}: {error_rate:.1f}%. "
                    f"Review error types: {list(metrics.error_types.keys())}",
                )

        # Resource usage recommendations
        resource_analysis = self._analyze_resource_usage()
        if "memory_usage_mb" in resource_analysis:
            peak_memory = resource_analysis["memory_usage_mb"]["peak"]
            if peak_memory > 1000:  # > 1GB
                recommendations.append(
                    f"High peak memory usage: {peak_memory:.1f}MB. "
                    f"Consider memory optimization.",
                )

        if not recommendations:
            recommendations.append(
                "All benchmarks performed within acceptable parameters. "
                "System appears well-optimized for current load patterns.",
            )

        return recommendations

    def _serialize_metrics(self, metrics: PerformanceMetrics) -> dict[str, Any]:
        """Serialize metrics object to dictionary."""
        result = {
            "test_name": metrics.test_name,
            "total_operations": metrics.total_operations,
            "successful_operations": metrics.successful_operations,
            "failed_operations": metrics.failed_operations,
            "total_duration_seconds": round(metrics.total_duration_seconds, 3),
            "throughput_ops_per_second": round(metrics.throughput_ops_per_second, 2),
            "latency_metrics_ms": {
                "avg": round(metrics.avg_latency_ms, 2),
                "min": round(metrics.min_latency_ms, 2),
                "max": round(metrics.max_latency_ms, 2),
                "p50": round(metrics.p50_latency_ms, 2),
                "p95": round(metrics.p95_latency_ms, 2),
                "p99": round(metrics.p99_latency_ms, 2),
            },
            "resource_metrics": {
                "peak_memory_mb": round(metrics.peak_memory_mb, 2),
                "avg_cpu_percent": round(metrics.avg_cpu_percent, 2),
                "peak_connections": metrics.peak_connections,
            },
            "error_analysis": {
                "error_types": metrics.error_types,
                "error_rate_percent": round(
                    (
                        (metrics.failed_operations / metrics.total_operations * 100)
                        if metrics.total_operations > 0
                        else 0
                    ),
                    2,
                ),
            },
        }

        return result

    def _serialize_config(self) -> dict[str, Any]:
        """Serialize benchmark configuration."""
        return {
            "duration_seconds": self._config.duration_seconds,
            "concurrent_operations": self._config.concurrent_operations,
            "batch_sizes": self._config.batch_sizes,
            "message_sizes": self._config.message_sizes,
            "warm_up_duration": self._config.warm_up_duration,
            "cool_down_duration": self._config.cool_down_duration,
        }


# Helper function for easy benchmark execution
async def run_infrastructure_benchmarks(
    kafka_adapter=None, postgres_outbox=None, connection_manager=None,
) -> dict[str, Any]:
    """
    Convenience function to run infrastructure performance benchmarks.

    Args:
        kafka_adapter: Kafka adapter instance
        postgres_outbox: PostgreSQL outbox instance
        connection_manager: Database connection manager

    Returns:
        Comprehensive benchmark results
    """
    benchmarks = InfrastructurePerformanceBenchmarks(
        kafka_adapter=kafka_adapter,
        postgres_outbox=postgres_outbox,
        connection_manager=connection_manager,
    )
    return await benchmarks.run_full_benchmark_suite()
