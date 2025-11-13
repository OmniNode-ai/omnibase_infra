"""Tests for BLAKE3 hash generator with performance benchmarks."""

import random
import time

import pytest

from omninode_bridge.services.metadata_stamping.engine.hash_generator import (
    BLAKE3HashGenerator,
    PerformanceMetricsCollector,
)


class TestBLAKE3HashGenerator:
    """Comprehensive test suite for BLAKE3 hash generator."""

    @pytest.fixture
    async def hash_generator(self):
        """Create hash generator instance for testing."""
        generator = BLAKE3HashGenerator(pool_size=10, max_workers=2)
        await generator._initialize_hasher_pool()
        yield generator
        # Cleanup
        await generator.cleanup()

    @pytest.mark.asyncio
    async def test_hash_generation_accuracy(self, hash_generator):
        """Test hash generation accuracy and consistency."""
        test_data = b"test data for hashing"

        # Generate hash multiple times
        hashes = []
        for _ in range(5):
            result = await hash_generator.generate_hash(test_data)
            hashes.append(result["hash"])

        # All hashes should be identical
        assert len(set(hashes)) == 1

        # Hash should be 64 characters (BLAKE3 hex output)
        assert len(hashes[0]) == 64
        assert all(c in "0123456789abcdef" for c in hashes[0])

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_benchmark_small_files(self, hash_generator):
        """Test <2ms requirement for small files (≤1KB)."""
        test_sizes = [100, 500, 1024]  # bytes

        for size in test_sizes:
            test_data = bytes(random.getrandbits(8) for _ in range(size))

            # Warm up
            await hash_generator.generate_hash(test_data)

            # Measure performance over multiple runs
            execution_times = []
            for _ in range(10):  # Reduced iterations for faster testing
                result = await hash_generator.generate_hash(test_data)
                execution_times.append(result["execution_time_ms"])

            # Performance requirements
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)

            assert (
                avg_time < 2.0
            ), f"Average execution time {avg_time:.2f}ms exceeds 2ms for {size} bytes"
            assert (
                max_time < 5.0
            ), f"Max execution time {max_time:.2f}ms exceeds 5ms for {size} bytes"
            assert result["performance_grade"] in ["A", "B"]

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_benchmark_large_files(self, hash_generator):
        """Test performance for large files with thread pool."""
        test_sizes = [1024 * 1024, 5 * 1024 * 1024]  # 1MB, 5MB

        for size in test_sizes:
            test_data = bytes(random.getrandbits(8) for _ in range(size))

            result = await hash_generator.generate_hash(test_data)

            # Large files should still be reasonable
            assert (
                result["execution_time_ms"] < 500
            ), f"Large file processing too slow: {result['execution_time_ms']:.2f}ms"
            assert result["cpu_usage_percent"] < 100

    @pytest.mark.asyncio
    async def test_batch_hash_generation(self, hash_generator):
        """Test concurrent batch hash generation."""
        test_files = [
            bytes(random.getrandbits(8) for _ in range(1024)) for _ in range(5)
        ]

        start_time = time.perf_counter()
        results = await hash_generator.batch_generate_hashes(test_files)
        total_time = (time.perf_counter() - start_time) * 1000

        assert len(results) == len(test_files)
        assert all("hash" in result for result in results)

        # Batch should be faster than sequential
        sequential_time_estimate = len(test_files) * 2  # 2ms per file
        assert total_time < sequential_time_estimate * 2  # Allow some overhead

    @pytest.mark.asyncio
    async def test_error_handling(self, hash_generator):
        """Test error handling for invalid inputs."""
        # Test with string instead of bytes
        with pytest.raises(TypeError, match="file_data must be bytes"):
            await hash_generator.generate_hash("not bytes")

        # Test with other invalid types
        with pytest.raises(TypeError, match="file_data must be bytes"):
            await hash_generator.generate_hash(123)

        with pytest.raises(TypeError, match="file_data must be bytes"):
            await hash_generator.generate_hash(None)

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, hash_generator):
        """Test memory efficiency for different file sizes."""
        import psutil

        process = psutil.Process()

        # Baseline memory
        baseline_memory = process.memory_info().rss

        # Test with moderately large file
        large_file = bytes(1024 * 1024 * 10)  # 10MB
        await hash_generator.generate_hash(large_file)

        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - baseline_memory) / 1024 / 1024  # MB

        # Memory increase should be reasonable (< 50MB for 10MB file)
        assert memory_increase < 50, f"Memory usage too high: {memory_increase:.2f}MB"


class TestPerformanceMetricsCollector:
    """Test performance metrics collection and reporting."""

    @pytest.fixture
    def metrics_collector(self):
        return PerformanceMetricsCollector()

    @pytest.mark.asyncio
    async def test_metrics_recording(self, metrics_collector):
        """Test basic metrics recording functionality."""
        await metrics_collector.record_hash_operation(1.5, 1024, 25.0)

        stats = await metrics_collector.get_performance_stats()
        assert stats["total_operations"] == 1
        assert stats["average_execution_time_ms"] == 1.5
        assert stats["performance_grade"] == "A"

    @pytest.mark.asyncio
    async def test_performance_violation_tracking(self, metrics_collector):
        """Test tracking of performance violations."""
        # Record violation
        await metrics_collector.record_performance_violation(
            3.5, 2048, "hash_generation_slow"
        )

        assert metrics_collector.violation_count == 1
        assert len(metrics_collector.performance_violations) == 1

    @pytest.mark.asyncio
    async def test_batch_metrics(self, metrics_collector):
        """Test batch operation metrics."""
        await metrics_collector.record_batch_operation(10, 9, 15.0)

        recent_ops = list(metrics_collector.recent_operations)

        batch_op = next(op for op in recent_ops if op.get("operation_type") == "batch")
        assert batch_op["batch_size"] == 10
        assert batch_op["successful_count"] == 9
        assert batch_op["average_time_per_file"] == 1.5
