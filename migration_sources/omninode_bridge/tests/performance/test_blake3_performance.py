"""
Performance Benchmark Tests for BLAKE3 Hash Generator.

Establishes baseline performance thresholds and provides regression testing
for BLAKE3 hash generation operations in the MetadataStampingService.

Expected Performance Baselines:
    - Small files (< 1KB): < 1ms average, < 2ms p99
    - Medium files (1KB-1MB): < 1.5ms average, < 2ms p99
    - Large files (> 1MB): < 5ms per MB
    - Batch operations: > 500 hashes/second

Hash Generation Performance:
    - Direct hash (≤1KB): < 0.5ms
    - Pooled hasher (≤1MB): < 1.5ms
    - Streaming hash (>1MB): < 5ms/MB

Usage:
    # Run all BLAKE3 benchmarks
    pytest tests/performance/test_blake3_performance.py -m performance

    # Run specific benchmark
    pytest tests/performance/test_blake3_performance.py::test_small_file_hash_performance

    # View benchmark results
    pytest tests/performance/test_blake3_performance.py --benchmark-only
"""

import asyncio
import time

import pytest

# Import BLAKE3 hash generator
from omninode_bridge.services.metadata_stamping.engine.hash_generator import (
    BLAKE3HashGenerator,
)

# Performance thresholds (in milliseconds)
PERFORMANCE_THRESHOLDS = {
    "small_file_avg": 1.0,  # < 1ms average for files < 1KB
    "small_file_p99": 2.0,  # < 2ms p99 for files < 1KB
    "medium_file_avg": 1.5,  # < 1.5ms average for files 1KB-1MB
    "medium_file_p99": 2.0,  # < 2ms p99 for files 1KB-1MB
    "large_file_per_mb": 5.0,  # < 5ms per MB for files > 1MB
    "batch_throughput": 500,  # > 500 hashes/second
    "direct_hash": 0.5,  # < 0.5ms for direct hash path
    "pooled_hasher": 1.5,  # < 1.5ms for pooled hasher path
}


@pytest.fixture
async def hash_generator():
    """Create BLAKE3HashGenerator for testing."""
    generator = BLAKE3HashGenerator(pool_size=100, max_workers=4)

    # Wait for pool initialization
    await asyncio.sleep(0.1)

    yield generator

    # Cleanup after test
    await generator.cleanup()


@pytest.fixture
def small_file_data():
    """Create small file data (< 1KB)."""
    return b"Hello World! This is a small test file." * 10  # ~400 bytes


@pytest.fixture
def medium_file_data():
    """Create medium file data (1KB-1MB)."""
    return b"x" * (100 * 1024)  # 100KB


@pytest.fixture
def large_file_data():
    """Create large file data (> 1MB)."""
    return b"y" * (2 * 1024 * 1024)  # 2MB


@pytest.fixture
def batch_files_data():
    """Create batch of files for concurrent testing."""
    return [b"test_data_" + str(i).encode() * 100 for i in range(100)]


# ===== Small File Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_small_file_hash_performance(benchmark, hash_generator, small_file_data):
    """
    Benchmark BLAKE3 hash generation for small files (< 1KB).

    Expected Performance:
        - Average: < 1ms
        - P99: < 2ms

    This test validates the direct hash path for very small files,
    ensuring optimal performance without pool overhead.
    """

    # Warmup rounds
    for _ in range(5):
        await hash_generator.generate_hash(small_file_data)

    # Collect timing samples
    times = []
    iterations = 100

    for _ in range(iterations):
        start = time.perf_counter()
        result = await hash_generator.generate_hash(small_file_data)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Validate result structure
    assert "hash" in result
    assert "execution_time_ms" in result
    assert "file_size_bytes" in result
    assert result["file_size_bytes"] == len(small_file_data)
    assert len(result["hash"]) == 64  # BLAKE3 produces 256-bit (64 hex char) hashes

    # Calculate statistics
    mean_time_ms = sum(times) / len(times)
    times_sorted = sorted(times)
    p99_time_ms = times_sorted[int(len(times_sorted) * 0.99)]

    # Report statistics to benchmark fixture
    benchmark.extra_info["mean_ms"] = f"{mean_time_ms:.3f}"
    benchmark.extra_info["p99_ms"] = f"{p99_time_ms:.3f}"
    benchmark.extra_info["min_ms"] = f"{min(times):.3f}"
    benchmark.extra_info["max_ms"] = f"{max(times):.3f}"

    # Validate performance thresholds
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["small_file_avg"]
    ), f"Small file hash too slow (avg): {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['small_file_avg']}ms"

    # Check p99 latency
    assert (
        p99_time_ms < PERFORMANCE_THRESHOLDS["small_file_p99"]
    ), f"Small file hash too slow (p99): {p99_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['small_file_p99']}ms"


# ===== Medium File Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_medium_file_hash_performance(
    benchmark, hash_generator, medium_file_data
):
    """
    Benchmark BLAKE3 hash generation for medium files (1KB-1MB).

    Expected Performance:
        - Average: < 1.5ms
        - P99: < 2ms

    This test validates the pooled hasher path for medium-sized files,
    ensuring efficient use of the pre-allocated hasher pool.
    """

    # Warmup rounds
    for _ in range(3):
        await hash_generator.generate_hash(medium_file_data)

    # Collect timing samples
    times = []
    iterations = 50

    for _ in range(iterations):
        start = time.perf_counter()
        result = await hash_generator.generate_hash(medium_file_data)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Validate result structure
    assert "hash" in result
    assert "execution_time_ms" in result
    assert result["file_size_bytes"] == len(medium_file_data)
    assert len(result["hash"]) == 64

    # Calculate statistics
    mean_time_ms = sum(times) / len(times)
    times_sorted = sorted(times)
    p99_time_ms = times_sorted[int(len(times_sorted) * 0.99)]

    # Report statistics to benchmark fixture
    benchmark.extra_info["mean_ms"] = f"{mean_time_ms:.3f}"
    benchmark.extra_info["p99_ms"] = f"{p99_time_ms:.3f}"
    benchmark.extra_info["min_ms"] = f"{min(times):.3f}"
    benchmark.extra_info["max_ms"] = f"{max(times):.3f}"

    # Validate performance thresholds
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["medium_file_avg"]
    ), f"Medium file hash too slow (avg): {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['medium_file_avg']}ms"

    # Check p99 latency
    assert (
        p99_time_ms < PERFORMANCE_THRESHOLDS["medium_file_p99"]
    ), f"Medium file hash too slow (p99): {p99_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['medium_file_p99']}ms"


# ===== Large File Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_large_file_hash_performance(benchmark, hash_generator, large_file_data):
    """
    Benchmark BLAKE3 hash generation for large files (> 1MB).

    Expected Performance: < 5ms per MB

    This test validates the streaming hash path with thread pool processing,
    ensuring efficient handling of large files without blocking the event loop.
    """

    # Warmup rounds
    for _ in range(2):
        await hash_generator.generate_hash(large_file_data)

    # Collect timing samples
    times = []
    iterations = 20

    for _ in range(iterations):
        start = time.perf_counter()
        result = await hash_generator.generate_hash(large_file_data)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Validate result structure
    assert "hash" in result
    assert "execution_time_ms" in result
    assert result["file_size_bytes"] == len(large_file_data)

    # Calculate statistics
    mean_time_ms = sum(times) / len(times)
    file_size_mb = len(large_file_data) / (1024 * 1024)
    time_per_mb = mean_time_ms / file_size_mb

    # Report statistics to benchmark fixture
    benchmark.extra_info["mean_ms"] = f"{mean_time_ms:.3f}"
    benchmark.extra_info["time_per_mb_ms"] = f"{time_per_mb:.3f}"
    benchmark.extra_info["file_size_mb"] = f"{file_size_mb:.2f}"

    # Validate performance threshold (per MB)
    assert (
        time_per_mb < PERFORMANCE_THRESHOLDS["large_file_per_mb"]
    ), f"Large file hash too slow: {time_per_mb:.2f}ms/MB > {PERFORMANCE_THRESHOLDS['large_file_per_mb']}ms/MB"


# ===== Batch Operations Performance Tests =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_hash_performance(benchmark, hash_generator, batch_files_data):
    """
    Benchmark BLAKE3 batch hash generation.

    Expected Performance: > 500 hashes per second

    This test validates concurrent hash generation with semaphore control,
    ensuring high-throughput batch processing for multiple files.
    """

    # Warmup rounds
    for _ in range(2):
        await hash_generator.batch_generate_hashes(batch_files_data)

    # Collect timing samples
    times = []
    iterations = 10

    for _ in range(iterations):
        start = time.perf_counter()
        results = await hash_generator.batch_generate_hashes(batch_files_data)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Validate all operations succeeded
    assert len(results) == len(batch_files_data)
    assert all("hash" in r for r in results)

    # Calculate statistics
    mean_time_ms = sum(times) / len(times)
    mean_time_s = mean_time_ms / 1000
    throughput = len(batch_files_data) / mean_time_s

    # Report statistics to benchmark fixture
    benchmark.extra_info["mean_ms"] = f"{mean_time_ms:.3f}"
    benchmark.extra_info["throughput_hashes_per_sec"] = f"{throughput:.1f}"
    benchmark.extra_info["batch_size"] = str(len(batch_files_data))

    # Validate throughput threshold
    assert (
        throughput >= PERFORMANCE_THRESHOLDS["batch_throughput"]
    ), f"Batch throughput too low: {throughput:.2f} hashes/sec < {PERFORMANCE_THRESHOLDS['batch_throughput']} hashes/sec"


# ===== Direct Hash Path Performance Test =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_direct_hash_path(benchmark, hash_generator):
    """
    Benchmark direct hash path for very small files (≤1KB).

    Expected Performance: < 0.5ms

    This test validates the optimized direct hash path that bypasses
    the hasher pool for minimal overhead on very small files.
    """
    tiny_data = b"test" * 10  # 40 bytes

    # Warmup rounds
    for _ in range(5):
        await hash_generator.generate_hash(tiny_data)

    # Collect timing samples
    times = []
    iterations = 200

    for _ in range(iterations):
        start = time.perf_counter()
        result = await hash_generator.generate_hash(tiny_data)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Validate result
    assert "hash" in result
    assert result["file_size_bytes"] == len(tiny_data)

    # Calculate statistics
    mean_time_ms = sum(times) / len(times)

    # Report statistics to benchmark fixture
    benchmark.extra_info["mean_ms"] = f"{mean_time_ms:.3f}"
    benchmark.extra_info["min_ms"] = f"{min(times):.3f}"
    benchmark.extra_info["max_ms"] = f"{max(times):.3f}"

    # Validate performance threshold
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["direct_hash"]
    ), f"Direct hash too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['direct_hash']}ms"


# ===== Pooled Hasher Path Performance Test =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_pooled_hasher_path(benchmark, hash_generator):
    """
    Benchmark pooled hasher path for medium files.

    Expected Performance: < 1.5ms

    This test validates the hasher pool efficiency for medium-sized files,
    ensuring proper pool management and minimal allocation overhead.
    """
    medium_data = b"x" * (50 * 1024)  # 50KB

    # Warmup rounds
    for _ in range(3):
        await hash_generator.generate_hash(medium_data)

    # Collect timing samples
    times = []
    iterations = 100

    for _ in range(iterations):
        start = time.perf_counter()
        result = await hash_generator.generate_hash(medium_data)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Validate result
    assert "hash" in result
    assert result["file_size_bytes"] == len(medium_data)

    # Calculate statistics
    mean_time_ms = sum(times) / len(times)

    # Report statistics to benchmark fixture
    benchmark.extra_info["mean_ms"] = f"{mean_time_ms:.3f}"
    benchmark.extra_info["min_ms"] = f"{min(times):.3f}"
    benchmark.extra_info["max_ms"] = f"{max(times):.3f}"

    # Validate performance threshold
    assert (
        mean_time_ms < PERFORMANCE_THRESHOLDS["pooled_hasher"]
    ), f"Pooled hasher too slow: {mean_time_ms:.2f}ms > {PERFORMANCE_THRESHOLDS['pooled_hasher']}ms"


# ===== Performance Grade Validation Test =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_grade_distribution(hash_generator, small_file_data):
    """
    Validate performance grade distribution for hash operations.

    This test ensures that the majority of operations achieve Grade A
    performance (< 1ms), with minimal Grade C violations (> 2ms).
    """
    results = []

    # Run 100 hash operations
    for _ in range(100):
        result = await hash_generator.generate_hash(small_file_data)
        results.append(result)

    # Analyze performance grades
    grade_counts = {"A": 0, "B": 0, "C": 0}
    for result in results:
        grade = result.get("performance_grade", "C")
        grade_counts[grade] += 1

    # Validate grade distribution
    total_operations = len(results)
    grade_a_percentage = (grade_counts["A"] / total_operations) * 100
    grade_c_percentage = (grade_counts["C"] / total_operations) * 100

    # At least 80% should be Grade A (< 1ms)
    assert (
        grade_a_percentage >= 80
    ), f"Grade A percentage too low: {grade_a_percentage:.1f}% < 80%"

    # At most 5% should be Grade C (> 2ms)
    assert (
        grade_c_percentage <= 5
    ), f"Grade C percentage too high: {grade_c_percentage:.1f}% > 5%"

    print("\n=== Performance Grade Distribution ===")
    print(f"Grade A (< 1ms): {grade_counts['A']} ({grade_a_percentage:.1f}%)")
    print(
        f"Grade B (1-2ms): {grade_counts['B']} ({(grade_counts['B']/total_operations)*100:.1f}%)"
    )
    print(f"Grade C (> 2ms): {grade_counts['C']} ({grade_c_percentage:.1f}%)")


# ===== Concurrent Performance Test =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_hash_operations(benchmark, hash_generator):
    """
    Benchmark concurrent hash operations under load.

    Expected Performance: Maintain < 2ms p99 under concurrent load

    This test validates that the hash generator maintains performance
    under concurrent stress, ensuring thread-safe pool management.
    """

    async def run_concurrent_hashes():
        """Execute 20 concurrent hash operations."""
        files = [b"concurrent_test_" + str(i).encode() * 100 for i in range(20)]

        tasks = [hash_generator.generate_hash(file_data) for file_data in files]
        results = await asyncio.gather(*tasks)
        return results

    # Warmup rounds
    for _ in range(2):
        await run_concurrent_hashes()

    # Collect timing samples
    times = []
    iterations = 10

    for _ in range(iterations):
        start = time.perf_counter()
        results = await run_concurrent_hashes()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    # Validate all operations succeeded
    assert len(results) == 20
    assert all("hash" in r for r in results)

    # Calculate statistics
    mean_time_ms = sum(times) / len(times)

    # Report statistics to benchmark fixture
    benchmark.extra_info["mean_ms"] = f"{mean_time_ms:.3f}"
    benchmark.extra_info["min_ms"] = f"{min(times):.3f}"
    benchmark.extra_info["max_ms"] = f"{max(times):.3f}"

    # Validate individual operation performance
    execution_times = [r["execution_time_ms"] for r in results]
    avg_time = sum(execution_times) / len(execution_times)
    max_time = max(execution_times)

    assert (
        avg_time < PERFORMANCE_THRESHOLDS["medium_file_avg"]
    ), f"Concurrent avg time too slow: {avg_time:.2f}ms > {PERFORMANCE_THRESHOLDS['medium_file_avg']}ms"

    assert (
        max_time < PERFORMANCE_THRESHOLDS["small_file_p99"]
    ), f"Concurrent max time too slow: {max_time:.2f}ms > {PERFORMANCE_THRESHOLDS['small_file_p99']}ms"


# ===== Performance Summary Test =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_performance_summary(
    hash_generator, small_file_data, medium_file_data, large_file_data
):
    """
    Generate comprehensive performance summary for all hash operations.

    This test provides a complete overview of BLAKE3 hash generation
    performance across different file sizes and operation types.
    """
    results = {}

    # Test small file
    start = time.perf_counter()
    small_result = await hash_generator.generate_hash(small_file_data)
    results["Small File (< 1KB)"] = (time.perf_counter() - start) * 1000

    # Test medium file
    start = time.perf_counter()
    medium_result = await hash_generator.generate_hash(medium_file_data)
    results["Medium File (100KB)"] = (time.perf_counter() - start) * 1000

    # Test large file
    start = time.perf_counter()
    large_result = await hash_generator.generate_hash(large_file_data)
    results["Large File (2MB)"] = (time.perf_counter() - start) * 1000

    # Test batch operations
    batch_data = [b"batch_" + str(i).encode() * 50 for i in range(50)]
    start = time.perf_counter()
    batch_results = await hash_generator.batch_generate_hashes(batch_data)
    batch_time = (time.perf_counter() - start) * 1000
    results["Batch (50 files)"] = batch_time

    # Calculate batch throughput
    batch_throughput = 50 / (batch_time / 1000)

    # Print summary
    print("\n=== BLAKE3 Hash Generation Performance Summary ===")
    for operation, time_ms in results.items():
        if "Batch" in operation:
            print(f"✓ {operation}: {time_ms:.2f}ms ({batch_throughput:.0f} hashes/sec)")
        else:
            threshold = PERFORMANCE_THRESHOLDS.get("small_file_p99", float("inf"))
            status = "✓" if time_ms < threshold else "✗"
            print(f"{status} {operation}: {time_ms:.2f}ms")

    # Validate critical thresholds
    assert (
        results["Small File (< 1KB)"] < PERFORMANCE_THRESHOLDS["small_file_p99"]
    ), "Small file performance degradation"

    assert (
        results["Medium File (100KB)"] < PERFORMANCE_THRESHOLDS["medium_file_p99"]
    ), "Medium file performance degradation"

    assert (
        batch_throughput >= PERFORMANCE_THRESHOLDS["batch_throughput"]
    ), "Batch throughput too low"


# ===== Hash Consistency Test =====


@pytest.mark.performance
@pytest.mark.asyncio
async def test_hash_consistency(hash_generator, small_file_data):
    """
    Validate that hash generation is consistent across multiple runs.

    This test ensures deterministic hash generation, which is critical
    for content verification and deduplication use cases.
    """
    # Generate hash 10 times for the same data
    hashes = []
    for _ in range(10):
        result = await hash_generator.generate_hash(small_file_data)
        hashes.append(result["hash"])

    # All hashes should be identical
    assert (
        len(set(hashes)) == 1
    ), f"Hash inconsistency detected: {len(set(hashes))} unique hashes from 10 runs"

    # Verify hash format
    assert all(
        len(h) == 64 for h in hashes
    ), "Invalid hash format (expected 64 hex characters)"
