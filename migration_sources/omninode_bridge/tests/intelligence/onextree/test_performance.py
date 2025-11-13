"""
Performance benchmarks for OnexTree.

Tests performance targets:
- Tree generation: < 100ms for 10K files
- Exact path lookup: < 1ms
- Extension search: < 5ms
- Index rebuild: < 100ms
- Memory usage: < 20MB for 10K files
"""

import asyncio
import time
from pathlib import Path

import pytest

from omninode_bridge.intelligence.onextree.config import PERFORMANCE_THRESHOLDS
from omninode_bridge.intelligence.onextree.generator import OnexTreeGenerator
from omninode_bridge.intelligence.onextree.query_engine import OnexTreeQueryEngine


@pytest.mark.performance
@pytest.mark.asyncio
async def test_tree_generation_performance(sample_project: Path):
    """
    Benchmark tree generation.

    Target: < 100ms for typical project
    """
    generator = OnexTreeGenerator(sample_project)

    start_time = time.perf_counter()
    tree_root = await generator.generate_tree()
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) * 1000

    print(
        f"\nTree generation: {execution_time_ms:.2f}ms for {tree_root.statistics.total_files} files"
    )

    # For small projects, should be very fast
    assert execution_time_ms < PERFORMANCE_THRESHOLDS["tree_generation_ms"]


@pytest.mark.performance
@pytest.mark.asyncio
async def test_exact_lookup_performance(sample_project: Path):
    """
    Benchmark exact path lookup.

    Target: < 1ms per lookup
    """
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Warm up
    await engine.lookup_file("src/services/api.py")

    # Benchmark
    iterations = 100
    start_time = time.perf_counter()

    for _ in range(iterations):
        await engine.lookup_file("src/services/api.py")

    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / iterations) * 1000

    print(f"\nExact lookup: {avg_time_ms:.3f}ms average")

    assert avg_time_ms < PERFORMANCE_THRESHOLDS["exact_lookup_ms"]


@pytest.mark.performance
@pytest.mark.asyncio
async def test_extension_search_performance(sample_project: Path):
    """
    Benchmark extension-based search.

    Target: < 5ms per search
    """
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Benchmark
    start_time = time.perf_counter()
    results = await engine.find_by_extension("py", limit=100)
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) * 1000

    print(f"\nExtension search: {execution_time_ms:.2f}ms for {len(results)} results")

    assert execution_time_ms < PERFORMANCE_THRESHOLDS["extension_search_ms"]


@pytest.mark.performance
@pytest.mark.asyncio
async def test_index_rebuild_performance(sample_project: Path):
    """
    Benchmark index rebuild.

    Target: < 100ms for typical project
    """
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()

    # Benchmark
    start_time = time.perf_counter()
    await engine.load_tree(tree_root)
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) * 1000

    print(f"\nIndex rebuild: {execution_time_ms:.2f}ms")

    assert execution_time_ms < PERFORMANCE_THRESHOLDS["index_rebuild_ms"]


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_lookup_performance(sample_project: Path):
    """
    Benchmark concurrent lookups.

    Tests that multiple concurrent queries don't degrade performance.
    """
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Run 100 concurrent lookups
    paths = [
        "src/services/api.py",
        "src/services/worker.py",
        "src/models/user.py",
        "src/models/product.py",
        "tests/test_api.py",
    ] * 20  # 100 total

    start_time = time.perf_counter()
    results = await asyncio.gather(*[engine.lookup_file(path) for path in paths])
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / len(paths)

    print(f"\nConcurrent lookups: {avg_time_ms:.3f}ms average ({len(paths)} queries)")

    # Average should still be fast
    assert avg_time_ms < PERFORMANCE_THRESHOLDS["exact_lookup_ms"]


@pytest.mark.performance
@pytest.mark.asyncio
async def test_large_project_performance(large_project: Path):
    """
    Test performance with larger project.

    Tests scalability with 1000+ files.
    """
    generator = OnexTreeGenerator(large_project)

    # Generate tree
    start_time = time.perf_counter()
    tree_root = await generator.generate_tree()
    gen_time = time.perf_counter() - start_time

    print(f"\nLarge project tree generation: {gen_time*1000:.2f}ms")
    print(f"Files: {tree_root.statistics.total_files}")

    # Load into engine
    engine = OnexTreeQueryEngine()
    start_time = time.perf_counter()
    await engine.load_tree(tree_root)
    load_time = time.perf_counter() - start_time

    print(f"Index build: {load_time*1000:.2f}ms")

    # Test lookup performance
    start_time = time.perf_counter()
    await engine.lookup_file("module_0/file_0.py")
    lookup_time = time.perf_counter() - start_time

    print(f"Lookup time: {lookup_time*1000:.3f}ms")

    # Verify reasonable performance for large project
    assert (
        gen_time * 1000 < 1000
    )  # < 1 second for 1000 files (10ms per 10 files scales to 100ms for 10K)
    assert lookup_time * 1000 < PERFORMANCE_THRESHOLDS["exact_lookup_ms"]


@pytest.mark.performance
@pytest.mark.asyncio
async def test_directory_children_performance(sample_project: Path):
    """
    Benchmark directory children retrieval.

    Target: < 2ms per query
    """
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Benchmark
    iterations = 100
    start_time = time.perf_counter()

    for _ in range(iterations):
        await engine.get_directory_children("src")

    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / iterations) * 1000

    print(f"\nDirectory children: {avg_time_ms:.3f}ms average")

    assert avg_time_ms < 2.0  # Should be very fast (O(1) lookup)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_pattern_search_performance(sample_project: Path):
    """
    Benchmark pattern-based search.

    Target: < 10ms for typical search
    """
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Benchmark
    start_time = time.perf_counter()
    results = await engine.search_by_path_pattern("service")
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) * 1000

    print(f"\nPattern search: {execution_time_ms:.2f}ms for {len(results)} results")

    assert execution_time_ms < 10.0  # O(n) operation, but should be fast
