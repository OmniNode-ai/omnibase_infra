"""
Tests for query engine.
"""

from pathlib import Path

import pytest

from omninode_bridge.intelligence.onextree.generator import OnexTreeGenerator
from omninode_bridge.intelligence.onextree.query_engine import OnexTreeQueryEngine


@pytest.mark.asyncio
async def test_lookup_file_found(sample_project: Path):
    """Test looking up existing file."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Look up existing file
    node = await engine.lookup_file("src/services/api.py")

    assert node is not None
    assert node.name == "api.py"
    assert node.type == "file"
    assert node.extension == "py"


@pytest.mark.asyncio
async def test_lookup_file_not_found(sample_project: Path):
    """Test looking up non-existent file."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    node = await engine.lookup_file("nonexistent/file.py")

    assert node is None


@pytest.mark.asyncio
async def test_check_file_exists(sample_project: Path):
    """Test existence check."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    assert await engine.check_file_exists("src/services/api.py") is True
    assert await engine.check_file_exists("nonexistent.py") is False


@pytest.mark.asyncio
async def test_find_by_extension(sample_project: Path):
    """Test finding files by extension."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Find all Python files
    py_files = await engine.find_by_extension("py")

    assert len(py_files) > 0
    assert all(node.extension == "py" for node in py_files)
    assert any(node.name == "api.py" for node in py_files)


@pytest.mark.asyncio
async def test_find_by_extension_with_limit(sample_project: Path):
    """Test extension search with limit."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Find with limit
    py_files = await engine.find_by_extension("py", limit=3)

    assert len(py_files) <= 3


@pytest.mark.asyncio
async def test_find_by_name(sample_project: Path):
    """Test finding files by exact name."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Find files named "api.py"
    results = await engine.find_by_name("api.py")

    assert len(results) >= 1
    assert all(node.name == "api.py" for node in results)


@pytest.mark.asyncio
async def test_find_similar_names(sample_project: Path):
    """Test finding files with similar names."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Find files with "test" in name
    results = await engine.find_similar_names("test")

    assert len(results) > 0
    assert all("test" in node.name.lower() for node in results)


@pytest.mark.asyncio
async def test_get_directory_children(sample_project: Path):
    """Test getting directory children."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Get children of src/services
    children = await engine.get_directory_children("src/services")

    assert len(children) > 0
    assert any(child.name == "api.py" for child in children)
    assert any(child.name == "worker.py" for child in children)


@pytest.mark.asyncio
async def test_get_directory_children_empty(sample_project: Path):
    """Test getting children of directory with no direct children."""
    # Create empty directory
    (sample_project / "empty").mkdir()

    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    children = await engine.get_directory_children("empty")

    assert len(children) == 0


@pytest.mark.asyncio
async def test_get_statistics(sample_project: Path):
    """Test getting tree statistics."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    stats = await engine.get_statistics()

    assert stats is not None
    assert "total_files" in stats
    assert "total_directories" in stats
    assert "file_type_distribution" in stats
    assert "index_sizes" in stats
    assert stats["total_files"] > 0


@pytest.mark.asyncio
async def test_search_by_path_pattern(sample_project: Path):
    """Test searching by path pattern."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Search for files in services directory
    results = await engine.search_by_path_pattern("services")

    assert len(results) > 0
    assert all("services" in node.path.lower() for node in results)


@pytest.mark.asyncio
async def test_index_rebuild(sample_project: Path):
    """Test that indexes are rebuilt correctly on load."""
    generator = OnexTreeGenerator(sample_project)
    tree_root1 = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root1)

    # Get initial stats
    initial_files = len(engine.exact_path_index)

    # Load new tree
    tree_root2 = await generator.generate_tree()
    await engine.load_tree(tree_root2)

    # Stats should be same (same tree)
    assert len(engine.exact_path_index) == initial_files


@pytest.mark.asyncio
async def test_concurrent_queries(sample_project: Path):
    """Test concurrent query operations."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Run multiple queries concurrently
    import asyncio

    results = await asyncio.gather(
        engine.lookup_file("src/services/api.py"),
        engine.find_by_extension("py"),
        engine.get_directory_children("src"),
        engine.check_file_exists("README.md"),
    )

    assert results[0] is not None  # lookup_file
    assert len(results[1]) > 0  # find_by_extension
    assert len(results[2]) > 0  # get_directory_children
    assert results[3] is True  # check_file_exists


@pytest.mark.asyncio
async def test_query_engine_empty_tree(tmp_path: Path):
    """Test query engine with empty tree."""
    generator = OnexTreeGenerator(tmp_path)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # All queries should return empty/None
    assert await engine.lookup_file("any/file.py") is None
    assert await engine.check_file_exists("any/file.py") is False
    assert len(await engine.find_by_extension("py")) == 0
    assert len(await engine.get_directory_children("")) == 0
