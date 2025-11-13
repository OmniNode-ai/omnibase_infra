"""
Tests for tree generator.
"""

from pathlib import Path

import pytest

from omninode_bridge.intelligence.onextree.generator import OnexTreeGenerator


@pytest.mark.asyncio
async def test_generate_tree_basic(sample_project: Path):
    """Test basic tree generation."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    assert tree_root is not None
    assert tree_root.project_root == str(sample_project)
    assert tree_root.tree.type == "directory"
    assert tree_root.statistics.total_files > 0
    assert tree_root.statistics.total_directories > 0


@pytest.mark.asyncio
async def test_generate_tree_statistics(sample_project: Path):
    """Test that statistics are correctly calculated."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    stats = tree_root.statistics

    # Sample project has specific structure
    assert stats.total_files >= 10  # At least 10 files
    assert stats.total_directories >= 5  # At least 5 directories
    assert "py" in stats.file_type_distribution
    assert "md" in stats.file_type_distribution
    assert stats.total_size_bytes > 0


@pytest.mark.asyncio
async def test_generate_tree_exclusions(tmp_path: Path):
    """Test that exclusion patterns work."""
    # Create project with files that should be excluded
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# Main")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_text("cached")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config")

    generator = OnexTreeGenerator(tmp_path)
    tree_root = await generator.generate_tree()

    # Convert tree to list of paths
    def collect_paths(node):
        paths = [node.path]
        if node.children:
            for child in node.children:
                paths.extend(collect_paths(child))
        return paths

    all_paths = collect_paths(tree_root.tree)

    # __pycache__ and .git should be excluded
    assert not any("__pycache__" in path for path in all_paths)
    assert not any(".git" in path for path in all_paths)
    assert any("src/main.py" in path for path in all_paths)


@pytest.mark.asyncio
async def test_generate_tree_nested_structure(tmp_path: Path):
    """Test deeply nested directory structure."""
    # Create nested structure
    nested = tmp_path / "a" / "b" / "c" / "d"
    nested.mkdir(parents=True)
    (nested / "deep_file.txt").write_text("deep")

    generator = OnexTreeGenerator(tmp_path)
    tree_root = await generator.generate_tree()

    # Verify deep nesting is captured
    def find_node(node, name):
        if node.name == name:
            return node
        if node.children:
            for child in node.children:
                result = find_node(child, name)
                if result:
                    return result
        return None

    deep_file = find_node(tree_root.tree, "deep_file.txt")
    assert deep_file is not None
    assert deep_file.type == "file"


@pytest.mark.asyncio
async def test_generate_tree_file_extensions(sample_project: Path):
    """Test that file extensions are correctly extracted."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    def collect_extensions(node):
        extensions = []
        if node.type == "file" and node.extension:
            extensions.append(node.extension)
        if node.children:
            for child in node.children:
                extensions.extend(collect_extensions(child))
        return extensions

    extensions = collect_extensions(tree_root.tree)

    assert "py" in extensions
    assert "md" in extensions
    assert "yaml" in extensions


@pytest.mark.asyncio
async def test_generate_tree_custom_exclusions(tmp_path: Path):
    """Test custom exclusion patterns."""
    (tmp_path / "include.txt").write_text("include")
    (tmp_path / "exclude.txt").write_text("exclude")
    (tmp_path / "normal.py").write_text("normal")

    # Use custom exclusion pattern
    generator = OnexTreeGenerator(tmp_path, exclude_patterns=["exclude.txt"])
    tree_root = await generator.generate_tree()

    def collect_names(node):
        names = [node.name]
        if node.children:
            for child in node.children:
                names.extend(collect_names(child))
        return names

    names = collect_names(tree_root.tree)

    assert "include.txt" in names
    assert "normal.py" in names
    assert "exclude.txt" not in names


@pytest.mark.asyncio
async def test_generate_tree_empty_directory(tmp_path: Path):
    """Test generating tree for empty directory."""
    generator = OnexTreeGenerator(tmp_path)
    tree_root = await generator.generate_tree()

    assert tree_root.statistics.total_files == 0
    assert tree_root.statistics.total_directories == 0
    assert len(tree_root.tree.children) == 0


@pytest.mark.asyncio
async def test_generate_tree_file_metadata(sample_project: Path):
    """Test that file metadata is captured."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    def find_file(node, name):
        if node.name == name and node.type == "file":
            return node
        if node.children:
            for child in node.children:
                result = find_file(child, name)
                if result:
                    return result
        return None

    # Find a specific file
    api_file = find_file(tree_root.tree, "api.py")

    assert api_file is not None
    assert api_file.size is not None
    assert api_file.size > 0
    assert api_file.extension == "py"
    assert api_file.last_modified is not None
