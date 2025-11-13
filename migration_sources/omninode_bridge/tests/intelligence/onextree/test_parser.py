"""
Tests for YAML parser.
"""

from pathlib import Path

from omninode_bridge.intelligence.onextree.models import (
    ModelOnextreeRoot,
    OnexTreeNode,
    ProjectStatistics,
)
from omninode_bridge.intelligence.onextree.parser import ToolOnextreeProcessor


def test_write_and_read_onextree_file(tmp_path: Path):
    """Test writing and reading .onextree file."""
    # Create sample tree
    root_node = OnexTreeNode(path="", name="project", type="directory")
    tree_root = ModelOnextreeRoot(
        project_root=str(tmp_path),
        tree=root_node,
    )

    # Write to file
    onextree_path = tmp_path / ".onextree"
    processor = ToolOnextreeProcessor()
    success = processor.write_onextree_file(tree_root, onextree_path)

    assert success is True
    assert onextree_path.exists()

    # Read from file
    loaded_tree = processor.parse_onextree_file(onextree_path)

    assert loaded_tree is not None
    assert loaded_tree.project_root == str(tmp_path)
    assert loaded_tree.tree.name == "project"
    assert loaded_tree.version == "1.0.0"


def test_write_complex_tree(tmp_path: Path):
    """Test writing and reading complex tree structure."""
    # Create complex tree
    child1 = OnexTreeNode(
        path="src/file1.py",
        name="file1.py",
        type="file",
        size=1024,
        extension="py",
    )
    child2 = OnexTreeNode(
        path="src/file2.py",
        name="file2.py",
        type="file",
        size=2048,
        extension="py",
    )

    root_node = OnexTreeNode(
        path="", name="project", type="directory", children=[child1, child2]
    )

    stats = ProjectStatistics(
        total_files=2,
        total_directories=1,
        file_type_distribution={"py": 2},
        total_size_bytes=3072,
    )

    tree_root = ModelOnextreeRoot(
        project_root=str(tmp_path), tree=root_node, statistics=stats
    )

    # Write and read
    onextree_path = tmp_path / ".onextree"
    processor = ToolOnextreeProcessor()
    processor.write_onextree_file(tree_root, onextree_path)

    loaded_tree = processor.parse_onextree_file(onextree_path)

    assert loaded_tree is not None
    assert loaded_tree.statistics.total_files == 2
    assert loaded_tree.statistics.file_type_distribution["py"] == 2
    assert len(loaded_tree.tree.children) == 2


def test_parse_nonexistent_file(tmp_path: Path):
    """Test parsing non-existent file returns None."""
    processor = ToolOnextreeProcessor()
    result = processor.parse_onextree_file(tmp_path / "missing.yaml")

    assert result is None


def test_parse_invalid_yaml(tmp_path: Path):
    """Test parsing invalid YAML returns None."""
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("{ invalid yaml content [")

    processor = ToolOnextreeProcessor()
    result = processor.parse_onextree_file(invalid_file)

    assert result is None


def test_parse_invalid_schema(tmp_path: Path):
    """Test parsing valid YAML with invalid schema returns None."""
    invalid_file = tmp_path / "invalid_schema.yaml"
    invalid_file.write_text(
        """
version: 1.0.0
missing_required_fields: true
    """
    )

    processor = ToolOnextreeProcessor()
    result = processor.parse_onextree_file(invalid_file)

    assert result is None


def test_validate_schema():
    """Test schema validation."""
    processor = ToolOnextreeProcessor()

    # Valid schema
    valid_data = {
        "version": "1.0.0",
        "project_root": "/path",
        "generated_at": "2024-01-01T00:00:00",
        "tree": {"path": "", "name": "root", "type": "directory"},
        "statistics": {
            "total_files": 0,
            "total_directories": 0,
            "file_type_distribution": {},
            "total_size_bytes": 0,
            "last_updated": "2024-01-01T00:00:00",
        },
    }

    assert processor.validate_onextree_schema(valid_data) is True

    # Invalid schema
    invalid_data = {"version": "1.0.0"}  # Missing required fields

    assert processor.validate_onextree_schema(invalid_data) is False


def test_write_creates_parent_directory(tmp_path: Path):
    """Test that write_onextree_file creates parent directories."""
    nested_path = tmp_path / "nested" / "dir" / ".onextree"

    root_node = OnexTreeNode(path="", name="project", type="directory")
    tree_root = ModelOnextreeRoot(project_root=str(tmp_path), tree=root_node)

    processor = ToolOnextreeProcessor()
    success = processor.write_onextree_file(tree_root, nested_path)

    assert success is True
    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_write_with_metadata(tmp_path: Path):
    """Test writing tree with custom metadata."""
    root_node = OnexTreeNode(path="", name="project", type="directory")
    tree_root = ModelOnextreeRoot(
        project_root=str(tmp_path),
        tree=root_node,
        metadata={"custom": "data", "number": 42},
    )

    onextree_path = tmp_path / ".onextree"
    processor = ToolOnextreeProcessor()
    processor.write_onextree_file(tree_root, onextree_path)

    loaded_tree = processor.parse_onextree_file(onextree_path)

    assert loaded_tree.metadata["custom"] == "data"
    assert loaded_tree.metadata["number"] == 42
