"""
Tests for Pydantic models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from omninode_bridge.intelligence.onextree.models import (
    ModelOnextreeRoot,
    OnexTreeNode,
    ProjectStatistics,
)


def test_onextree_node_file():
    """Test creating a file node."""
    node = OnexTreeNode(
        path="src/models/user.py",
        name="user.py",
        type="file",
        size=1024,
        extension="py",
        last_modified=datetime.now(),
    )

    assert node.path == "src/models/user.py"
    assert node.name == "user.py"
    assert node.type == "file"
    assert node.size == 1024
    assert node.extension == "py"
    assert node.last_modified is not None


def test_onextree_node_directory():
    """Test creating a directory node with children."""
    child1 = OnexTreeNode(path="src/file1.py", name="file1.py", type="file")
    child2 = OnexTreeNode(path="src/file2.py", name="file2.py", type="file")

    parent = OnexTreeNode(
        path="src", name="src", type="directory", children=[child1, child2]
    )

    assert parent.type == "directory"
    assert len(parent.children) == 2
    assert parent.children[0].name == "file1.py"


def test_onextree_node_validation():
    """Test Pydantic validation."""
    # Missing required fields should raise error
    with pytest.raises(ValidationError):
        OnexTreeNode(name="test.py")  # Missing path and type


def test_onextree_node_semantic_metadata():
    """Test semantic metadata fields."""
    node = OnexTreeNode(
        path="src/services/api.py",
        name="api.py",
        type="file",
        inferred_purpose="service",
        architectural_pattern="service",
        related_files=["src/models/user.py", "src/utils/helper.py"],
    )

    assert node.inferred_purpose == "service"
    assert node.architectural_pattern == "service"
    assert len(node.related_files) == 2


def test_project_statistics():
    """Test project statistics model."""
    stats = ProjectStatistics(
        total_files=100,
        total_directories=20,
        file_type_distribution={"py": 80, "md": 10, "yaml": 10},
        total_size_bytes=1024000,
    )

    assert stats.total_files == 100
    assert stats.total_directories == 20
    assert stats.file_type_distribution["py"] == 80
    assert stats.total_size_bytes == 1024000
    assert stats.last_updated is not None


def test_model_onextree_root():
    """Test root model."""
    root_node = OnexTreeNode(path="", name="project", type="directory")

    tree_root = ModelOnextreeRoot(
        project_root="/path/to/project",
        tree=root_node,
    )

    assert tree_root.version == "1.0.0"
    assert tree_root.project_root == "/path/to/project"
    assert tree_root.tree.name == "project"
    assert tree_root.generated_at is not None
    assert tree_root.statistics.total_files == 0


def test_model_onextree_root_with_metadata():
    """Test root model with custom metadata."""
    root_node = OnexTreeNode(path="", name="project", type="directory")

    tree_root = ModelOnextreeRoot(
        project_root="/path/to/project",
        tree=root_node,
        metadata={"git_commit": "abc123", "custom_field": "value"},
    )

    assert tree_root.metadata["git_commit"] == "abc123"
    assert tree_root.metadata["custom_field"] == "value"


def test_model_serialization():
    """Test model serialization to dict."""
    node = OnexTreeNode(
        path="src/test.py",
        name="test.py",
        type="file",
        size=100,
        extension="py",
    )

    data = node.model_dump()

    assert data["path"] == "src/test.py"
    assert data["name"] == "test.py"
    assert data["type"] == "file"
    assert data["size"] == 100


def test_model_json_serialization():
    """Test JSON mode serialization (for datetime handling)."""
    now = datetime.now()
    node = OnexTreeNode(
        path="src/test.py", name="test.py", type="file", last_modified=now
    )

    data = node.model_dump(mode="json")

    # DateTime should be serialized as ISO string
    assert isinstance(data["last_modified"], str)
    assert "T" in data["last_modified"]
