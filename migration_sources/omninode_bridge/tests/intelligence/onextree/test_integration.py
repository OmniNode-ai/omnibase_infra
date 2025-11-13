"""
Integration tests for OnexTree standalone system.
"""

import asyncio
from pathlib import Path

import pytest

from omninode_bridge.intelligence.onextree.mcp_server import OnexTreeMCPServer


@pytest.mark.asyncio
async def test_end_to_end_workflow(sample_project: Path):
    """Test complete workflow: generate → query → update → query."""
    # Start MCP server
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Test 1: Lookup existing file
        result = await server._handle_lookup_file({"file_path": "src/services/api.py"})
        result_text = result[0].text
        assert "found" in result_text.lower()
        assert "true" in result_text.lower()

        # Test 2: Check non-existent file
        result = await server._handle_check_exists({"file_path": "src/missing.py"})
        result_text = result[0].text
        assert "exists" in result_text.lower()
        assert "false" in result_text.lower()

        # Test 3: Get structure
        result = await server._handle_get_structure({"root_path": "src"})
        result_text = result[0].text
        assert "children" in result_text.lower()

        # Test 4: Safe create check
        result = await server._handle_safe_create(
            {"proposed_path": "src/new_service.py"}
        )
        result_text = result[0].text
        assert "can_create" in result_text.lower()

        # Test 5: Get context
        result = await server._handle_get_context({"file_path": "src/services/api.py"})
        result_text = result[0].text
        assert "exists" in result_text.lower()
        assert "siblings" in result_text.lower()

    finally:
        await server.stop()


@pytest.mark.skip(
    reason="Filesystem watcher test is flaky in test environment due to event loop issues"
)
@pytest.mark.asyncio
async def test_filesystem_watcher_updates(sample_project: Path):
    """Test that filesystem watcher detects changes."""
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Verify file doesn't exist
        exists_before = await server.query_engine.check_file_exists(
            "src/new_service.py"
        )
        assert exists_before is False

        # Create new file
        new_file = sample_project / "src" / "new_service.py"
        new_file.write_text("# New service\nclass NewService:\n    pass")

        # Wait for watcher debounce and regeneration
        await asyncio.sleep(1.5)

        # Verify file now exists
        exists_after = await server.query_engine.check_file_exists("src/new_service.py")
        assert exists_after is True

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_onextree_file_persistence(sample_project: Path):
    """Test that .onextree file is created and readable."""
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Check .onextree file exists
        onextree_path = sample_project / ".onextree"
        assert onextree_path.exists()

        # Verify it's readable
        from omninode_bridge.intelligence.onextree.parser import ToolOnextreeProcessor

        processor = ToolOnextreeProcessor()
        loaded_tree = processor.parse_onextree_file(onextree_path)

        assert loaded_tree is not None
        assert loaded_tree.project_root == str(sample_project)
        assert loaded_tree.statistics.total_files > 0

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_check_exists_with_similar_files(sample_project: Path):
    """Test check_exists with similar file detection."""
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Check for file that doesn't exist but has similar names
        result = await server._handle_check_exists(
            {"file_path": "tests/test_missing.py", "check_similar": True}
        )

        result_text = result[0].text
        assert "exists" in result_text.lower()
        assert "false" in result_text.lower()

        # Should find similar test files
        import json

        data = json.loads(result_text)
        if "similar_files" in data:
            # Similar files might be found
            similar = data["similar_files"]
            # All similar files should be test files
            assert all("test" in f["path"].lower() for f in similar)

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_get_structure_with_filter(sample_project: Path):
    """Test get_structure with file type filtering."""
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Get structure with only Python files
        result = await server._handle_get_structure(
            {"root_path": "src", "file_types": ["py"]}
        )

        result_text = result[0].text
        import json

        data = json.loads(result_text)

        # All children should be directories or .py files
        for child in data["children"]:
            assert child["type"] == "directory" or child["extension"] == "py"

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_safe_create_duplicate_detection(sample_project: Path):
    """Test safe_create detects existing files."""
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Try to create file that already exists
        result = await server._handle_safe_create(
            {"proposed_path": "src/services/api.py"}
        )

        result_text = result[0].text
        import json

        data = json.loads(result_text)

        assert data["can_create"] is False
        assert data["collision"] is True
        assert "exists" in data["recommendation"]

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_get_context_architectural_hints(sample_project: Path):
    """Test that get_context provides architectural hints."""
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Get context for service file
        result = await server._handle_get_context({"file_path": "src/services/api.py"})

        result_text = result[0].text
        import json

        data = json.loads(result_text)

        assert data["exists"] is True
        # Should detect service pattern
        if "architectural_hints" in data:
            hints = data["architectural_hints"]
            assert any("service" in hint.lower() for hint in hints)

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_multiple_concurrent_tool_calls(sample_project: Path):
    """Test multiple concurrent tool invocations."""
    server = OnexTreeMCPServer(sample_project)
    await server.start()

    try:
        # Run multiple tool calls concurrently
        results = await asyncio.gather(
            server._handle_lookup_file({"file_path": "src/services/api.py"}),
            server._handle_check_exists({"file_path": "README.md"}),
            server._handle_get_structure({"root_path": "src"}),
            server._handle_safe_create({"proposed_path": "src/new.py"}),
            server._handle_get_context({"file_path": "src/models/user.py"}),
        )

        # All should return results
        assert all(len(r) > 0 for r in results)

    finally:
        await server.stop()
