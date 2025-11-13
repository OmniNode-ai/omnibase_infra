"""
Tests for OnexTree integration layer (Phase 2).

Tests adapter classes for integrating with:
- File processor (context enrichment)
- Batch processing system
- Event publishing (Kafka)
"""

import pytest

from omninode_bridge.intelligence.onextree.generator import OnexTreeGenerator
from omninode_bridge.intelligence.onextree.integration import (
    OnexTreeBatchProcessingIntegration,
    OnexTreeEventPublisher,
    OnexTreeIntegrationManager,
    OnexTreeUnifiedFileProcessorIntegration,
)
from omninode_bridge.intelligence.onextree.query_engine import OnexTreeQueryEngine


@pytest.mark.asyncio
async def test_file_processor_integration_enrich_context(sample_project):
    """Test file processor context enrichment."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeUnifiedFileProcessorIntegration(engine)

    # Test enriching context for existing file
    context = await integration.enrich_file_context("src/services/api.py")

    assert "onextree_context" in context
    assert context["onextree_context"]["exists_in_tree"] is True
    assert "file_metadata" in context["onextree_context"]
    assert context["onextree_context"]["file_metadata"]["name"] == "api.py"
    assert "structural_context" in context["onextree_context"]
    assert "semantic_context" in context["onextree_context"]
    assert "project_context" in context["onextree_context"]


@pytest.mark.asyncio
async def test_file_processor_integration_file_not_found(sample_project):
    """Test context enrichment for non-existent file."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeUnifiedFileProcessorIntegration(engine)

    # Test enriching context for non-existent file
    context = await integration.enrich_file_context("nonexistent/file.py")

    assert "onextree_context" in context
    assert context["onextree_context"]["exists_in_tree"] is False
    assert context["onextree_context"]["status"] == "file_not_found"


@pytest.mark.asyncio
async def test_file_processor_integration_validate_path(sample_project):
    """Test file path validation."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeUnifiedFileProcessorIntegration(engine)

    # Test validating existing file
    result = await integration.validate_file_path("src/services/api.py")

    assert result["path"] == "src/services/api.py"
    assert result["exists"] is True
    assert result["is_valid"] is False  # Already exists
    assert "warning" in result

    # Test validating new file
    result = await integration.validate_file_path("src/new_file.py")

    assert result["path"] == "src/new_file.py"
    assert result["exists"] is False
    assert result["is_valid"] is True


@pytest.mark.asyncio
async def test_file_processor_integration_with_base_context(sample_project):
    """Test enriching existing context."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeUnifiedFileProcessorIntegration(engine)

    # Test with existing base context
    base_context = {"existing_key": "existing_value"}
    context = await integration.enrich_file_context("src/services/api.py", base_context)

    assert "existing_key" in context
    assert context["existing_key"] == "existing_value"
    assert "onextree_context" in context


@pytest.mark.asyncio
async def test_batch_processing_integration_get_files(sample_project):
    """Test batch processing file list retrieval."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeBatchProcessingIntegration(engine)

    # Test getting all files
    files = await integration.get_files_for_batch_processing(max_files=100)

    assert len(files) > 0
    assert all("path" in f for f in files)
    assert all("name" in f for f in files)


@pytest.mark.asyncio
async def test_batch_processing_integration_filter_by_type(sample_project):
    """Test batch processing with file type filter."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeBatchProcessingIntegration(engine)

    # Test filtering by file type
    files = await integration.get_files_for_batch_processing(
        file_types=["py"], max_files=100
    )

    assert len(files) > 0
    assert all(f["extension"] == "py" for f in files)


@pytest.mark.asyncio
async def test_batch_processing_integration_filter_by_directory(sample_project):
    """Test batch processing with directory filter."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeBatchProcessingIntegration(engine)

    # Test filtering by directory
    files = await integration.get_files_for_batch_processing(
        directory_filter="src/services", max_files=100
    )

    assert len(files) > 0
    assert all(f["path"].startswith("src/services") for f in files)


@pytest.mark.asyncio
async def test_batch_processing_integration_statistics(sample_project):
    """Test batch processing statistics."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeBatchProcessingIntegration(engine)

    # Test getting statistics
    stats = await integration.get_batch_statistics()

    assert stats["available"] is True
    assert "total_files" in stats
    assert "total_directories" in stats
    assert "file_type_distribution" in stats
    assert stats["total_files"] > 0


@pytest.mark.asyncio
async def test_batch_processing_integration_statistics_filtered(sample_project):
    """Test batch processing statistics with filter."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    integration = OnexTreeBatchProcessingIntegration(engine)

    # Test getting filtered statistics
    stats = await integration.get_batch_statistics(file_types=["py", "md"])

    assert stats["available"] is True
    assert "filtered_count" in stats
    assert "filter_applied" in stats
    assert stats["filter_applied"] == ["py", "md"]


@pytest.mark.asyncio
async def test_event_publisher_disabled():
    """Test event publisher with no backend (disabled mode)."""
    publisher = OnexTreeEventPublisher(event_publisher=None)

    assert publisher.enabled is False

    # Should not raise, just return False
    result = await publisher.publish_tree_generated_event(None, 100.0)
    assert result is False


@pytest.mark.asyncio
async def test_event_publisher_tree_generated(sample_project):
    """Test publishing tree generated event."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    # Mock event publisher
    class MockEventPublisher:
        def __init__(self):
            self.published_events = []

        async def publish(self, topic, data):
            self.published_events.append({"topic": topic, "data": data})

    mock_publisher = MockEventPublisher()
    publisher = OnexTreeEventPublisher(event_publisher=mock_publisher)

    assert publisher.enabled is True

    # Publish event
    result = await publisher.publish_tree_generated_event(tree_root, 123.45)

    assert result is True
    assert len(mock_publisher.published_events) == 1
    assert mock_publisher.published_events[0]["topic"] == "onextree.tree.generated"
    assert (
        mock_publisher.published_events[0]["data"]["event_type"]
        == "onextree.tree.generated"
    )
    assert (
        mock_publisher.published_events[0]["data"]["performance"]["generation_time_ms"]
        == 123.45
    )


@pytest.mark.asyncio
async def test_event_publisher_tree_updated(sample_project):
    """Test publishing tree updated event."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    # Mock event publisher
    class MockEventPublisher:
        def __init__(self):
            self.published_events = []

        async def publish(self, topic, data):
            self.published_events.append({"topic": topic, "data": data})

    mock_publisher = MockEventPublisher()
    publisher = OnexTreeEventPublisher(event_publisher=mock_publisher)

    # Publish event
    result = await publisher.publish_tree_updated_event(
        tree_root, "filesystem_change", ["src/new_file.py"]
    )

    assert result is True
    assert len(mock_publisher.published_events) == 1
    assert mock_publisher.published_events[0]["topic"] == "onextree.tree.updated"
    assert mock_publisher.published_events[0]["data"]["trigger"] == "filesystem_change"
    assert (
        "src/new_file.py" in mock_publisher.published_events[0]["data"]["files_changed"]
    )


@pytest.mark.asyncio
async def test_event_publisher_query_metrics():
    """Test publishing query metrics event."""

    # Mock event publisher
    class MockEventPublisher:
        def __init__(self):
            self.published_events = []

        async def publish(self, topic, data):
            self.published_events.append({"topic": topic, "data": data})

    mock_publisher = MockEventPublisher()
    publisher = OnexTreeEventPublisher(event_publisher=mock_publisher)

    # Publish metrics
    result = await publisher.publish_query_metrics_event("lookup_file", 0.5, 1)

    assert result is True
    assert len(mock_publisher.published_events) == 1
    assert mock_publisher.published_events[0]["topic"] == "onextree.query.metrics"
    assert mock_publisher.published_events[0]["data"]["operation"] == "lookup_file"
    assert mock_publisher.published_events[0]["data"]["execution_time_ms"] == 0.5


@pytest.mark.asyncio
async def test_integration_manager_all_disabled(sample_project):
    """Test integration manager with all features disabled."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    manager = OnexTreeIntegrationManager(engine)

    assert manager.is_file_processor_enabled() is False
    assert manager.is_batch_processing_enabled() is False
    assert manager.is_events_enabled() is False

    status = await manager.get_integration_status()
    assert status["integrations"]["file_processor"]["enabled"] is False
    assert status["integrations"]["batch_processing"]["enabled"] is False
    assert status["integrations"]["events"]["enabled"] is False


@pytest.mark.asyncio
async def test_integration_manager_all_enabled(sample_project):
    """Test integration manager with all features enabled."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Mock event publisher
    class MockEventPublisher:
        async def publish(self, topic, data):
            pass

    mock_publisher = MockEventPublisher()

    manager = OnexTreeIntegrationManager(
        engine,
        enable_file_processor=True,
        enable_batch_processing=True,
        enable_events=True,
        event_publisher=mock_publisher,
    )

    assert manager.is_file_processor_enabled() is True
    assert manager.is_batch_processing_enabled() is True
    assert manager.is_events_enabled() is True

    status = await manager.get_integration_status()
    assert status["integrations"]["file_processor"]["enabled"] is True
    assert status["integrations"]["batch_processing"]["enabled"] is True
    assert status["integrations"]["events"]["enabled"] is True
    assert status["query_engine"]["loaded"] is True
    assert status["query_engine"]["total_files"] > 0


@pytest.mark.asyncio
async def test_integration_manager_selective_enable(sample_project):
    """Test integration manager with selective feature enablement."""
    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    # Enable only file processor and batch processing
    manager = OnexTreeIntegrationManager(
        engine, enable_file_processor=True, enable_batch_processing=True
    )

    assert manager.is_file_processor_enabled() is True
    assert manager.is_batch_processing_enabled() is True
    assert manager.is_events_enabled() is False

    # Verify integrations are accessible
    assert manager.file_processor is not None
    assert manager.batch_processing is not None
    assert manager.event_publisher is None
