"""
Unit tests for batch_sizes module.

Tests the centralized batch size configuration functionality.
"""

from omninode_bridge.config.batch_sizes import (
    BatchSizeManager,
    get_batch_manager,
    get_cleanup_batch_size,
    get_database_batch_size,
    get_file_processing_batch_size,
    get_kafka_batch_size,
    get_orchestrator_batch_size,
    get_redis_batch_size,
    get_reducer_batch_size,
)


class TestBatchSizeManager:
    """Test BatchSizeManager functionality."""

    def test_batch_size_manager_initialization(self):
        """Test BatchSizeManager can be initialized."""
        manager = BatchSizeManager()
        assert manager.environment == "development"
        assert manager.config is not None

    def test_batch_size_manager_with_environment(self):
        """Test BatchSizeManager with different environments."""
        for env in ["development", "test", "staging", "production"]:
            manager = BatchSizeManager(environment=env)
            assert manager.environment == env
            assert manager.config is not None

    def test_database_batch_sizes(self):
        """Test database batch size properties."""
        manager = BatchSizeManager(environment="test")

        assert manager.database_batch_size > 0
        assert manager.database_query_limit > 0
        assert manager.database_statement_cache_size > 0

    def test_kafka_batch_sizes(self):
        """Test Kafka batch size properties."""
        manager = BatchSizeManager(environment="test")

        assert manager.kafka_producer_batch_size > 0
        assert manager.kafka_consumer_max_poll_records > 0
        assert manager.kafka_consumer_fetch_min_bytes > 0
        assert manager.kafka_consumer_fetch_max_bytes > 0

    def test_redis_batch_sizes(self):
        """Test Redis batch size properties."""
        manager = BatchSizeManager(environment="test")

        assert manager.redis_batch_size > 0
        assert manager.redis_pipeline_size > 0

    def test_node_batch_sizes(self):
        """Test node batch size properties."""
        manager = BatchSizeManager(environment="test")

        assert manager.orchestrator_batch_size > 0
        assert manager.reducer_batch_size > 0
        assert manager.registry_batch_size > 0

    def test_performance_batch_sizes(self):
        """Test performance batch size properties."""
        manager = BatchSizeManager(environment="test")

        assert manager.performance_task_batch_size > 0
        assert manager.processing_buffer_size > 0

    def test_cleanup_batch_sizes(self):
        """Test cleanup batch size properties."""
        manager = BatchSizeManager(environment="test")

        assert manager.cleanup_batch_size > 0
        assert manager.retention_cleanup_batch_size > 0

    def test_file_processing_batch_sizes(self):
        """Test file processing batch size properties."""
        manager = BatchSizeManager(environment="test")

        assert manager.file_processing_batch_size > 0
        assert manager.metadata_extraction_batch_size > 0

    def test_batch_size_summary(self):
        """Test batch size summary generation."""
        manager = BatchSizeManager(environment="test")
        summary = manager.get_summary()

        assert isinstance(summary, dict)
        assert "environment" in summary
        assert "database_operations" in summary
        assert "kafka_operations" in summary
        assert "redis_operations" in summary
        assert "node_operations" in summary
        assert "performance" in summary
        assert "cleanup" in summary
        assert "file_operations" in summary

    def test_environment_specific_sizes(self):
        """Test that different environments have different batch sizes."""
        dev_manager = BatchSizeManager(environment="development")
        prod_manager = BatchSizeManager(environment="production")

        # Production should have larger batch sizes than development
        assert prod_manager.database_batch_size >= dev_manager.database_batch_size
        assert (
            prod_manager.orchestrator_batch_size >= dev_manager.orchestrator_batch_size
        )
        assert prod_manager.reducer_batch_size >= dev_manager.reducer_batch_size


class TestGlobalBatchManager:
    """Test global batch manager functions."""

    def test_get_batch_manager_default(self):
        """Test get_batch_manager with default environment."""
        manager = get_batch_manager()
        assert manager.environment == "development"
        assert isinstance(manager, BatchSizeManager)

    def test_get_batch_manager_with_environment(self):
        """Test get_batch_manager with specific environment."""
        manager = get_batch_manager(environment="test")
        assert manager.environment == "test"
        assert isinstance(manager, BatchSizeManager)

    def test_get_batch_manager_caching(self):
        """Test that get_batch_manager returns cached instance."""
        manager1 = get_batch_manager(environment="staging")
        manager2 = get_batch_manager(environment="staging")

        # Should return the same instance for same environment
        assert manager1 is manager2

    def test_convenience_functions(self):
        """Test convenience functions for getting batch sizes."""
        # Test that all convenience functions work
        assert get_database_batch_size("test") > 0
        assert get_kafka_batch_size("test") > 0
        assert get_redis_batch_size("test") > 0
        assert get_orchestrator_batch_size("test") > 0
        assert get_reducer_batch_size("test") > 0
        assert get_cleanup_batch_size("test") > 0
        assert get_file_processing_batch_size("test") > 0

    def test_convenience_functions_default_environment(self):
        """Test convenience functions with default environment."""
        # Test that all convenience functions work with default environment
        assert get_database_batch_size() > 0
        assert get_kafka_batch_size() > 0
        assert get_redis_batch_size() > 0
        assert get_orchestrator_batch_size() > 0
        assert get_reducer_batch_size() > 0
        assert get_cleanup_batch_size() > 0
        assert get_file_processing_batch_size() > 0


class TestBatchSizeIntegration:
    """Test batch sizes integration with other components."""

    def test_batch_sizes_in_reducer_context(self):
        """Test batch sizes work in reducer context."""
        manager = get_batch_manager(environment="production")

        # These are the batch sizes typically used by reducer
        assert manager.reducer_batch_size >= 50
        assert manager.database_batch_size >= 50
        assert manager.processing_buffer_size >= 1000

    def test_batch_sizes_in_orchestrator_context(self):
        """Test batch sizes work in orchestrator context."""
        manager = get_batch_manager(environment="production")

        # These are the batch sizes typically used by orchestrator
        assert manager.orchestrator_batch_size >= 25
        assert manager.kafka_producer_batch_size >= 8192
        assert manager.processing_buffer_size >= 1000
