"""Comprehensive tests for advanced metadata operations.

Tests for:
- Smart upsert functionality with conflict resolution
- High-throughput batch processing (>50 items/sec)
- JSONB deep merge capabilities
- Version control with optimistic locking
- Idempotent operations using op_id tracking
"""

import asyncio
import uuid

import pytest

from ..database.client import DatabaseConfig, MetadataStampingPostgresClient
from ..database.operations import AdvancedMetadataOperations, ConflictResolutionStrategy


@pytest.fixture
def sample_stamp_data():
    """Sample stamp data for testing."""
    return {
        "hash_algorithm": "blake3",
        "file_metadata": {
            "creation_date": "2025-09-28T12:00:00Z",
            "modification_date": "2025-09-28T12:00:00Z",
            "permissions": "644",
        },
        "intelligence_data": {
            "file_type": "text",
            "encoding": "utf-8",
            "language": "python",
            "complexity_score": 7.5,
        },
        "analysis_results": [
            {"analyzer": "syntax", "score": 0.95},
            {"analyzer": "style", "score": 0.88},
        ],
    }


@pytest.fixture
def updated_stamp_data():
    """Updated stamp data for conflict testing."""
    return {
        "hash_algorithm": "blake3",
        "file_metadata": {
            "creation_date": "2025-09-28T12:00:00Z",
            "modification_date": "2025-09-28T14:00:00Z",  # Updated
            "permissions": "755",  # Updated
            "owner": "test_user",  # New field
        },
        "intelligence_data": {
            "file_type": "text",
            "encoding": "utf-8",
            "language": "python",
            "complexity_score": 8.2,  # Updated
            "security_score": 0.92,  # New field
            "performance_metrics": {  # New nested object
                "execution_time": 1.5,
                "memory_usage": 128,
            },
        },
        "analysis_results": [
            {"analyzer": "syntax", "score": 0.97},  # Updated
            {"analyzer": "style", "score": 0.88},  # Same
            {"analyzer": "security", "score": 0.91},  # New
        ],
        "processing_history": [  # New array
            {"timestamp": "2025-09-28T14:00:00Z", "operation": "update"}
        ],
    }


class TestJSONBDeepMerge:
    """Test JSONB deep merge functionality."""

    def test_simple_merge(self):
        """Test basic object merging."""
        ops = AdvancedMetadataOperations(None)

        existing = {"a": 1, "b": {"x": 10}}
        new = {"b": {"y": 20}, "c": 3}

        result = ops.jsonb_deep_merge(existing, new)

        assert result == {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}

    def test_array_concatenation(self):
        """Test array concatenation with deduplication."""
        ops = AdvancedMetadataOperations(None)

        existing = {"tags": ["python", "test"], "scores": [1, 2]}
        new = {"tags": ["test", "advanced"], "scores": [2, 3]}

        result = ops.jsonb_deep_merge(existing, new)

        assert "python" in result["tags"]
        assert "test" in result["tags"]
        assert "advanced" in result["tags"]
        assert len([x for x in result["tags"] if x == "test"]) == 1  # No duplicates
        assert set(result["scores"]) == {1, 2, 3}

    def test_nested_object_merge(self):
        """Test deep nested object merging."""
        ops = AdvancedMetadataOperations(None)

        existing = {
            "config": {
                "database": {"host": "localhost", "port": 5432},
                "cache": {"enabled": True},
            }
        }
        new = {"config": {"database": {"timeout": 30}, "logging": {"level": "INFO"}}}

        result = ops.jsonb_deep_merge(existing, new)

        assert result["config"]["database"]["host"] == "localhost"
        assert result["config"]["database"]["port"] == 5432
        assert result["config"]["database"]["timeout"] == 30
        assert result["config"]["cache"]["enabled"] is True
        assert result["config"]["logging"]["level"] == "INFO"

    def test_complex_data_merge(self, sample_stamp_data, updated_stamp_data):
        """Test merging complex stamp data."""
        ops = AdvancedMetadataOperations(None)

        result = ops.jsonb_deep_merge(sample_stamp_data, updated_stamp_data)

        # Check merged fields
        assert result["file_metadata"]["permissions"] == "755"  # Updated
        assert (
            result["file_metadata"]["creation_date"] == "2025-09-28T12:00:00Z"
        )  # Preserved
        assert result["file_metadata"]["owner"] == "test_user"  # New field

        # Check intelligence_data merge
        assert result["intelligence_data"]["complexity_score"] == 8.2  # Updated
        assert result["intelligence_data"]["encoding"] == "utf-8"  # Preserved
        assert result["intelligence_data"]["security_score"] == 0.92  # New field
        assert "performance_metrics" in result["intelligence_data"]  # New nested object

        # Check array concatenation
        assert len(result["analysis_results"]) == 3  # Original 2 + 1 new
        assert result["processing_history"][0]["operation"] == "update"  # New array


@pytest.mark.asyncio
class TestSmartUpsert:
    """Test smart upsert functionality."""

    @pytest.fixture
    async def db_client(self):
        """Database client for testing."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="metadata_stamping_test",
            user="test_user",
            password="test_password",
            min_connections=2,
            max_connections=5,
        )
        client = MetadataStampingPostgresClient(config)
        await client.initialize()
        yield client
        await client.close()

    @pytest.fixture
    async def advanced_ops(self, db_client):
        """Advanced operations instance."""
        ops = AdvancedMetadataOperations(db_client)
        await ops._ensure_schema_extensions()
        return ops

    async def test_insert_new_record(self, advanced_ops, sample_stamp_data):
        """Test inserting a new record."""
        file_hash = f"test_hash_{uuid.uuid4().hex[:8]}"

        result = await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=sample_stamp_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        assert result.operation == "inserted"
        assert result.version == 1
        assert result.execution_time_ms < 5.0  # Performance target
        assert not result.conflict_resolved
        assert result.id is not None

    async def test_update_existing_record_merge(
        self, advanced_ops, sample_stamp_data, updated_stamp_data
    ):
        """Test updating existing record with JSONB merge."""
        file_hash = f"test_hash_{uuid.uuid4().hex[:8]}"

        # Insert initial record
        await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=sample_stamp_data,
        )

        # Update with merge strategy
        result = await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file_updated.py",
            file_size=1536,
            content_type="text/python",
            stamp_data=updated_stamp_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        assert result.operation == "updated"
        assert result.version == 2
        assert result.conflict_resolved

        # Verify merge occurred
        record = await advanced_ops.get_versioned_metadata(file_hash)
        assert record.file_path == "/test/file_updated.py"
        assert record.file_size == 1536
        assert (
            record.stamp_data["intelligence_data"]["complexity_score"] == 8.2
        )  # Updated value
        assert (
            record.stamp_data["intelligence_data"]["encoding"] == "utf-8"
        )  # Preserved value

    async def test_conflict_resolution_strategies(
        self, advanced_ops, sample_stamp_data, updated_stamp_data
    ):
        """Test different conflict resolution strategies."""
        file_hash = f"test_hash_{uuid.uuid4().hex[:8]}"

        # Insert initial record
        await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=sample_stamp_data,
        )

        # Test REPLACE_ALL strategy
        result = await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=updated_stamp_data,
            conflict_strategy=ConflictResolutionStrategy.REPLACE_ALL,
        )

        assert result.operation == "updated"
        assert result.version == 2

        # Verify complete replacement
        record = await advanced_ops.get_versioned_metadata(file_hash)
        assert "owner" in record.stamp_data["file_metadata"]  # From updated data
        assert (
            "creation_date" in record.stamp_data["file_metadata"]
        )  # From updated data
        # Original complexity_score should be replaced
        assert record.stamp_data["intelligence_data"]["complexity_score"] == 8.2

    async def test_idempotent_operations(self, advanced_ops, sample_stamp_data):
        """Test idempotent operations using op_id."""
        file_hash = f"test_hash_{uuid.uuid4().hex[:8]}"
        op_id = str(uuid.uuid4())

        # First operation
        result1 = await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=sample_stamp_data,
            op_id=op_id,
        )

        # Second operation with same op_id should be skipped
        result2 = await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file_modified.py",  # Different data
            file_size=2048,
            content_type="text/python",
            stamp_data=sample_stamp_data,
            op_id=op_id,  # Same op_id
        )

        assert result1.operation == "inserted"
        assert result2.operation == "skipped"
        assert result1.id == result2.id

        # Verify original data is preserved
        record = await advanced_ops.get_versioned_metadata(file_hash)
        assert record.file_path == "/test/file.py"  # Original path
        assert record.file_size == 1024  # Original size

    async def test_version_control(self, advanced_ops, sample_stamp_data):
        """Test version control functionality."""
        file_hash = f"test_hash_{uuid.uuid4().hex[:8]}"

        # Insert initial version
        result1 = await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=sample_stamp_data,
        )

        # Update to create version 2
        updated_data = {**sample_stamp_data, "version_note": "second version"}
        result2 = await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/file.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=updated_data,
        )

        assert result1.version == 1
        assert result2.version == 2

        # Get version history
        history = await advanced_ops.get_version_history(file_hash)
        assert len(history) == 2
        assert history[0].version == 2  # Newest first
        assert history[1].version == 1

        # Get specific version
        v1_record = await advanced_ops.get_versioned_metadata(file_hash, version=1)
        assert v1_record.version == 1
        assert "version_note" not in v1_record.stamp_data


@pytest.mark.asyncio
class TestBatchOperations:
    """Test high-performance batch operations."""

    @pytest.fixture
    async def db_client(self):
        """Database client for testing."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="metadata_stamping_test",
            user="test_user",
            password="test_password",
            min_connections=5,
            max_connections=20,
        )
        client = MetadataStampingPostgresClient(config)
        await client.initialize()
        yield client
        await client.close()

    @pytest.fixture
    async def advanced_ops(self, db_client):
        """Advanced operations instance."""
        ops = AdvancedMetadataOperations(db_client)
        await ops._ensure_schema_extensions()
        return ops

    def generate_test_data(self, count: int) -> list[dict]:
        """Generate test data for batch operations."""
        base_stamp_data = {
            "hash_algorithm": "blake3",
            "file_metadata": {"creation_date": "2025-09-28T12:00:00Z"},
            "intelligence_data": {"file_type": "test"},
        }

        return [
            {
                "file_hash": f"test_batch_{i:06d}_{uuid.uuid4().hex[:8]}",
                "file_path": f"/test/batch/file_{i:06d}.py",
                "file_size": 1024 + i,
                "content_type": "text/python",
                "stamp_data": {
                    **base_stamp_data,
                    "sequence_number": i,
                    "batch_id": "test_batch_001",
                },
            }
            for i in range(count)
        ]

    async def test_batch_insert_performance(self, advanced_ops):
        """Test batch insert performance target: >50 items/sec."""
        test_data = self.generate_test_data(100)

        result = await advanced_ops.batch_upsert_metadata_stamps(
            stamps_data=test_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
            batch_size=20,
            max_concurrency=5,
        )

        assert result.total_processed == 100
        assert result.successful_operations == 100
        assert result.errors == 0
        assert result.throughput_per_sec > 50  # Performance target
        assert result.execution_time_ms < 2000  # Should complete within 2 seconds

    async def test_batch_upsert_with_conflicts(self, advanced_ops):
        """Test batch upsert with conflict resolution."""
        # First batch - initial inserts
        initial_data = self.generate_test_data(50)

        result1 = await advanced_ops.batch_upsert_metadata_stamps(
            stamps_data=initial_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        assert result1.successful_operations == 50

        # Second batch - updates with conflicts
        updated_data = []
        for item in initial_data[:25]:  # Update first 25 items
            updated_item = item.copy()
            updated_item["stamp_data"] = {
                **item["stamp_data"],
                "updated": True,
                "update_timestamp": "2025-09-28T14:00:00Z",
            }
            updated_data.append(updated_item)

        result2 = await advanced_ops.batch_upsert_metadata_stamps(
            stamps_data=updated_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        assert result2.total_processed == 25
        assert result2.conflicts_resolved == 25
        assert result2.successful_operations == 25

    async def test_batch_operation_error_handling(self, advanced_ops):
        """Test batch operation error handling."""
        # Create test data with some invalid entries
        test_data = self.generate_test_data(10)

        # Add invalid entries
        invalid_entries = [
            {
                "file_hash": "",
                "file_path": "/test/invalid1.py",
            },  # Missing required fields
            {"file_hash": "valid_hash", "file_path": None},  # None file_path
            {},  # Empty dict
        ]

        mixed_data = test_data + invalid_entries

        result = await advanced_ops.batch_upsert_metadata_stamps(
            stamps_data=mixed_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        assert result.total_processed == 13  # 10 valid + 3 invalid
        assert result.successful_operations == 10  # Only valid entries succeed
        assert result.errors == 3  # Invalid entries fail
        assert len(result.results) == 13

    async def test_concurrent_batch_operations(self, advanced_ops):
        """Test concurrent batch operations."""
        # Create multiple batches
        batch1 = self.generate_test_data(30)
        batch2 = self.generate_test_data(30)
        batch3 = self.generate_test_data(30)

        # Run batches concurrently
        tasks = [
            advanced_ops.batch_upsert_metadata_stamps(batch1),
            advanced_ops.batch_upsert_metadata_stamps(batch2),
            advanced_ops.batch_upsert_metadata_stamps(batch3),
        ]

        results = await asyncio.gather(*tasks)

        # Verify all batches completed successfully
        for result in results:
            assert result.total_processed == 30
            assert result.successful_operations == 30
            assert result.errors == 0

        # Verify total throughput
        total_items = sum(r.total_processed for r in results)
        total_time = max(r.execution_time_ms for r in results) / 1000
        overall_throughput = total_items / total_time

        assert overall_throughput > 50  # Should still meet performance target


@pytest.mark.asyncio
class TestPerformanceOptimization:
    """Test performance optimization features."""

    @pytest.fixture
    async def db_client(self):
        """Database client for testing."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="metadata_stamping_test",
            user="test_user",
            password="test_password",
        )
        client = MetadataStampingPostgresClient(config)
        await client.initialize()
        yield client
        await client.close()

    @pytest.fixture
    async def advanced_ops(self, db_client):
        """Advanced operations instance."""
        return AdvancedMetadataOperations(db_client)

    async def test_performance_metrics(self, advanced_ops, sample_stamp_data):
        """Test performance metrics collection."""
        # Perform some operations
        file_hash = f"test_metrics_{uuid.uuid4().hex[:8]}"

        await advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/test/metrics.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=sample_stamp_data,
        )

        # Get metrics
        metrics = await advanced_ops.get_performance_metrics()

        assert "upsert_operations" in metrics
        assert "batch_operations" in metrics
        assert "merge_operations" in metrics
        assert "version_conflicts" in metrics
        assert "avg_throughput" in metrics
        assert "timestamp" in metrics
        assert "client_metrics" in metrics

    async def test_performance_optimization(self, advanced_ops):
        """Test performance optimization routines."""
        result = await advanced_ops.optimize_performance()

        assert "optimizations_applied" in result
        assert "execution_time_ms" in result
        assert "status" in result
        assert result["status"] in ["completed", "failed"]


if __name__ == "__main__":
    # Run with: python -m pytest test_advanced_operations.py -v
    pytest.main([__file__, "-v", "--tb=short"])
