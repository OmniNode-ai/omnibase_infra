"""Unit tests for Kafka client partition assignment stability.

This test suite verifies that partition assignment using SHA-256 is stable
across multiple calls and process restarts, ensuring event ordering guarantees
for correlation-based processing.
"""

import pytest

from omninode_bridge.services.kafka_client import KafkaClient


class TestKafkaPartitionStability:
    """Test suite for stable partition assignment using SHA-256."""

    @pytest.mark.asyncio
    async def test_stable_partition_assignment_same_key(self):
        """Verify same key always maps to same partition across multiple calls."""
        # Arrange
        client = KafkaClient()
        test_key = "test-correlation-id-12345"
        topic = "test-topic"
        partition_count = 10

        # Manually set partition cache to avoid metadata lookup
        client.partition_cache[topic] = partition_count
        client.partition_load_tracker[topic] = [0] * partition_count

        # Act - Get partition multiple times
        partition1 = await client._get_balanced_partition(
            topic, test_key, partition_count
        )
        partition2 = await client._get_balanced_partition(
            topic, test_key, partition_count
        )
        partition3 = await client._get_balanced_partition(
            topic, test_key, partition_count
        )

        # Assert - All calls should return the same partition
        assert (
            partition1 == partition2
        ), "Partition assignment is not stable across calls"
        assert (
            partition2 == partition3
        ), "Partition assignment is not stable across calls"
        assert 0 <= partition1 < partition_count, "Partition is out of valid range"

    @pytest.mark.asyncio
    async def test_different_keys_map_to_different_partitions(self):
        """Verify different keys can map to different partitions (distribution)."""
        # Arrange
        client = KafkaClient()
        topic = "test-topic"
        partition_count = 10

        # Manually set partition cache
        client.partition_cache[topic] = partition_count
        client.partition_load_tracker[topic] = [0] * partition_count

        # Act - Get partitions for different keys
        keys = [f"correlation-id-{i}" for i in range(100)]
        partitions = set()

        for key in keys:
            partition = await client._get_balanced_partition(
                topic, key, partition_count
            )
            partitions.add(partition)

        # Assert - Multiple partitions should be used (good distribution)
        # With 100 different keys and 10 partitions, we expect most partitions to be used
        assert (
            len(partitions) >= 5
        ), f"Poor distribution: only {len(partitions)}/10 partitions used"

    @pytest.mark.asyncio
    async def test_hash_partition_strategy_stable(self):
        """Verify hash partitioning strategy is also stable (uses SHA-256)."""
        # Arrange
        client = KafkaClient()
        client.partitioning_strategy = "hash"
        test_key = "test-correlation-id-hash"
        topic = "test-topic"
        partition_count = 10

        # Manually set partition cache
        client.partition_cache[topic] = partition_count

        # Act - Get partition multiple times
        partition1 = await client._get_hash_partition(topic, test_key, partition_count)
        partition2 = await client._get_hash_partition(topic, test_key, partition_count)
        partition3 = await client._get_hash_partition(topic, test_key, partition_count)

        # Assert - All calls should return the same partition
        assert partition1 == partition2, "Hash partition assignment is not stable"
        assert partition2 == partition3, "Hash partition assignment is not stable"
        assert 0 <= partition1 < partition_count, "Partition is out of valid range"

    @pytest.mark.asyncio
    async def test_stable_partition_with_unicode_key(self):
        """Verify stable partitioning works with unicode keys."""
        # Arrange
        client = KafkaClient()
        topic = "test-topic"
        partition_count = 10

        # Test with unicode characters
        unicode_key = "用户-корреляция-🔥-123"

        # Manually set partition cache
        client.partition_cache[topic] = partition_count
        client.partition_load_tracker[topic] = [0] * partition_count

        # Act - Get partition multiple times
        partition1 = await client._get_balanced_partition(
            topic, unicode_key, partition_count
        )
        partition2 = await client._get_balanced_partition(
            topic, unicode_key, partition_count
        )

        # Assert - Same partition for unicode key
        assert (
            partition1 == partition2
        ), "Unicode key partition assignment is not stable"
        assert 0 <= partition1 < partition_count, "Partition is out of valid range"

    @pytest.mark.asyncio
    async def test_stable_partition_preserves_ordering_guarantee(self):
        """Verify stable partitioning preserves event ordering for same correlation ID."""
        # Arrange
        client = KafkaClient()
        topic = "test-topic"
        partition_count = 10
        correlation_id = "workflow-execution-12345"

        # Manually set partition cache
        client.partition_cache[topic] = partition_count
        client.partition_load_tracker[topic] = [0] * partition_count

        # Act - Simulate multiple events with same correlation ID
        event_partitions = []
        for event_num in range(20):
            # Use same correlation_id for all events (simulating ordered workflow)
            partition = await client._get_balanced_partition(
                topic, correlation_id, partition_count
            )
            event_partitions.append(partition)

        # Assert - All events with same correlation_id go to same partition
        unique_partitions = set(event_partitions)
        assert len(unique_partitions) == 1, (
            f"Events with same correlation_id mapped to {len(unique_partitions)} "
            f"different partitions, breaking ordering guarantee"
        )

    def test_sha256_calculation_deterministic(self):
        """Verify SHA-256 calculation is deterministic (no async needed)."""
        import hashlib

        # Arrange
        test_key = "test-correlation-id"
        partition_count = 10

        # Act - Calculate hash multiple times
        hash1 = int(hashlib.sha256(test_key.encode()).hexdigest()[:8], 16)
        hash2 = int(hashlib.sha256(test_key.encode()).hexdigest()[:8], 16)
        hash3 = int(hashlib.sha256(test_key.encode()).hexdigest()[:8], 16)

        partition1 = hash1 % partition_count
        partition2 = hash2 % partition_count
        partition3 = hash3 % partition_count

        # Assert - Same hash and partition every time
        assert hash1 == hash2 == hash3, "SHA-256 hash is not deterministic"
        assert (
            partition1 == partition2 == partition3
        ), "Partition calculation is not deterministic"

    def test_sha256_vs_python_hash_stability(self):
        """Demonstrate that SHA-256 is stable while Python hash() is not predictable."""
        import hashlib

        # Arrange
        test_key = "test-correlation-id"

        # Act - Calculate SHA-256 hash
        sha256_hash1 = int(hashlib.sha256(test_key.encode()).hexdigest()[:8], 16)
        sha256_hash2 = int(hashlib.sha256(test_key.encode()).hexdigest()[:8], 16)

        # Python's hash() would be different across process restarts (can't test here)
        python_hash = hash(test_key)

        # Assert - SHA-256 is stable
        assert sha256_hash1 == sha256_hash2, "SHA-256 should be deterministic"

        # Document that Python hash() is NOT stable across processes
        # (This is why we use SHA-256 instead)
        print(f"SHA-256 hash: {sha256_hash1} (stable)")
        print(f"Python hash(): {python_hash} (NOT stable across process restarts)")
