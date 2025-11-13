"""Backward compatibility tests for Kafka client partition stability fix."""

import pytest

from omninode_bridge.services.kafka_client import KafkaClient


class TestKafkaClientBackwardCompatibility:
    """Verify that partition stability fix maintains backward compatibility."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Verify KafkaClient initializes correctly with no configuration changes."""
        # This should work exactly as before
        client = KafkaClient()

        assert client is not None
        assert client.partitioning_strategy == "balanced"
        assert client.partition_cache == {}
        assert client.partition_load_tracker == {}

    @pytest.mark.asyncio
    async def test_partitioning_strategies_available(self):
        """Verify all partitioning strategies are still available."""
        strategies = ["balanced", "hash", "round_robin", "default"]

        for strategy in strategies:
            client = KafkaClient()
            client.partitioning_strategy = strategy

            # Should not raise any errors
            assert client.partitioning_strategy == strategy

    @pytest.mark.asyncio
    async def test_balanced_partition_with_none_key(self):
        """Verify balanced partitioning still works for keyless messages."""
        client = KafkaClient()
        topic = "test-topic"
        partition_count = 10

        # Set up partition cache
        client.partition_cache[topic] = partition_count
        client.partition_load_tracker[topic] = [0] * partition_count

        # Get partition for keyless message (should use load balancing)
        partition = await client._get_balanced_partition(topic, None, partition_count)

        # Should return a valid partition
        assert 0 <= partition < partition_count

    @pytest.mark.asyncio
    async def test_hash_partition_unchanged(self):
        """Verify hash partitioning strategy is unchanged (already uses SHA-256)."""
        client = KafkaClient()
        test_key = "test-key"
        topic = "test-topic"
        partition_count = 10

        # Hash partition should already be deterministic (uses SHA-256)
        partition1 = await client._get_hash_partition(topic, test_key, partition_count)
        partition2 = await client._get_hash_partition(topic, test_key, partition_count)

        assert partition1 == partition2
        assert 0 <= partition1 < partition_count

    @pytest.mark.asyncio
    async def test_round_robin_partition_unchanged(self):
        """Verify round-robin partitioning strategy is unchanged."""
        client = KafkaClient()
        topic = "test-topic"
        partition_count = 5

        # Round robin should cycle through partitions
        partitions = []
        for _ in range(partition_count):
            partition = await client._get_round_robin_partition(topic, partition_count)
            partitions.append(partition)

        # Should cycle through all partitions
        assert len(set(partitions)) == partition_count

    @pytest.mark.asyncio
    async def test_no_api_changes(self):
        """Verify public API has not changed."""
        client = KafkaClient()

        # All these methods should still exist
        assert hasattr(client, "connect")
        assert hasattr(client, "disconnect")
        assert hasattr(client, "publish_event")
        assert hasattr(client, "publish_raw_event")
        assert hasattr(client, "publish_with_envelope")
        assert hasattr(client, "_get_balanced_partition")
        assert hasattr(client, "_get_hash_partition")
        assert hasattr(client, "_get_round_robin_partition")
        assert hasattr(client, "_get_optimal_partition")

    @pytest.mark.asyncio
    async def test_hashlib_import_exists(self):
        """Verify hashlib is imported (required for SHA-256)."""
        import omninode_bridge.services.kafka_client as kafka_module

        # hashlib should be imported in the module
        assert hasattr(kafka_module, "hashlib")
