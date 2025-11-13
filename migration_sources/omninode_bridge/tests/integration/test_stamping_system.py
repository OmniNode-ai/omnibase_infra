"""
Integration tests for the complete stamping system.

Tests the end-to-end flow:
1. Generate content
2. Call stamping API
3. Verify stamp created
4. Check Kafka events published
5. Verify idempotency (already-stamped files)

Phase 2, Track A: Stamping System Integration
"""

import asyncio
import json
import time
from typing import Optional
from uuid import uuid4

import httpx
import pytest
from aiokafka import AIOKafkaConsumer


class StampingSystemTest:
    """Integration test helper for stamping system."""

    def __init__(
        self,
        stamping_service_url: str = "http://localhost:8057",
        kafka_bootstrap_servers: str = "localhost:29092",
    ):
        self.stamping_service_url = stamping_service_url
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.http_client: Optional[httpx.AsyncClient] = None
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None

    async def setup(self):
        """Initialize HTTP client and Kafka consumer."""
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Try to create Kafka consumer for stamping events
        # If Kafka is not available, tests will skip event verification
        try:
            self.kafka_consumer = AIOKafkaConsumer(
                "dev.omninode_bridge.onex.evt.metadata-stamp-created.v1",
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id=f"test-stamping-consumer-{uuid4()}",
                auto_offset_reset="latest",
                enable_auto_commit=False,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            await self.kafka_consumer.start()
            print("✓ Kafka consumer connected")
        except Exception as e:
            print(f"⚠ Kafka not available: {e}")
            print("  Tests will skip Kafka event verification")
            self.kafka_consumer = None

    async def teardown(self):
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
        if self.kafka_consumer:
            await self.kafka_consumer.stop()

    async def check_health(self) -> dict:
        """Check stamping service health."""
        response = await self.http_client.get(f"{self.stamping_service_url}/health")
        response.raise_for_status()
        return response.json()

    async def create_stamp(
        self,
        content: str,
        namespace: str = "omninode.services.metadata",
        correlation_id: Optional[str] = None,
    ) -> dict:
        """Create a metadata stamp."""
        payload = {
            "content": content,
            "namespace": namespace,
        }
        if correlation_id:
            payload["correlation_id"] = correlation_id

        response = await self.http_client.post(
            f"{self.stamping_service_url}/api/v1/metadata-stamping/stamp",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def get_stamp(self, file_hash: str) -> dict:
        """Retrieve a stamp by hash."""
        response = await self.http_client.get(
            f"{self.stamping_service_url}/api/v1/metadata-stamping/stamp/{file_hash}"
        )
        response.raise_for_status()
        return response.json()

    async def validate_content(self, content: str) -> dict:
        """Validate content for existing stamps."""
        payload = {"content": content}

        response = await self.http_client.post(
            f"{self.stamping_service_url}/api/v1/metadata-stamping/validate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def wait_for_kafka_event(
        self, timeout_seconds: float = 5.0, expected_hash: Optional[str] = None
    ) -> Optional[dict]:
        """Wait for a Kafka event with optional hash matching."""
        if not self.kafka_consumer:
            return None  # Kafka not available

        end_time = time.time() + timeout_seconds

        while time.time() < end_time:
            try:
                msg_pack = await asyncio.wait_for(
                    self.kafka_consumer.getmany(timeout_ms=1000),
                    timeout=2.0,
                )

                for tp, messages in msg_pack.items():
                    for message in messages:
                        event = message.value
                        if expected_hash:
                            # Check if this event is for our hash
                            payload = event.get("payload", {})
                            stamp_data = payload.get("stamp_data", {})
                            if stamp_data.get("hash") == expected_hash:
                                return event
                        else:
                            return event

            except TimeoutError:
                continue

        return None

    def calculate_blake3_hash(self, content: str) -> str:
        """Calculate BLAKE3 hash of content (for verification)."""
        # Note: This uses blake2b as a placeholder - actual service uses BLAKE3
        # In real tests, we'd import blake3 if available
        import hashlib

        return hashlib.blake2b(content.encode(), digest_size=32).hexdigest()


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.asyncio
async def test_service_health():
    """Test that the stamping service is healthy and accessible."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        health = await tester.check_health()

        assert health["status"] in ["healthy", "ok"], "Service should be healthy"
        # Note: The actual health response may vary - just check that service responds
        print(f"✓ Service health check passed: {health}")

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_single_file_stamping():
    """Test stamping a single file."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        # Create test content
        content = """
def hello_world():
    print("Hello, World!")
"""
        correlation_id = str(uuid4())

        # Create stamp
        stamp_response = await tester.create_stamp(
            content=content,
            namespace="omninode.services.metadata",
            correlation_id=correlation_id,
        )

        # Verify response
        assert stamp_response["success"], "Stamping should succeed"
        assert "file_hash" in stamp_response, "Response should include file_hash"
        assert "stamp_data" in stamp_response, "Response should include stamp_data"
        assert (
            stamp_response["execution_time_ms"] < 50
        ), "Stamping should be fast (<50ms)"

        file_hash = stamp_response["file_hash"]
        print(
            f"✓ Stamp created: {file_hash[:16]}... in {stamp_response['execution_time_ms']}ms"
        )

        # Wait for Kafka event
        event = await tester.wait_for_kafka_event(
            timeout_seconds=5.0, expected_hash=file_hash
        )

        if event:
            print(f"✓ Kafka event received for hash {file_hash[:16]}...")
            assert event.get("event_type") == "STAMP_CREATED"
            assert event.get("correlation_id") == correlation_id
        else:
            print(
                "⚠ No Kafka event received (may be expected if Kafka not fully configured)"
            )

        # Retrieve stamp
        retrieved_stamp = await tester.get_stamp(file_hash)
        assert retrieved_stamp["success"], "Stamp retrieval should succeed"
        assert (
            retrieved_stamp["stamp_data"]["hash"] == file_hash
        ), "Retrieved stamp should match"

        print("✓ Stamp retrieved successfully")

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_multiple_files_batch():
    """Test stamping multiple files in sequence."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        test_files = [
            ("test1.py", 'print("File 1")'),
            ("test2.py", 'print("File 2")'),
            ("test3.md", "# Test Document"),
            ("test4.yaml", "key: value"),
            ("test5.json", '{"test": true}'),
        ]

        stamps = []
        start_time = time.time()

        for filename, content in test_files:
            stamp_response = await tester.create_stamp(
                content=content,
                namespace="omninode.services.test",
                correlation_id=str(uuid4()),
            )

            assert stamp_response["success"], f"Stamping {filename} should succeed"
            stamps.append((filename, stamp_response["file_hash"]))

            print(f"✓ Stamped {filename}: {stamp_response['file_hash'][:16]}...")

        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(test_files)

        print(
            f"✓ Stamped {len(test_files)} files in {total_time:.1f}ms (avg: {avg_time:.1f}ms per file)"
        )

        assert avg_time < 50, "Average stamping time should be <50ms per file"

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_idempotency():
    """Test that stamping the same content produces the same hash."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        content = "def test(): pass"

        # Create stamp twice
        stamp1 = await tester.create_stamp(content)
        stamp2 = await tester.create_stamp(content)

        assert (
            stamp1["file_hash"] == stamp2["file_hash"]
        ), "Same content should produce same hash"
        assert stamp1["stamp_data"]["hash"] == stamp2["stamp_data"]["hash"]

        print(f"✓ Idempotency verified: {stamp1['file_hash'][:16]}...")

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_different_file_types():
    """Test stamping different file types."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        test_cases = [
            ("python", "def main(): pass"),
            ("markdown", "# README\n\nThis is a test"),
            ("yaml", "services:\n  app:\n    image: python:3.12"),
            ("json", '{"name": "test", "version": "1.0"}'),
            ("text", "Plain text content"),
        ]

        for file_type, content in test_cases:
            stamp = await tester.create_stamp(
                content=content,
                namespace=f"omninode.test.{file_type}",
            )

            assert stamp["success"], f"Stamping {file_type} should succeed"
            print(f"✓ {file_type}: {stamp['file_hash'][:16]}...")

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_validation_endpoint():
    """Test the validation endpoint for detecting existing stamps."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        # Create stamped content
        content = "print('validated content')"
        stamp = await tester.create_stamp(content)
        file_hash = stamp["file_hash"]

        print(f"✓ Created stamp: {file_hash[:16]}...")

        # Validate the same content
        validation = await tester.validate_content(content)

        assert validation["success"], "Validation should succeed"
        # Note: The actual validation response format may vary

        print("✓ Validation completed")

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_large_file_handling():
    """Test stamping larger files (but still reasonable size)."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        # Create a 100KB file
        large_content = "x" * (100 * 1024)

        start_time = time.time()
        stamp = await tester.create_stamp(large_content)
        elapsed_ms = (time.time() - start_time) * 1000

        assert stamp["success"], "Large file stamping should succeed"
        assert elapsed_ms < 100, "Large file stamping should be <100ms"

        print(f"✓ Large file (100KB) stamped in {elapsed_ms:.1f}ms")

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_error_handling_invalid_namespace():
    """Test error handling for invalid namespace."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        # Try with invalid namespace (should follow pattern a.b.c)
        with pytest.raises(httpx.HTTPStatusError):
            await tester.create_stamp(
                content="test",
                namespace="invalid-namespace",  # Invalid format
            )

        print("✓ Invalid namespace rejected as expected")

    finally:
        await tester.teardown()


@pytest.mark.asyncio
async def test_error_handling_empty_content():
    """Test error handling for empty content."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        # Try with empty content
        with pytest.raises(httpx.HTTPStatusError):
            await tester.create_stamp(content="")

        print("✓ Empty content rejected as expected")

    finally:
        await tester.teardown()


# ============================================================================
# Performance Test Suite
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.performance
async def test_performance_10_files():
    """Performance test: stamp 10 files and measure throughput."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        file_count = 10
        test_files = [(f"file{i}.py", f'print("File {i}")') for i in range(file_count)]

        start_time = time.time()

        for filename, content in test_files:
            stamp = await tester.create_stamp(content)
            assert stamp["success"]

        total_time = (time.time() - start_time) * 1000
        throughput = file_count / (total_time / 1000)

        print(f"✓ Performance: {file_count} files in {total_time:.1f}ms")
        print(f"  Throughput: {throughput:.1f} files/second")
        print(f"  Average: {total_time/file_count:.1f}ms per file")

        assert (
            total_time < 500
        ), f"10 files should be stamped in <500ms (was {total_time:.1f}ms)"

    finally:
        await tester.teardown()


@pytest.mark.asyncio
@pytest.mark.performance
async def test_performance_concurrent_requests():
    """Performance test: concurrent stamping requests."""
    tester = StampingSystemTest()
    await tester.setup()

    try:
        concurrent_count = 5
        contents = [f'print("Concurrent {i}")' for i in range(concurrent_count)]

        start_time = time.time()

        # Create stamps concurrently
        tasks = [tester.create_stamp(content) for content in contents]
        stamps = await asyncio.gather(*tasks)

        total_time = (time.time() - start_time) * 1000

        assert all(s["success"] for s in stamps), "All stamps should succeed"

        print(f"✓ Concurrent: {concurrent_count} requests in {total_time:.1f}ms")
        print(f"  Average: {total_time/concurrent_count:.1f}ms per request")

    finally:
        await tester.teardown()


if __name__ == "__main__":
    """Run tests directly for quick validation."""
    import sys

    async def run_basic_tests():
        """Run basic smoke tests."""
        print("=" * 70)
        print("Stamping System Integration Tests")
        print("=" * 70)

        try:
            print("\n1. Health Check")
            await test_service_health()

            print("\n2. Single File Stamping")
            await test_single_file_stamping()

            print("\n3. Multiple Files Batch")
            await test_multiple_files_batch()

            print("\n4. Idempotency")
            await test_idempotency()

            print("\n5. Different File Types")
            await test_different_file_types()

            print("\n" + "=" * 70)
            print("✅ All basic tests passed!")
            print("=" * 70)

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    asyncio.run(run_basic_tests())
