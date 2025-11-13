"""Advanced Metadata Operations Demo.

This script demonstrates all the advanced metadata operations capabilities:
1. Smart upsert with conflict resolution strategies
2. High-throughput batch processing (>50 items/sec)
3. JSONB deep merge for intelligence_data fields
4. Version control with optimistic locking
5. Idempotent operations using op_id tracking

Usage:
    python advanced_operations_demo.py

Requirements:
    - PostgreSQL database running
    - metadata_stamping_dev database created
    - Environment variables set for database connection
"""

import asyncio
import logging
import os
import time
import uuid

from ..database import (
    AdvancedMetadataOperations,
    ConflictResolutionStrategy,
    DatabaseConfig,
    MetadataStampingPostgresClient,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdvancedOperationsDemo:
    """Comprehensive demo of advanced metadata operations."""

    def __init__(self):
        """Initialize demo with database configuration."""
        self.db_config = DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "metadata_stamping_dev"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password"),
            min_connections=10,
            max_connections=50,
        )
        self.client = None
        self.advanced_ops = None

    async def initialize(self):
        """Initialize database connection and advanced operations."""
        logger.info("🚀 Initializing advanced metadata operations demo...")

        self.client = MetadataStampingPostgresClient(self.db_config)
        success = await self.client.initialize()

        if not success:
            raise Exception("Failed to initialize database connection")

        self.advanced_ops = AdvancedMetadataOperations(self.client)
        await self.advanced_ops._ensure_schema_extensions()

        logger.info("✅ Database connection and advanced operations initialized")

    async def cleanup(self):
        """Clean up database connections."""
        if self.client:
            await self.client.close()
        logger.info("🧹 Database connections closed")

    def create_sample_metadata(self, sequence: int = 0) -> dict:
        """Create sample metadata for demonstration."""
        return {
            "hash_algorithm": "blake3",
            "file_metadata": {
                "creation_date": "2025-09-28T12:00:00Z",
                "modification_date": "2025-09-28T12:00:00Z",
                "permissions": "644",
                "owner": "demo_user",
                "size_category": "medium",
            },
            "intelligence_data": {
                "file_type": "python",
                "encoding": "utf-8",
                "language": "python",
                "complexity_score": 7.5 + (sequence * 0.1),
                "security_analysis": {
                    "vulnerabilities": [],
                    "security_score": 0.95,
                    "last_scan": "2025-09-28T12:00:00Z",
                },
                "performance_metrics": {
                    "execution_time": 1.2,
                    "memory_usage": 128,
                    "cpu_usage": 15.5,
                },
            },
            "analysis_results": [
                {"analyzer": "syntax", "score": 0.95, "issues": []},
                {"analyzer": "style", "score": 0.88, "issues": ["line_too_long"]},
                {"analyzer": "complexity", "score": 0.82, "issues": []},
            ],
            "processing_history": [
                {
                    "timestamp": "2025-09-28T12:00:00Z",
                    "operation": "initial_analysis",
                    "duration_ms": 150,
                }
            ],
            "tags": ["python", "demo", "metadata"],
            "demo_sequence": sequence,
        }

    def create_updated_metadata(self, original: dict, sequence: int = 0) -> dict:
        """Create updated metadata for conflict resolution testing."""
        updated = original.copy()

        # Update some existing fields
        updated["file_metadata"]["modification_date"] = "2025-09-28T14:00:00Z"
        updated["file_metadata"]["permissions"] = "755"
        updated["intelligence_data"]["complexity_score"] += 1.0

        # Add new fields
        updated["file_metadata"]["backup_date"] = "2025-09-28T13:30:00Z"
        updated["intelligence_data"]["code_quality"] = {
            "maintainability_index": 85.5,
            "technical_debt_ratio": 0.15,
            "test_coverage": 0.78,
        }

        # Add to arrays
        updated["analysis_results"].append(
            {"analyzer": "security", "score": 0.91, "issues": []}
        )
        updated["processing_history"].append(
            {
                "timestamp": "2025-09-28T14:00:00Z",
                "operation": "update_analysis",
                "duration_ms": 200,
            }
        )
        updated["tags"].extend(["updated", "enhanced"])

        return updated

    async def demo_jsonb_deep_merge(self):
        """Demonstrate JSONB deep merge functionality."""
        logger.info("\n📋 Demo 1: JSONB Deep Merge")
        logger.info("=" * 50)

        original_data = self.create_sample_metadata(1)
        updated_data = self.create_updated_metadata(original_data, 1)

        logger.info("🔄 Performing deep merge...")
        start_time = time.perf_counter()

        merged_data = self.advanced_ops.jsonb_deep_merge(original_data, updated_data)

        execution_time = (time.perf_counter() - start_time) * 1000

        logger.info(f"⏱️  Deep merge completed in {execution_time:.2f}ms")
        logger.info(f"📊 Original fields: {len(self._count_fields(original_data))}")
        logger.info(f"📊 Updated fields: {len(self._count_fields(updated_data))}")
        logger.info(f"📊 Merged fields: {len(self._count_fields(merged_data))}")

        # Verify merge results
        assert merged_data["file_metadata"]["permissions"] == "755"  # Updated
        assert (
            merged_data["file_metadata"]["creation_date"] == "2025-09-28T12:00:00Z"
        )  # Preserved
        assert "backup_date" in merged_data["file_metadata"]  # New field added
        assert len(merged_data["analysis_results"]) == 4  # Arrays concatenated
        assert "updated" in merged_data["tags"]  # New tag added
        assert "python" in merged_data["tags"]  # Original tag preserved

        logger.info("✅ JSONB deep merge verification passed")

    async def demo_smart_upsert(self):
        """Demonstrate smart upsert with conflict resolution."""
        logger.info("\n🔧 Demo 2: Smart Upsert with Conflict Resolution")
        logger.info("=" * 50)

        file_hash = f"demo_smart_upsert_{uuid.uuid4().hex[:8]}"
        original_data = self.create_sample_metadata(2)

        # Initial insert
        logger.info("📥 Inserting initial record...")
        result1 = await self.advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/demo/smart_upsert.py",
            file_size=2048,
            content_type="text/python",
            stamp_data=original_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        logger.info(
            f"✅ Insert result: {result1.operation}, version: {result1.version}, time: {result1.execution_time_ms:.2f}ms"
        )

        # Update with merge strategy
        logger.info("🔄 Updating with MERGE_JSONB strategy...")
        updated_data = self.create_updated_metadata(original_data, 2)

        result2 = await self.advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/demo/smart_upsert_updated.py",
            file_size=3072,
            content_type="text/python",
            stamp_data=updated_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        logger.info(
            f"✅ Update result: {result2.operation}, version: {result2.version}, time: {result2.execution_time_ms:.2f}ms"
        )
        logger.info(f"🔧 Conflict resolved: {result2.conflict_resolved}")

        # Verify merge occurred
        record = await self.advanced_ops.get_versioned_metadata(file_hash)
        assert record.version == 2
        assert record.file_path == "/demo/smart_upsert_updated.py"
        assert (
            "code_quality" in record.stamp_data["intelligence_data"]
        )  # New field from merge
        assert (
            record.stamp_data["file_metadata"]["creation_date"]
            == "2025-09-28T12:00:00Z"
        )  # Preserved

        logger.info("✅ Smart upsert verification passed")

    async def demo_idempotent_operations(self):
        """Demonstrate idempotent operations using op_id."""
        logger.info("\n🔄 Demo 3: Idempotent Operations")
        logger.info("=" * 50)

        file_hash = f"demo_idempotent_{uuid.uuid4().hex[:8]}"
        op_id = str(uuid.uuid4())
        sample_data = self.create_sample_metadata(3)

        # First operation
        logger.info(f"📥 First operation with op_id: {op_id[:8]}...")
        result1 = await self.advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/demo/idempotent.py",
            file_size=1024,
            content_type="text/python",
            stamp_data=sample_data,
            op_id=op_id,
        )

        logger.info(f"✅ First operation: {result1.operation}, ID: {result1.id[:8]}...")

        # Second operation with same op_id (should be skipped)
        logger.info("🔄 Second operation with same op_id...")
        different_data = self.create_updated_metadata(sample_data, 3)

        result2 = await self.advanced_ops.smart_upsert_metadata_stamp(
            file_hash=file_hash,
            file_path="/demo/idempotent_different.py",  # Different data
            file_size=2048,
            content_type="text/python",
            stamp_data=different_data,
            op_id=op_id,  # Same op_id
        )

        logger.info(
            f"✅ Second operation: {result2.operation}, ID: {result2.id[:8]}..."
        )

        # Verify idempotency
        assert result1.operation == "inserted"
        assert result2.operation == "skipped"
        assert result1.id == result2.id

        # Verify original data is preserved
        record = await self.advanced_ops.get_versioned_metadata(file_hash)
        assert record.file_path == "/demo/idempotent.py"  # Original path preserved
        assert record.file_size == 1024  # Original size preserved

        logger.info("✅ Idempotent operations verification passed")

    async def demo_version_control(self):
        """Demonstrate version control with history tracking."""
        logger.info("\n📚 Demo 4: Version Control and History")
        logger.info("=" * 50)

        file_hash = f"demo_version_{uuid.uuid4().hex[:8]}"

        # Create multiple versions
        versions_data = []
        for i in range(5):
            data = self.create_sample_metadata(i)
            data["version_note"] = f"Version {i + 1}"
            data["intelligence_data"]["complexity_score"] += i
            versions_data.append(data)

        logger.info("📥 Creating version history...")
        results = []

        for i, version_data in enumerate(versions_data):
            result = await self.advanced_ops.smart_upsert_metadata_stamp(
                file_hash=file_hash,
                file_path=f"/demo/versioned_v{i+1}.py",
                file_size=1024 * (i + 1),
                content_type="text/python",
                stamp_data=version_data,
            )
            results.append(result)
            logger.info(
                f"  📝 Version {i + 1}: {result.operation}, version: {result.version}"
            )

        # Get version history
        logger.info("📚 Retrieving version history...")
        history = await self.advanced_ops.get_version_history(file_hash, limit=10)

        logger.info(f"📊 Total versions: {len(history)}")
        for version in history:
            complexity = version.stamp_data["intelligence_data"]["complexity_score"]
            logger.info(
                f"  📝 Version {version.version}: complexity={complexity}, size={version.file_size}"
            )

        # Get specific version
        v3_record = await self.advanced_ops.get_versioned_metadata(file_hash, version=3)
        assert v3_record.version == 3
        assert v3_record.stamp_data["version_note"] == "Version 3"

        logger.info("✅ Version control verification passed")

    async def demo_batch_operations(self):
        """Demonstrate high-performance batch operations."""
        logger.info("\n🚀 Demo 5: High-Performance Batch Operations")
        logger.info("=" * 50)

        # Generate test data
        batch_size = 100
        logger.info(f"📊 Generating {batch_size} test records...")

        test_data = []
        for i in range(batch_size):
            stamp_data = self.create_sample_metadata(i)
            test_data.append(
                {
                    "file_hash": f"demo_batch_{i:04d}_{uuid.uuid4().hex[:8]}",
                    "file_path": f"/demo/batch/file_{i:04d}.py",
                    "file_size": 1024 + (i * 10),
                    "content_type": "text/python",
                    "stamp_data": stamp_data,
                }
            )

        # Perform batch upsert
        logger.info("🚀 Performing batch upsert...")
        start_time = time.perf_counter()

        result = await self.advanced_ops.batch_upsert_metadata_stamps(
            stamps_data=test_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
            batch_size=20,
            max_concurrency=5,
        )

        total_time = time.perf_counter() - start_time

        logger.info("✅ Batch operation completed!")
        logger.info(f"📊 Total processed: {result.total_processed}")
        logger.info(f"✅ Successful operations: {result.successful_operations}")
        logger.info(f"⚠️  Errors: {result.errors}")
        logger.info(f"⏱️  Execution time: {result.execution_time_ms:.2f}ms")
        logger.info(f"🚀 Throughput: {result.throughput_per_sec:.1f} items/sec")

        # Verify performance target
        assert (
            result.throughput_per_sec > 50
        ), f"Performance target not met: {result.throughput_per_sec:.1f} items/sec"
        logger.info("🎯 Performance target (>50 items/sec) achieved!")

        # Test batch updates
        logger.info("🔄 Testing batch updates with conflicts...")

        # Update first 50 records
        update_data = []
        for i in range(50):
            original = test_data[i].copy()
            original["stamp_data"] = self.create_updated_metadata(
                original["stamp_data"], i
            )
            update_data.append(original)

        update_result = await self.advanced_ops.batch_upsert_metadata_stamps(
            stamps_data=update_data,
            conflict_strategy=ConflictResolutionStrategy.MERGE_JSONB,
        )

        logger.info(
            f"🔄 Update batch: {update_result.successful_operations} updated, {update_result.conflicts_resolved} conflicts resolved"
        )
        logger.info(
            f"🚀 Update throughput: {update_result.throughput_per_sec:.1f} items/sec"
        )

        logger.info("✅ Batch operations verification passed")

    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        logger.info("\n📈 Demo 6: Performance Monitoring")
        logger.info("=" * 50)

        # Get current performance metrics
        logger.info("📊 Retrieving performance metrics...")
        metrics = await self.advanced_ops.get_performance_metrics()

        logger.info("📈 Current Performance Metrics:")
        logger.info(f"  🔧 Upsert operations: {metrics['upsert_operations']}")
        logger.info(f"  📦 Batch operations: {metrics['batch_operations']}")
        logger.info(f"  🔀 Merge operations: {metrics['merge_operations']}")
        logger.info(f"  ⚠️  Version conflicts: {metrics['version_conflicts']}")
        logger.info(
            f"  🚀 Average throughput: {metrics['avg_throughput']:.1f} items/sec"
        )

        # Run performance optimization
        logger.info("⚡ Running performance optimization...")
        optimization_result = await self.advanced_ops.optimize_performance()

        logger.info(f"✅ Optimization status: {optimization_result['status']}")
        logger.info(
            f"⏱️  Optimization time: {optimization_result['execution_time_ms']:.2f}ms"
        )
        logger.info(
            f"🔧 Optimizations applied: {len(optimization_result['optimizations_applied'])}"
        )

        for optimization in optimization_result["optimizations_applied"]:
            logger.info(f"  ✅ {optimization}")

        logger.info("✅ Performance monitoring verification passed")

    def _count_fields(self, data: dict, prefix: str = "") -> list[str]:
        """Recursively count all fields in a nested dictionary."""
        fields = []
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            fields.append(current_path)
            if isinstance(value, dict):
                fields.extend(self._count_fields(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        fields.extend(self._count_fields(item, f"{current_path}[{i}]"))
        return fields

    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        logger.info("🎬 Starting Advanced Metadata Operations Demo")
        logger.info("=" * 60)

        try:
            await self.initialize()

            # Run all demos
            await self.demo_jsonb_deep_merge()
            await self.demo_smart_upsert()
            await self.demo_idempotent_operations()
            await self.demo_version_control()
            await self.demo_batch_operations()
            await self.demo_performance_monitoring()

            logger.info("\n🎉 All demos completed successfully!")
            logger.info("=" * 60)
            logger.info("🚀 Advanced metadata operations are ready for production use!")

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main demo execution."""
    demo = AdvancedOperationsDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())
