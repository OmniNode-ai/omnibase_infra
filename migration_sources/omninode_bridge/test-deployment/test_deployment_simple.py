#!/usr/bin/env python3
"""
Simplified End-to-End Deployment Test

Tests deployment workflow without complex container configuration.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from omnibase_core.enums import EnumContainerType
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.deployment_sender_effect.v1_0_0.models import (
    ModelContainerPackageInput,
    ModelPackageTransferInput,
)
from omninode_bridge.nodes.deployment_sender_effect.v1_0_0.node import (
    NodeDeploymentSenderEffect,
)


async def test_deployment():
    """Test deployment workflow."""
    print("=" * 80)
    print("ONEX v2.0 Deployment Test - Simplified")
    print("=" * 80)

    correlation_id = uuid4()
    print(f"\n📋 Correlation ID: {correlation_id}")

    # Initialize sender node with minimal config
    print("\n📦 Step 1: Initialize Sender Node")
    container = ModelContainer(
        container_type=EnumContainerType.EFFECT,
        value={"package_dir": "/tmp/deployment_packages"},
    )
    sender = NodeDeploymentSenderEffect(container)
    print("   ✅ Sender node initialized")

    # Package the test container
    print("\n🔨 Step 2: Package Test Container")
    package_input = ModelContainerPackageInput(
        container_name="omninode-test-deployment",
        image_tag="v1.0.0",
        build_context=str(Path(__file__).parent),
        dockerfile_path="Dockerfile",
        compression="gzip",
        correlation_id=correlation_id,
    )

    print("   Building and packaging omninode-test-deployment:v1.0.0...")
    result = await sender.package_container(package_input)

    if not result.success:
        print(f"   ❌ Failed: {result.error_message}")
        return False

    print("   ✅ Package created")
    print(f"      Package ID: {result.package_id}")
    print(f"      Package Path: {result.package_path}")
    print(f"      Size: {result.package_size_mb:.2f} MB")
    print(f"      Checksum: {result.package_checksum[:32]}...")
    print(f"      Build Time: {result.build_duration_ms}ms")
    print(f"      Compression: {result.compression_ratio:.1%}")

    # Transfer to remote receiver
    print("\n🚀 Step 3: Transfer to Remote Receiver (192.168.86.200:8001)")
    transfer_input = ModelPackageTransferInput(
        package_id=result.package_id,
        package_path=result.package_path,
        package_checksum=result.package_checksum,
        remote_receiver_url="http://192.168.86.200:8001/deployment/receive",
        container_name="omninode-test-deployment",
        image_tag="v1.0.0",
        verify_checksum=True,
        correlation_id=correlation_id,
    )

    print("   Transferring package...")
    transfer_result = await sender.transfer_package(transfer_input)

    if not transfer_result.success:
        print(f"   ❌ Transfer failed: {transfer_result.error_message}")
        await sender.cleanup()
        return False

    print("   ✅ Transfer completed")
    print(f"      Duration: {transfer_result.transfer_duration_ms}ms")
    print(f"      Throughput: {transfer_result.transfer_throughput_mbps:.2f} MB/s")
    print(f"      Remote Deployment ID: {transfer_result.remote_deployment_id}")
    print(f"      Checksum Verified: {transfer_result.checksum_verified}")

    # Show metrics
    print("\n📊 Sender Node Metrics")
    metrics = sender._metrics
    print(f"   Builds: {metrics['successful_builds']}/{metrics['total_builds']}")
    print(
        f"   Transfers: {metrics['successful_transfers']}/{metrics['total_transfers']}"
    )
    print(f"   Events Published: {metrics['total_events_published']}")

    await sender.cleanup()

    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\n📝 Summary:")
    print(f"   Package Size: {result.package_size_mb:.2f} MB")
    print(
        f"   Total Time: {result.build_duration_ms + transfer_result.transfer_duration_ms}ms"
    )
    print(f"   Remote Deployment ID: {transfer_result.remote_deployment_id}")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_deployment())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
