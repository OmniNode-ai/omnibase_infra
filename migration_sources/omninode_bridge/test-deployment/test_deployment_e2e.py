#!/usr/bin/env python3
"""
End-to-End Deployment Test - ONEX v2.0

Tests complete deployment workflow:
1. Package Docker container using deployment_sender_effect
2. Transfer to remote receiver on 192.168.86.200:8001
3. Verify transfer success
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.deployment_sender_effect.v1_0_0.models import (
    ModelContainerPackageInput,
    ModelPackageTransferInput,
)
from omninode_bridge.nodes.deployment_sender_effect.v1_0_0.node import (
    NodeDeploymentSenderEffect,
)


async def main():
    """Execute deployment test."""
    print("=" * 80)
    print("🚀 ONEX v2.0 Deployment System - End-to-End Test")
    print("=" * 80)

    correlation_id = uuid4()
    print(f"\n📋 Correlation ID: {correlation_id}\n")

    # Step 1: Initialize sender node
    print("Step 1: Initialize Deployment Sender Node")
    print("-" * 80)
    container = ModelContainer(
        container_type="deployment_sender",
        value={"package_dir": "/tmp/deployment_packages"},
    )
    sender = NodeDeploymentSenderEffect(container)
    print("✅ Sender node initialized\n")

    # Step 2: Package the test container
    print("Step 2: Package Test Container")
    print("-" * 80)
    print("Building Docker image: omninode-test-deployment:v1.0.0")
    print("Dockerfile location: test-deployment/Dockerfile")

    package_input = ModelContainerPackageInput(
        container_name="omninode-test-deployment",
        image_tag="v1.0.0",
        build_context=str(Path(__file__).parent),
        dockerfile_path="Dockerfile",
        compression="gzip",
        correlation_id=correlation_id,
    )

    package_result = await sender.package_container(package_input)

    if not package_result.success:
        print(f"❌ Packaging failed: {package_result.error_message}")
        await sender.cleanup()
        return False

    print("✅ Package created successfully")
    print(f"   • Package ID: {package_result.package_id}")
    print(f"   • Image ID: {package_result.image_id}")
    print(f"   • Package Path: {package_result.package_path}")
    print(f"   • Package Size: {package_result.package_size_mb:.2f} MB")
    print(f"   • BLAKE3 Checksum: {package_result.package_checksum}")
    print(f"   • Build Duration: {package_result.build_duration_ms}ms")
    print(f"   • Compression Ratio: {package_result.compression_ratio:.1%}\n")

    # Step 3: Transfer to remote receiver
    print("Step 3: Transfer Package to Remote Receiver")
    print("-" * 80)
    print("Target: http://192.168.86.200:8001/deployment/receive")

    transfer_input = ModelPackageTransferInput(
        package_id=package_result.package_id,
        package_path=package_result.package_path,
        package_checksum=package_result.package_checksum,
        remote_receiver_url="http://192.168.86.200:8001/deployment/receive",
        container_name="omninode-test-deployment",
        image_tag="v1.0.0",
        verify_checksum=True,
        correlation_id=correlation_id,
    )

    transfer_result = await sender.transfer_package(transfer_input)

    if not transfer_result.success:
        print(f"❌ Transfer failed: {transfer_result.error_message}")
        await sender.cleanup()
        return False

    print("✅ Transfer completed successfully")
    print(f"   • Remote Deployment ID: {transfer_result.remote_deployment_id}")
    print(f"   • Transfer Duration: {transfer_result.transfer_duration_ms}ms")
    print(
        f"   • Bytes Transferred: {transfer_result.bytes_transferred:,} bytes ({transfer_result.bytes_transferred / (1024 * 1024):.2f} MB)"
    )
    print(
        f"   • Transfer Throughput: {transfer_result.transfer_throughput_mbps:.2f} MB/s"
    )
    print(f"   • Checksum Verified: {transfer_result.checksum_verified}\n")

    # Step 4: Show metrics
    print("Step 4: Deployment Metrics")
    print("-" * 80)
    metrics = sender._metrics
    print(f"   • Total Builds: {metrics['total_builds']}")
    print(f"   • Successful Builds: {metrics['successful_builds']}")
    print(f"   • Failed Builds: {metrics['failed_builds']}")
    print(f"   • Total Transfers: {metrics['total_transfers']}")
    print(f"   • Successful Transfers: {metrics['successful_transfers']}")
    print(f"   • Failed Transfers: {metrics['failed_transfers']}")
    print(f"   • Events Published: {metrics['total_events_published']}\n")

    await sender.cleanup()

    print("=" * 80)
    print("🎉 END-TO-END DEPLOYMENT TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

    total_time_ms = (
        package_result.build_duration_ms + transfer_result.transfer_duration_ms
    )
    print("\n📊 Test Summary:")
    print("   • Container: omninode-test-deployment:v1.0.0")
    print(f"   • Package Size: {package_result.package_size_mb:.2f} MB")
    print(f"   • Total Execution Time: {total_time_ms}ms ({total_time_ms/1000:.1f}s)")
    print(f"   • Build Time: {package_result.build_duration_ms}ms")
    print(f"   • Transfer Time: {transfer_result.transfer_duration_ms}ms")
    print(f"   • Remote Deployment ID: {transfer_result.remote_deployment_id}")

    print("\n📝 Next Steps:")
    print("   1. Verify container on remote system: ssh jonah@192.168.86.200")
    print("   2. Check Docker images: docker images | grep omninode-test-deployment")
    print("   3. Run container: docker run --rm omninode-test-deployment:v1.0.0")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print("\n\n❌ Test failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        print("\n" + "=" * 80)
        print("Stack Trace:")
        print("=" * 80)
        traceback.print_exc()
        sys.exit(1)
