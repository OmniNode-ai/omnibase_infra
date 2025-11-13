#!/usr/bin/env python3
"""
End-to-End Deployment Workflow Test

Tests the complete ONEX v2.0 deployment system:
1. Package container using NodeDeploymentSenderEffect
2. Transfer package to remote receiver (192.168.86.200:8001)
3. Verify container deployment and health
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
    ModelKafkaPublishInput,
    ModelPackageTransferInput,
)
from omninode_bridge.nodes.deployment_sender_effect.v1_0_0.node import (
    NodeDeploymentSenderEffect,
)


async def test_deployment_workflow():
    """Test complete deployment workflow."""
    print("=" * 80)
    print("ONEX v2.0 Deployment Workflow Test")
    print("=" * 80)

    correlation_id = uuid4()
    print(f"\n📋 Correlation ID: {correlation_id}")

    # Configuration
    remote_receiver_url = "http://192.168.86.200:8001/deployment/receive"
    container_name = "omninode-test-deployment"
    image_tag = "v1.0.0"
    build_context = str(Path(__file__).parent)

    print("\n🎯 Target Configuration:")
    print(f"   Remote Receiver: {remote_receiver_url}")
    print(f"   Container: {container_name}:{image_tag}")
    print(f"   Build Context: {build_context}")

    # Initialize sender node
    print("\n📦 Step 1: Initialize Deployment Sender Node")
    container_config = ModelContainer(
        value={
            "package_dir": "/tmp/deployment_packages",
        }
    )
    sender_node = NodeDeploymentSenderEffect(container_config)
    print("   ✅ Sender node initialized")

    # Step 1: Package Container
    print("\n📦 Step 2: Package Docker Container")
    print(f"   Building and packaging {container_name}:{image_tag}...")

    package_input = ModelContainerPackageInput(
        container_name=container_name,
        image_tag=image_tag,
        build_context=build_context,
        dockerfile_path="Dockerfile",
        compression="gzip",
        correlation_id=correlation_id,
    )

    package_result = await sender_node.package_container(package_input)

    if not package_result.success:
        print(f"   ❌ Packaging failed: {package_result.error_message}")
        return False

    print("   ✅ Package created successfully")
    print(f"      Package ID: {package_result.package_id}")
    print(f"      Image ID: {package_result.image_id}")
    print(f"      Package Path: {package_result.package_path}")
    print(f"      Package Size: {package_result.package_size_mb:.2f} MB")
    print(f"      Checksum: {package_result.package_checksum[:16]}...")
    print(f"      Build Duration: {package_result.build_duration_ms}ms")
    print(f"      Compression Ratio: {package_result.compression_ratio:.2%}")

    # Step 2: Publish BUILD_COMPLETED event
    print("\n📡 Step 3: Publish BUILD_COMPLETED Event")
    event_input = ModelKafkaPublishInput(
        event_type="BUILD_COMPLETED",
        correlation_id=correlation_id,
        metadata={
            "package_id": str(package_result.package_id),
            "image_id": package_result.image_id,
            "container_name": container_name,
            "image_tag": image_tag,
        },
    )

    event_result = await sender_node.publish_transfer_event(event_input)
    if event_result.success:
        print(f"   ✅ Event published to topic: {event_result.topic}")
        print(f"      Publish Duration: {event_result.publish_duration_ms}ms")
    else:
        print(f"   ⚠️ Event publishing failed: {event_result.error_message}")

    # Step 3: Transfer Package to Remote Receiver
    print("\n🚀 Step 4: Transfer Package to Remote Receiver")
    print(f"   Transferring to {remote_receiver_url}...")

    transfer_input = ModelPackageTransferInput(
        package_id=package_result.package_id,
        package_path=package_result.package_path,
        package_checksum=package_result.package_checksum,
        remote_receiver_url=remote_receiver_url,
        container_name=container_name,
        image_tag=image_tag,
        verify_checksum=True,
        correlation_id=correlation_id,
    )

    transfer_result = await sender_node.transfer_package(transfer_input)

    if not transfer_result.success:
        print(f"   ❌ Transfer failed: {transfer_result.error_message}")
        return False

    print("   ✅ Transfer completed successfully")
    print(f"      Remote Deployment ID: {transfer_result.remote_deployment_id}")
    print(f"      Transfer Duration: {transfer_result.transfer_duration_ms}ms")
    print(
        f"      Bytes Transferred: {transfer_result.bytes_transferred / (1024 * 1024):.2f} MB"
    )
    print(f"      Throughput: {transfer_result.transfer_throughput_mbps:.2f} MB/s")
    print(f"      Checksum Verified: {transfer_result.checksum_verified}")

    # Step 4: Publish TRANSFER_COMPLETED event
    print("\n📡 Step 5: Publish TRANSFER_COMPLETED Event")
    transfer_event_input = ModelKafkaPublishInput(
        event_type="TRANSFER_COMPLETED",
        correlation_id=correlation_id,
        metadata={
            "package_id": str(package_result.package_id),
            "remote_deployment_id": transfer_result.remote_deployment_id,
            "transfer_duration_ms": transfer_result.transfer_duration_ms,
            "throughput_mbps": transfer_result.transfer_throughput_mbps,
        },
    )

    transfer_event_result = await sender_node.publish_transfer_event(
        transfer_event_input
    )
    if transfer_event_result.success:
        print(f"   ✅ Event published to topic: {transfer_event_result.topic}")
    else:
        print(f"   ⚠️ Event publishing failed: {transfer_event_result.error_message}")

    # Step 5: Get Metrics
    print("\n📊 Step 6: Deployment Sender Metrics")
    metrics = sender_node._metrics
    print(f"   Total Builds: {metrics['total_builds']}")
    print(f"   Successful Builds: {metrics['successful_builds']}")
    print(f"   Failed Builds: {metrics['failed_builds']}")
    print(f"   Total Transfers: {metrics['total_transfers']}")
    print(f"   Successful Transfers: {metrics['successful_transfers']}")
    print(f"   Failed Transfers: {metrics['failed_transfers']}")
    print(f"   Total Events Published: {metrics['total_events_published']}")

    # Cleanup
    await sender_node.cleanup()

    print("\n" + "=" * 80)
    print("🎉 DEPLOYMENT WORKFLOW TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\n📝 Summary:")
    print(f"   Container: {container_name}:{image_tag}")
    print(f"   Package Size: {package_result.package_size_mb:.2f} MB")
    print(
        f"   Total Time: {package_result.build_duration_ms + transfer_result.transfer_duration_ms}ms"
    )
    print(f"   Remote Deployment ID: {transfer_result.remote_deployment_id}")
    print(
        "\n✅ Next Step: Verify container deployment on remote system (192.168.86.200)"
    )

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_deployment_workflow())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
