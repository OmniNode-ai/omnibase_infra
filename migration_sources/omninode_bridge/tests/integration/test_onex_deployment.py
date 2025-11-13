#!/usr/bin/env python3
"""
Test ONEX deployment sender/receiver nodes using proper ONEX Effect Node interface.

This tests the actual node.py implementations, not the main_standalone.py wrappers.
"""

import asyncio
import json
from typing import Any
from uuid import uuid4

from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.deployment_receiver_effect.v1_0_0.node import (
    NodeDeploymentReceiverEffect,
)
from omninode_bridge.nodes.deployment_sender_effect.v1_0_0.node import (
    NodeDeploymentSenderEffect,
)


def create_onex_container(config: dict[str, Any] | None = None) -> ModelContainer:
    """
    Create a basic ONEX ModelContainer for dependency injection.

    In a real ONEX runtime, this would be provided by the framework.
    For testing, we create a minimal container.
    """
    # ModelContainer requires 'value' and 'container_type'
    container = ModelContainer(
        value=config or {},  # Configuration dict or empty dict
        container_type="test_deployment",  # Container type identifier
        source="test_onex_deployment.py",
        is_validated=True,
    )

    return container


def create_effect_contract(
    operation: str,
    input_data: dict[str, Any],
    correlation_id: str | None = None,
) -> ModelContractEffect:
    """
    Create ONEX Effect Contract for node invocation.

    Args:
        operation: Operation name (e.g., 'package_container', 'load_image')
        input_data: Operation input parameters
        correlation_id: Optional correlation ID for tracking

    Returns:
        ModelContractEffect ready for node.execute_effect()
    """
    return ModelContractEffect(
        name=f"deployment_{operation}",
        version="1.0.0",
        description=f"Test {operation} operation",
        correlation_id=correlation_id if correlation_id else uuid4(),
        input_state={
            "operation": operation,
            "operation_type": operation,  # Receiver uses 'operation_type'
            "input": input_data,
            **input_data,  # Also merge directly for receiver compatibility
        },
        output_state={},
    )


async def test_sender_package_operation():
    """Test sender node package_container operation via ONEX interface."""
    print("\n" + "=" * 80)
    print("TEST 1: Sender Node - Package Container (ONEX Interface)")
    print("=" * 80)

    # Initialize ONEX container and sender node
    container = create_onex_container(
        config={"package_dir": "/tmp/test_deployment_packages"}
    )
    sender = NodeDeploymentSenderEffect(container)

    print("✓ Sender node initialized")
    print(f"  - Node ID: {sender.node_id}")
    print(f"  - Package dir: {sender.package_dir}")

    # Create ONEX Effect Contract
    contract = create_effect_contract(
        operation="package_container",
        input_data={
            "image_name": "test-onex-deployment:v1.0.0",
            "registry_url": "",
            "compression": "gzip",
        },
    )

    print("\n✓ Contract created")
    print(f"  - Operation: {contract.input_state.get('operation')}")
    print(f"  - Correlation ID: {contract.correlation_id}")

    # Execute via ONEX interface
    print("\n⏳ Executing effect via contract...")
    try:
        result = await sender.execute_effect(contract)

        print("\n✅ Package created successfully!")
        print(f"  - Package path: {result.package_path}")
        print(f"  - Package size: {result.package_size_mb:.2f} MB")
        print(f"  - Checksum: {result.package_checksum[:16]}...")
        print(f"  - Build time: {result.build_duration_ms}ms")

        return result

    except Exception as e:
        print(f"\n❌ Packaging failed: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_receiver_load_operation(package_path: str, checksum: str):
    """Test receiver node load_image operation via ONEX interface."""
    print("\n" + "=" * 80)
    print("TEST 2: Receiver Node - Load Image (ONEX Interface)")
    print("=" * 80)

    # Initialize ONEX container and receiver node
    container = create_onex_container(
        config={"package_dir": "/tmp/test_deployment_packages"}
    )
    receiver = NodeDeploymentReceiverEffect(container)

    print("✓ Receiver node initialized")
    print(f"  - Node ID: {receiver.node_id}")
    print(f"  - Docker host: {receiver.docker_client.docker_host}")

    # Create ONEX Effect Contract for load_image operation
    contract = create_effect_contract(
        operation="load_image",
        input_data={
            "package_path": package_path,
            "expected_checksum": checksum,
            "validate_checksum": True,
        },
    )

    print("\n✓ Contract created for load_image")
    print(f"  - Package: {package_path}")
    print(f"  - Checksum: {checksum[:16]}...")

    # Execute via ONEX interface
    print("\n⏳ Executing effect via contract...")
    try:
        result = await receiver.execute_effect(contract)

        print("\n✅ Image loaded successfully!")
        print(f"  - Image ID: {result.image_id[:16]}...")
        print(f"  - Image tags: {result.image_tags}")
        print(f"  - Load time: {result.load_duration_ms}ms")
        print(f"  - Checksum valid: {result.checksum_valid}")

        return result

    except Exception as e:
        print(f"\n❌ Image load failed: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_receiver_deploy_operation(image_name: str):
    """Test receiver node deploy_container operation via ONEX interface."""
    print("\n" + "=" * 80)
    print("TEST 3: Receiver Node - Deploy Container (ONEX Interface)")
    print("=" * 80)

    # Initialize ONEX container and receiver node
    container = create_onex_container()
    receiver = NodeDeploymentReceiverEffect(container)

    print("✓ Receiver node initialized")

    # Create ONEX Effect Contract for deploy_container operation
    contract = create_effect_contract(
        operation="deploy_container",
        input_data={
            "image_name": image_name,
            "container_config": {
                "name": "test-onex-deployment-container",
                "detach": True,
                "remove": True,  # Auto-remove after exit
                "environment": {"TEST_ENV": "onex_deployment_test"},
            },
        },
    )

    print("\n✓ Contract created for deploy_container")
    print(f"  - Image: {image_name}")

    # Execute via ONEX interface
    print("\n⏳ Executing effect via contract...")
    try:
        result = await receiver.execute_effect(contract)

        print("\n✅ Container deployed successfully!")
        print(f"  - Container ID: {result.container_id[:16]}...")
        print(f"  - Status: {result.container_status}")
        print(f"  - Deploy time: {result.deploy_duration_ms}ms")

        # Wait for container to complete
        print("\n⏳ Waiting for container execution...")
        await asyncio.sleep(2)

        # Check container logs
        import docker

        client = docker.from_env()
        try:
            container_obj = client.containers.get(result.container_id)
            logs = container_obj.logs().decode("utf-8")
            print("\n📋 Container output:")
            print(f"  {logs.strip()}")

            # Container should auto-remove, but check
            status = container_obj.status
            print(f"  - Final status: {status}")

        except docker.errors.NotFound:
            print("  - Container auto-removed (expected)")
        except Exception as e:
            print(f"  - Error checking container: {e}")

        return result

    except Exception as e:
        print(f"\n❌ Container deployment failed: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_full_deployment_workflow():
    """Test complete ONEX deployment workflow: package → load → deploy."""
    print("\n" + "=" * 80)
    print("ONEX DEPLOYMENT NODE TEST - Full Workflow")
    print("Testing actual node.py implementations, not standalone wrappers")
    print("=" * 80)

    # Step 1: Package container using sender node
    package_result = await test_sender_package_operation()
    if not package_result:
        print("\n❌ TEST FAILED: Packaging failed")
        return False

    # Step 2: Load image using receiver node
    load_result = await test_receiver_load_operation(
        package_path=package_result.package_path,
        checksum=package_result.package_checksum,
    )
    if not load_result:
        print("\n❌ TEST FAILED: Image load failed")
        return False

    # Step 3: Deploy container using receiver node
    deploy_result = await test_receiver_deploy_operation(
        image_name=(
            load_result.image_tags[0]
            if load_result.image_tags
            else "test-onex-deployment:v1.0.0"
        )
    )
    if not deploy_result:
        print("\n❌ TEST FAILED: Container deployment failed")
        return False

    # Success!
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - ONEX Deployment Workflow Complete")
    print("=" * 80)
    print("\nWorkflow Summary:")
    print(f"  1. ✅ Package created: {package_result.package_size_mb:.2f} MB")
    print(f"  2. ✅ Image loaded: {load_result.load_duration_ms}ms")
    print(f"  3. ✅ Container deployed: {deploy_result.deploy_duration_ms}ms")
    print(
        f"\nTotal workflow time: {package_result.build_duration_ms + load_result.load_duration_ms + deploy_result.deploy_duration_ms}ms"
    )

    return True


async def test_metrics_operation():
    """Test sender node metrics retrieval."""
    print("\n" + "=" * 80)
    print("TEST 4: Sender Node - Get Metrics")
    print("=" * 80)

    container = create_onex_container()
    sender = NodeDeploymentSenderEffect(container)

    # Create contract for get_metrics operation
    contract = create_effect_contract(
        operation="get_metrics",
        input_data={},
    )

    print("⏳ Retrieving metrics...")
    result = await sender.execute_effect(contract)

    print("\n📊 Sender Metrics:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    print("\n🚀 Starting ONEX Deployment Node Tests")
    print("Testing: NodeDeploymentSenderEffect + NodeDeploymentReceiverEffect")
    print("Method: Proper ONEX Effect Node interface via execute_effect(contract)")

    # Run full workflow test
    success = asyncio.run(test_full_deployment_workflow())

    # Run metrics test
    asyncio.run(test_metrics_operation())

    exit(0 if success else 1)
