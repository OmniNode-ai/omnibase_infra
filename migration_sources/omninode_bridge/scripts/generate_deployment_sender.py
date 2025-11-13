#!/usr/bin/env python3
"""
Direct generation script for deployment_sender_effect node.

Uses TemplateEngine directly without requiring the event-driven orchestration.
"""

import asyncio
import sys
from pathlib import Path

from src.omninode_bridge.codegen.node_classifier import (
    EnumNodeType,
    ModelClassificationResult,
)
from src.omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from src.omninode_bridge.codegen.template_engine import TemplateEngine


async def main():
    """Generate deployment_sender_effect node."""

    # Configuration
    prompt = """Create deployment sender effect node for packaging and transferring Docker containers to remote systems.
Builds Docker images, creates compressed packages with BLAKE3 checksums, and transfers via HTTP/rsync to remote receivers.
Publishes Kafka events for lifecycle tracking.
Implements io_operations: package_container (build/export/compress/checksum), transfer_package (validate/upload/verify), publish_transfer_event.
Performance: <20s image build, <10s transfer for 1GB packages."""

    output_dir = Path("./src/omninode_bridge/nodes/deployment_sender_effect/v1_0_0")

    print("🚀 Generating deployment_sender_effect node...")
    print(f"   Output: {output_dir}")
    print()

    # Create classification result
    classification = ModelClassificationResult(
        node_type=EnumNodeType.EFFECT,
        confidence=1.0,
        template_name="effect",
        template_variant=None,
        primary_indicators=["docker operations", "deployment", "network transfer"],
        reasoning="Effect node for Docker container deployment operations",
        recommended_subcontracts=[
            "effect_operations",
            "kafka_events",
            "state_management",
            "docker_operations",
        ],
        io_operations=[
            "package_container",
            "transfer_package",
            "publish_transfer_event",
        ],
        side_effects=[
            "docker_image_build",
            "filesystem_write",
            "network_io",
            "kafka_producer_call",
        ],
    )

    # Create PRD requirements
    prd_requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="deployment_sender",
        domain="deployment",
        business_description=prompt,
        operations=["package_container", "transfer_package", "publish_transfer_event"],
        features=[
            "Build Docker images from Dockerfiles",
            "Export images to compressed packages",
            "Generate BLAKE3 checksums for integrity",
            "Transfer packages via HTTP/rsync",
            "Publish lifecycle events to Kafka",
        ],
        performance_requirements={
            "image_build_seconds": 20,
            "package_transfer_seconds": 10,
            "throughput_deployments_per_hour": 100,
            "max_concurrent_transfers": 5,
        },
        dependencies={
            "docker": ">=7.0.0",
            "httpx": ">=0.27.0",
            "blake3": ">=0.4.0",
            "pydantic": ">=2.0.0",
            "aiokafka": ">=0.11.0",
            "aiofiles": ">=24.0.0",
        },
        data_models=[
            "ModelContainerPackageInput",
            "ModelContainerPackageOutput",
            "ModelPackageTransferInput",
            "ModelPackageTransferOutput",
            "ModelKafkaPublishInput",
            "ModelKafkaPublishOutput",
        ],
    )

    # Initialize template engine
    template_dir = Path("./src/omninode_bridge/codegen/templates")
    engine = TemplateEngine(templates_directory=template_dir)

    # Generate artifacts
    print("📝 Generating code from templates...")
    artifacts = await engine.generate(
        requirements=prd_requirements,
        classification=classification,
        output_directory=output_dir,
    )

    # Write files to disk
    print("\n💾 Writing files to disk...")
    output_dir.mkdir(parents=True, exist_ok=True)

    files_written = []
    for filename, content in artifacts.get_all_files().items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        files_written.append(str(file_path))
        print(f"   ✓ {file_path}")

    # Copy pre-designed contract
    contract_source = Path("./contracts/effects/deployment_sender_effect.yaml")
    contract_dest = output_dir / "contract.yaml"

    if contract_source.exists():
        print("\n📋 Copying pre-designed contract...")
        import shutil

        shutil.copy2(contract_source, contract_dest)
        print(f"   ✓ {contract_dest}")

    # Summary
    print("\n✅ Generation complete!")
    print(f"   Node type: {artifacts.node_type}")
    print(f"   Node name: {artifacts.node_name}")
    print(f"   Service name: {artifacts.service_name}")
    print(f"   Files generated: {len(files_written)}")
    print(f"   Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
