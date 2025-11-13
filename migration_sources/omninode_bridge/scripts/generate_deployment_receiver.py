#!/usr/bin/env python3
"""
Direct generation script for deployment receiver effect node.

Bypasses event-driven architecture for simpler generation.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omninode_bridge.codegen.node_classifier import (
    EnumNodeType,
    ModelClassificationResult,
)
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.template_engine import TemplateEngine


async def main():
    """Generate deployment receiver effect node."""

    # Define requirements
    requirements = ModelPRDRequirements(
        node_type="effect",  # Must be string, not EnumNodeType
        service_name="deployment_receiver",
        business_description=(
            "Receives and deploys Docker containers on remote systems. "
            "Handles Docker image packages with authentication, loads images "
            "into Docker daemon, deploys containers with configuration, runs "
            "health checks, and publishes Kafka events."
        ),
        operations=[
            "receive_package",  # Authenticate and accept image package
            "load_image",  # Import image to Docker
            "deploy_container",  # Start container with config
            "health_check",  # Verify container running
            "publish_deployment_event",  # Publish to Kafka
        ],
        features=[
            "HMAC authentication",
            "BLAKE3 hash validation",
            "IP whitelisting",
            "Sandbox execution",
            "Docker image loading",
            "Container deployment",
            "Health monitoring",
            "Kafka event publishing",
        ],
        domain="deployment_automation",
        dependencies={  # Must be dict, not list
            "docker": ">=7.0.0",
            "aiokafka": ">=0.10.0",
            "blake3": ">=0.4.0",
            "pydantic": ">=2.0.0",
        },
        performance_requirements={
            "image_load_time_ms": 3000,  # <3s image load
            "container_start_time_ms": 2000,  # <2s container start
            "auth_validation_ms": 100,  # <100ms auth check
            "health_check_timeout_ms": 5000,  # 5s health check timeout
        },
        best_practices=[
            "Validate all inputs before processing",
            "Use Docker SDK for image/container operations",
            "Implement circuit breaker for Docker API",
            "Log all security events",
            "Use async I/O for all network operations",
        ],
        code_examples=[],
        data_models=[
            "ModelDeploymentPackage",
            "ModelAuthCredentials",
            "ModelContainerConfig",
            "ModelHealthCheckResult",
        ],
    )

    # Define classification
    classification = ModelClassificationResult(
        node_type=EnumNodeType.EFFECT,
        template_name="docker_deployment_effect",
        template_variant="deployment_receiver",
        confidence=0.95,
        reasoning="Receives packages and performs I/O operations (Docker, network, Kafka)",
    )

    # Output directory
    output_dir = (
        Path(__file__).parent.parent
        / "src"
        / "omninode_bridge"
        / "nodes"
        / "deployment_receiver_effect"
        / "v1_0_0"
    )

    print("🚀 Generating deployment receiver effect node...")
    print(f"   Output directory: {output_dir}")
    print(f"   Node type: {classification.node_type.value}")
    print()

    # Create template engine
    engine = TemplateEngine(enable_inline_templates=True)

    # Generate artifacts
    artifacts = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=output_dir,
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write all files
    files_written = []
    for filename, content in artifacts.get_all_files().items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        files_written.append(str(file_path))
        print(f"✅ Generated: {filename}")

    print()
    print("=" * 60)
    print("✅ Generation complete!")
    print("=" * 60)
    print(f"   Node: {artifacts.node_name}")
    print(f"   Service: {artifacts.service_name}")
    print(f"   Files: {len(files_written)}")
    print()
    print("Generated files:")
    for file_path in sorted(files_written):
        rel_path = Path(file_path).relative_to(Path(__file__).parent.parent)
        print(f"   - {rel_path}")

    # Copy pre-designed contract
    contract_source = (
        Path(__file__).parent.parent
        / "contracts"
        / "effects"
        / "deployment_receiver_effect.yaml"
    )
    if contract_source.exists():
        print()
        print("📄 Copying pre-designed contract...")
        contract_dest = output_dir / "contract.yaml"
        contract_dest.write_text(contract_source.read_text())
        print(f"   ✅ Copied: {contract_source} -> {contract_dest}")
    else:
        print()
        print(f"⚠️  Warning: Pre-designed contract not found at {contract_source}")
        print("   Using generated contract instead")

    print()
    print("Next steps:")
    print(
        "   1. Review generated code in: src/omninode_bridge/nodes/deployment_receiver_effect/v1_0_0/"
    )
    print("   2. Implement TODO sections in node.py")
    print("   3. Run tests: pytest tests/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
