#!/usr/bin/env python3
"""
Model conversion utilities for code generation pipeline.

Converts between different artifact models to enable integration between
TemplateEngineLoader and BusinessLogicGenerator.

ONEX v2.0 Compliance:
- Type-safe conversions
- Preserves stub information
- Maintains metadata integrity
"""

import logging
from datetime import UTC, datetime

from .template_engine import ModelGeneratedArtifacts
from .template_engine_loader.models import ModelTemplateArtifacts

logger = logging.getLogger(__name__)


class ArtifactConverter:
    """
    Converts template artifacts to generated artifacts format.

    Enables integration between TemplateEngineLoader (loads templates)
    and BusinessLogicGenerator (enhances stubs with LLM).
    """

    @staticmethod
    def template_to_generated(
        template_artifacts: ModelTemplateArtifacts,
        service_name: str,
        node_class_name: str,
    ) -> ModelGeneratedArtifacts:
        """
        Convert ModelTemplateArtifacts to ModelGeneratedArtifacts.

        Args:
            template_artifacts: Artifacts from TemplateEngineLoader
            service_name: Service name (e.g., "postgres_crud")
            node_class_name: Node class name (e.g., "NodePostgresCRUDEffect")

        Returns:
            ModelGeneratedArtifacts compatible with BusinessLogicGenerator

        Example:
            >>> template_artifacts = engine.load_template("effect", "v1_0_0")
            >>> generated = ArtifactConverter.template_to_generated(
            ...     template_artifacts=template_artifacts,
            ...     service_name="postgres_crud",
            ...     node_class_name="NodePostgresCRUDEffect"
            ... )
            >>> # Now can pass to BusinessLogicGenerator
            >>> enhanced = await generator.enhance_artifacts(generated, requirements)
        """
        logger.info(
            f"Converting template artifacts for {node_class_name} "
            f"({len(template_artifacts.stubs)} stubs detected)"
        )

        # Generate minimal contract file (BusinessLogicGenerator only needs node_file)
        # Contract will be generated separately if needed
        contract_content = _generate_minimal_contract(
            template_artifacts.node_type,
            template_artifacts.version,
            service_name,
        )

        # Generate minimal __init__.py
        init_content = _generate_init_file(node_class_name)

        # Create ModelGeneratedArtifacts
        # Note: Most fields are optional - BusinessLogicGenerator only needs node_file
        generated = ModelGeneratedArtifacts(
            node_file=template_artifacts.template_code,
            contract_file=contract_content,
            init_file=init_content,
            node_type=template_artifacts.node_type,
            node_name=node_class_name,
            service_name=service_name,
            generated_at=datetime.now(UTC),
            # Optional fields - leave empty for now
            models={},
            tests={},
            documentation={},
            output_directory=template_artifacts.template_path.parent,
            test_results=None,
            failure_analysis=None,
        )

        logger.info(
            f"Converted template artifacts successfully: "
            f"{len(template_artifacts.template_code)} chars, "
            f"{len(template_artifacts.stubs)} stubs"
        )

        return generated


def _generate_minimal_contract(node_type: str, version: str, service_name: str) -> str:
    """
    Generate minimal contract YAML for converted artifacts.

    BusinessLogicGenerator doesn't validate contracts, so this is just
    a placeholder to satisfy the ModelGeneratedArtifacts interface.
    """
    return f"""# {service_name} Contract
# Generated from template: {node_type}/{version}

name: "{service_name}"
version:
  major: 1
  minor: 0
  patch: 0
description: "Generated from template"
node_type: "{node_type.lower()}"

# Minimal fields to satisfy interface
input_model: "ModelRequest"
output_model: "ModelResponse"
tool_specification:
  tool_name: "{service_name}"
  main_tool_class: "omninode_bridge.nodes.{service_name}.v1_0_0.node.Node{service_name.title().replace('_', '')}Effect"
"""


def _generate_init_file(node_class_name: str) -> str:
    """Generate minimal __init__.py for converted artifacts."""
    return f'''"""
Generated node module.

ONEX v2.0 Compliant
"""

from .node import {node_class_name}

__all__ = ["{node_class_name}"]
'''


__all__ = ["ArtifactConverter"]
