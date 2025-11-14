"""ONEX Node Generation Pipeline.

This module provides utilities for generating ONEX nodes from templates and contracts.

Components:
- Template-based node generation (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
- Contract-to-model generation
- Contract validation and compliance checking

Usage:
    from omnibase_infra.generation import NodeGenerator

    generator = NodeGenerator(output_dir="/path/to/output")

    generator.generate_node(
        node_type="effect",
        repository_name="omnibase_infra",
        domain="infrastructure",
        microservice_name="postgres_adapter",
        business_description="PostgreSQL database adapter",
        external_system="PostgreSQL",
    )
"""

from .node_generator import NodeGenerator
from .utils.name_converter import NameConverter
from .utils.template_processor import TemplateProcessor

__all__ = [
    "NodeGenerator",
    "TemplateProcessor",
    "NameConverter",
]
