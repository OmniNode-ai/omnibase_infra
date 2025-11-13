"""
Template variant system for ONEX v2.0 code generation.

This module provides the template variant system that selects the optimal
template based on requirements analysis and complexity assessment.
"""

from .variant_metadata import (
    VARIANT_METADATA_REGISTRY,
    EnumTemplateVariant,
    ModelTemplateSelection,
    ModelVariantMetadata,
)
from .variant_selector import ModelRequirementsAnalysis, VariantSelector

__all__ = [
    "EnumTemplateVariant",
    "ModelVariantMetadata",
    "ModelTemplateSelection",
    "VARIANT_METADATA_REGISTRY",
    "VariantSelector",
    "ModelRequirementsAnalysis",
]
