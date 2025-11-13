#!/usr/bin/env python3
"""
Template Selector for intelligent template variant selection (Phase 3, Task C8).

Analyzes requirements and selects the optimal template variant based on:
- Node type and use case
- Requirements complexity
- Performance constraints
- Integration needs
- Quality standards

Performance Target: <5ms per selection
Accuracy Target: >95% correct template selection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from omninode_bridge.codegen.models_contract import (
    EnumQualityLevel,
    EnumTemplateVariant,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Selection Models
# ============================================================================


@dataclass
class ModelTemplateSelection:
    """
    Results of template variant selection.

    Attributes:
        variant: Selected template variant
        confidence: Confidence score (0.0-1.0)
        template_path: Path to selected template (if known)
        patterns: Recommended patterns for this variant
        rationale: Human-readable explanation of selection
        selection_time_ms: Time taken for selection
        metadata: Additional selection metadata
    """

    variant: EnumTemplateVariant
    confidence: float
    template_path: Optional[Path] = None
    patterns: list[str] = field(default_factory=list)
    rationale: str = ""
    selection_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EnumComplexityLevel(str, Enum):
    """Requirement complexity levels."""

    SIMPLE = "simple"  # 1-2 operations, no dependencies
    MODERATE = "moderate"  # 3-5 operations, few dependencies
    COMPLEX = "complex"  # 6+ operations, many dependencies


# ============================================================================
# Template Selector
# ============================================================================


class TemplateSelector:
    """
    Intelligent template variant selector.

    Analyzes requirements and selects optimal template variant with confidence
    scoring and pattern recommendations.
    """

    def __init__(self, template_root: Optional[Path] = None):
        """
        Initialize template selector.

        Args:
            template_root: Root directory for templates (uses default if None)
        """
        self.template_root = template_root or self._get_default_template_root()

        logger.info(
            f"TemplateSelector initialized (template_root={self.template_root})"
        )

    def select_template(
        self,
        requirements: Any,
        node_type: str,
        target_environment: Optional[str] = None,
    ) -> ModelTemplateSelection:
        """
        Select optimal template variant based on requirements.

        Args:
            requirements: Contract or requirements object
            node_type: Node type (effect/compute/reducer/orchestrator)
            target_environment: Target environment (development/staging/production)

        Returns:
            ModelTemplateSelection with variant, confidence, and rationale

        Example:
            >>> selector = TemplateSelector()
            >>> selection = selector.select_template(
            ...     requirements=contract,
            ...     node_type="effect",
            ...     target_environment="production"
            ... )
            >>> print(f"Selected: {selection.variant} (confidence: {selection.confidence:.2f})")
        """
        import time

        start_time = time.perf_counter()

        # Analyze requirements
        complexity = self._analyze_complexity(requirements)
        quality_level = self._get_quality_level(requirements, target_environment)
        has_integrations = self._check_integrations(requirements)

        # Select variant based on analysis
        variant = self._select_variant(complexity, quality_level, has_integrations)

        # Calculate confidence
        confidence = self._calculate_confidence(
            variant, complexity, quality_level, has_integrations
        )

        # Determine recommended patterns
        patterns = self._recommend_patterns(
            variant, node_type, complexity, has_integrations
        )

        # Build rationale
        rationale = self._build_rationale(
            variant, complexity, quality_level, has_integrations, patterns
        )

        # Calculate selection time
        selection_time_ms = (time.perf_counter() - start_time) * 1000

        # Build template path
        template_path = self._build_template_path(node_type, variant)

        selection = ModelTemplateSelection(
            variant=variant,
            confidence=confidence,
            template_path=template_path,
            patterns=patterns,
            rationale=rationale,
            selection_time_ms=selection_time_ms,
            metadata={
                "complexity": complexity.value,
                "quality_level": quality_level,
                "has_integrations": has_integrations,
                "node_type": node_type,
            },
        )

        logger.info(
            f"Template selection: {variant.value} "
            f"(confidence: {confidence:.2f}, "
            f"time: {selection_time_ms:.2f}ms)"
        )

        return selection

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def _analyze_complexity(self, requirements: Any) -> EnumComplexityLevel:
        """
        Analyze requirements complexity.

        Args:
            requirements: Requirements object

        Returns:
            Complexity level (SIMPLE/MODERATE/COMPLEX)
        """
        # Count operations
        num_operations = 0
        if hasattr(requirements, "io_operations") and requirements.io_operations:
            num_operations = len(requirements.io_operations)
        elif hasattr(requirements, "operations") and requirements.operations:
            num_operations = len(requirements.operations)

        # Count dependencies
        num_dependencies = 0
        if hasattr(requirements, "dependencies") and requirements.dependencies:
            if isinstance(requirements.dependencies, dict) or isinstance(
                requirements.dependencies, list
            ):
                num_dependencies = len(requirements.dependencies)

        # Determine complexity
        if num_operations <= 2 and num_dependencies <= 2:
            return EnumComplexityLevel.SIMPLE
        elif num_operations <= 5 and num_dependencies <= 5:
            return EnumComplexityLevel.MODERATE
        else:
            return EnumComplexityLevel.COMPLEX

    def _get_quality_level(
        self, requirements: Any, target_environment: Optional[str]
    ) -> str:
        """
        Determine quality level from requirements or environment.

        Args:
            requirements: Requirements object
            target_environment: Target environment

        Returns:
            Quality level string (minimal/standard/production)
        """
        # Check if requirements specify quality level
        if hasattr(requirements, "generation"):
            generation_config = requirements.generation
            if hasattr(generation_config, "quality_level"):
                return generation_config.quality_level

        # Infer from target environment
        if target_environment:
            if target_environment.lower() in ("production", "prod"):
                return EnumQualityLevel.PRODUCTION.value
            elif target_environment.lower() in ("staging", "stage"):
                return EnumQualityLevel.STANDARD.value
            else:
                return EnumQualityLevel.MINIMAL.value

        # Default to standard
        return EnumQualityLevel.STANDARD.value

    def _check_integrations(self, requirements: Any) -> bool:
        """
        Check if requirements include external integrations.

        Args:
            requirements: Requirements object

        Returns:
            True if integrations detected
        """
        # Check for integration patterns in requirements
        integration_keywords = {
            "database",
            "api",
            "kafka",
            "redis",
            "consul",
            "http",
            "rest",
            "grpc",
            "postgres",
            "mysql",
        }

        # Check operations for integration keywords
        operations = []
        if hasattr(requirements, "io_operations") and requirements.io_operations:
            operations.extend([op.name.lower() for op in requirements.io_operations])
        elif hasattr(requirements, "operations") and requirements.operations:
            operations.extend([str(op).lower() for op in requirements.operations])

        # Check if any operation contains integration keywords
        for operation in operations:
            if any(keyword in operation for keyword in integration_keywords):
                return True

        # Check dependencies
        if hasattr(requirements, "dependencies") and requirements.dependencies:
            deps = (
                requirements.dependencies
                if isinstance(requirements.dependencies, list)
                else list(requirements.dependencies.keys())
            )
            deps_lower = [str(dep).lower() for dep in deps]
            if any(
                keyword in dep for dep in deps_lower for keyword in integration_keywords
            ):
                return True

        return False

    # ========================================================================
    # Variant Selection
    # ========================================================================

    def _select_variant(
        self,
        complexity: EnumComplexityLevel,
        quality_level: str,
        has_integrations: bool,
    ) -> EnumTemplateVariant:
        """
        Select variant based on analysis.

        Decision tree:
        1. If quality_level == production → PRODUCTION
        2. If complexity == SIMPLE and !has_integrations → MINIMAL
        3. Else → STANDARD

        Args:
            complexity: Requirements complexity
            quality_level: Quality level
            has_integrations: Whether integrations detected

        Returns:
            Selected template variant
        """
        # Production quality always uses production variant
        if quality_level == EnumQualityLevel.PRODUCTION.value:
            return EnumTemplateVariant.PRODUCTION

        # Simple requirements without integrations use minimal
        if (
            complexity == EnumComplexityLevel.SIMPLE
            and not has_integrations
            and quality_level == EnumQualityLevel.MINIMAL.value
        ):
            return EnumTemplateVariant.MINIMAL

        # Default to standard
        return EnumTemplateVariant.STANDARD

    def _calculate_confidence(
        self,
        variant: EnumTemplateVariant,
        complexity: EnumComplexityLevel,
        quality_level: str,
        has_integrations: bool,
    ) -> float:
        """
        Calculate confidence score for selection.

        Args:
            variant: Selected variant
            complexity: Requirements complexity
            quality_level: Quality level
            has_integrations: Whether integrations detected

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.9  # Base confidence

        # Adjust for clear production requirements
        if variant == EnumTemplateVariant.PRODUCTION:
            if quality_level == EnumQualityLevel.PRODUCTION.value:
                confidence = 0.99  # Very high confidence
            else:
                confidence = 0.85  # Lower if production not explicitly requested

        # Adjust for minimal
        elif variant == EnumTemplateVariant.MINIMAL:
            if complexity == EnumComplexityLevel.SIMPLE and not has_integrations:
                confidence = 0.95  # High confidence for clear minimal case
            else:
                confidence = 0.70  # Lower if not clearly minimal

        # Standard is always reasonably confident
        else:
            confidence = 0.90

        return confidence

    # ========================================================================
    # Pattern Recommendations
    # ========================================================================

    def _recommend_patterns(
        self,
        variant: EnumTemplateVariant,
        node_type: str,
        complexity: EnumComplexityLevel,
        has_integrations: bool,
    ) -> list[str]:
        """
        Recommend patterns based on variant and requirements.

        Args:
            variant: Selected variant
            node_type: Node type
            complexity: Requirements complexity
            has_integrations: Whether integrations detected

        Returns:
            List of recommended pattern names
        """
        patterns = []

        # Minimal variant - no additional patterns
        if variant == EnumTemplateVariant.MINIMAL:
            return patterns

        # Standard variant - core patterns
        if variant == EnumTemplateVariant.STANDARD:
            patterns.extend(["lifecycle", "error_handling"])
            if has_integrations:
                patterns.append("retry_logic")
            return patterns

        # Production variant - full patterns
        if variant == EnumTemplateVariant.PRODUCTION:
            patterns.extend(
                [
                    "lifecycle",
                    "health_checks",
                    "metrics",
                    "error_handling",
                    "retry_logic",
                ]
            )

            if has_integrations:
                patterns.append("circuit_breaker")

            if node_type in ("effect", "compute"):
                patterns.append("event_publishing")

            if complexity == EnumComplexityLevel.COMPLEX:
                patterns.append("consul_integration")

        return patterns

    # ========================================================================
    # Rationale Building
    # ========================================================================

    def _build_rationale(
        self,
        variant: EnumTemplateVariant,
        complexity: EnumComplexityLevel,
        quality_level: str,
        has_integrations: bool,
        patterns: list[str],
    ) -> str:
        """
        Build human-readable rationale for selection.

        Args:
            variant: Selected variant
            complexity: Requirements complexity
            quality_level: Quality level
            has_integrations: Whether integrations detected
            patterns: Recommended patterns

        Returns:
            Rationale string
        """
        reasons = []

        # Variant explanation
        if variant == EnumTemplateVariant.PRODUCTION:
            reasons.append(
                f"Selected PRODUCTION template (quality_level={quality_level})"
            )
        elif variant == EnumTemplateVariant.MINIMAL:
            reasons.append(
                f"Selected MINIMAL template (complexity={complexity.value}, "
                f"no integrations)"
            )
        else:
            reasons.append(
                f"Selected STANDARD template (complexity={complexity.value})"
            )

        # Integrations
        if has_integrations:
            reasons.append("Detected external integrations")

        # Patterns
        if patterns:
            reasons.append(f"Recommended patterns: {', '.join(patterns)}")

        return ". ".join(reasons)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _build_template_path(
        self, node_type: str, variant: EnumTemplateVariant
    ) -> Path:
        """
        Build template file path.

        Args:
            node_type: Node type
            variant: Template variant

        Returns:
            Path to template file
        """
        # Template path: <template_root>/node_templates/<node_type>_<variant>.py.jinja
        template_name = f"{node_type}_{variant.value}.py.jinja"
        return self.template_root / "node_templates" / template_name

    def _get_default_template_root(self) -> Path:
        """
        Get default template root directory.

        Returns:
            Path to templates directory
        """
        # Assume templates are in src/omninode_bridge/codegen/templates/
        current_file = Path(__file__)
        codegen_dir = current_file.parent
        return codegen_dir / "templates"


__all__ = [
    "TemplateSelector",
    "ModelTemplateSelection",
    "EnumComplexityLevel",
]
