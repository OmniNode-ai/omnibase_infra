#!/usr/bin/env python3
"""
Enhanced Context Builder for LLM code generation (Phase 3, Task I3).

Builds comprehensive LLM context by aggregating:
- Contract requirements
- Template selections
- Pattern matches
- Mixin recommendations
- Operation details
- Best practices and examples

Performance Target: <50ms per context build
Size Target: <8K tokens per context
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from omninode_bridge.codegen.mixins.models import ModelMixinRecommendation
from omninode_bridge.codegen.pattern_library import ModelPatternMatch
from omninode_bridge.codegen.template_selector import ModelTemplateSelection

logger = logging.getLogger(__name__)


# ============================================================================
# Context Models
# ============================================================================


@dataclass
class ModelLLMContext:
    """
    Comprehensive LLM generation context.

    Attributes:
        operation_name: Name of operation to implement
        operation_description: Description of operation
        node_type: Node type (effect/compute/reducer/orchestrator)
        template_variant: Selected template variant
        patterns: Matched patterns with code examples
        mixins: Recommended mixins with usage
        requirements: Requirements and constraints
        examples: Similar code examples
        best_practices: ONEX best practices
        metadata: Additional metadata
        estimated_tokens: Estimated token count
    """

    operation_name: str
    operation_description: str
    node_type: str
    template_variant: str
    patterns: list[dict[str, Any]] = field(default_factory=list)
    mixins: list[dict[str, Any]] = field(default_factory=list)
    requirements: dict[str, Any] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    best_practices: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "operation_name": self.operation_name,
            "operation_description": self.operation_description,
            "node_type": self.node_type,
            "template_variant": self.template_variant,
            "patterns": self.patterns,
            "mixins": self.mixins,
            "requirements": self.requirements,
            "examples": self.examples,
            "best_practices": self.best_practices,
            "metadata": self.metadata,
            "estimated_tokens": self.estimated_tokens,
        }


# ============================================================================
# Enhanced Context Builder
# ============================================================================


class EnhancedContextBuilder:
    """
    Builds comprehensive LLM context from Phase 3 components.

    Aggregates data from:
    - Contract requirements
    - Template selection results
    - Pattern matching results
    - Mixin recommendations
    - Operation specifications
    """

    def __init__(self, max_tokens: int = 8000):
        """
        Initialize context builder.

        Args:
            max_tokens: Maximum tokens per context (default: 8000)
        """
        self.max_tokens = max_tokens
        logger.info(f"EnhancedContextBuilder initialized (max_tokens={max_tokens})")

    def build_context(
        self,
        requirements: Any,
        operation: Any,
        template_selection: ModelTemplateSelection,
        mixin_selection: list[ModelMixinRecommendation],
        pattern_matches: Optional[list[ModelPatternMatch]] = None,
        additional_examples: Optional[list[str]] = None,
    ) -> ModelLLMContext:
        """
        Build comprehensive LLM context.

        Args:
            requirements: Contract or requirements object
            operation: Operation specification
            template_selection: Template selection result
            mixin_selection: Mixin recommendations
            pattern_matches: Pattern matching results (optional)
            additional_examples: Additional code examples (optional)

        Returns:
            ModelLLMContext with aggregated information

        Example:
            >>> builder = EnhancedContextBuilder()
            >>> context = builder.build_context(
            ...     requirements=contract,
            ...     operation=operation,
            ...     template_selection=template_selection,
            ...     mixin_selection=mixin_recommendations,
            ...     pattern_matches=pattern_matches
            ... )
            >>> print(f"Context tokens: {context.estimated_tokens}")
        """
        import time

        start_time = time.perf_counter()

        # Extract operation details
        operation_name = self._get_operation_name(operation)
        operation_description = self._get_operation_description(operation, requirements)

        # Get node type
        node_type = self._get_node_type(requirements)

        # Build patterns section
        patterns = self._build_patterns_section(pattern_matches or [])

        # Build mixins section
        mixins = self._build_mixins_section(mixin_selection)

        # Build requirements section
        requirements_dict = self._build_requirements_section(requirements, operation)

        # Build examples section
        examples = self._build_examples_section(
            pattern_matches or [], additional_examples or []
        )

        # Build best practices section
        best_practices = self._build_best_practices_section(
            template_selection, mixin_selection
        )

        # Build metadata
        metadata = {
            "template_confidence": template_selection.confidence,
            "template_rationale": template_selection.rationale,
            "num_patterns": len(patterns),
            "num_mixins": len(mixins),
            "num_examples": len(examples),
        }

        # Estimate token count
        estimated_tokens = self._estimate_tokens(
            patterns, mixins, requirements_dict, examples, best_practices
        )

        # Build context
        context = ModelLLMContext(
            operation_name=operation_name,
            operation_description=operation_description,
            node_type=node_type,
            template_variant=template_selection.variant.value,
            patterns=patterns,
            mixins=mixins,
            requirements=requirements_dict,
            examples=examples,
            best_practices=best_practices,
            metadata=metadata,
            estimated_tokens=estimated_tokens,
        )

        build_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Context built: {estimated_tokens} tokens, "
            f"{len(patterns)} patterns, "
            f"{len(mixins)} mixins "
            f"(time: {build_time_ms:.2f}ms)"
        )

        return context

    # ========================================================================
    # Extraction Methods
    # ========================================================================

    def _get_operation_name(self, operation: Any) -> str:
        """Extract operation name."""
        if hasattr(operation, "name"):
            return operation.name
        elif hasattr(operation, "operation_name"):
            return operation.operation_name
        elif isinstance(operation, str):
            return operation
        return "unknown_operation"

    def _get_operation_description(self, operation: Any, requirements: Any) -> str:
        """Extract operation description."""
        # Try operation description
        if hasattr(operation, "description"):
            return operation.description

        # Try requirements description
        if hasattr(requirements, "business_description"):
            return requirements.business_description
        elif hasattr(requirements, "description"):
            return requirements.description

        return "No description provided"

    def _get_node_type(self, requirements: Any) -> str:
        """Extract node type."""
        if hasattr(requirements, "node_type"):
            return requirements.node_type
        return "effect"  # Default

    # ========================================================================
    # Section Builders
    # ========================================================================

    def _build_patterns_section(
        self, pattern_matches: list[ModelPatternMatch]
    ) -> list[dict[str, Any]]:
        """
        Build patterns section for LLM context.

        Args:
            pattern_matches: Pattern matching results

        Returns:
            List of pattern dictionaries
        """
        patterns = []

        for match in pattern_matches:
            pattern_info = match.pattern_info
            patterns.append(
                {
                    "name": pattern_info.name,
                    "category": pattern_info.category.value,
                    "description": pattern_info.description,
                    "relevance_score": match.relevance_score,
                    "match_reason": match.match_reason,
                    "tags": list(pattern_info.tags),
                }
            )

        return patterns

    def _build_mixins_section(
        self, mixin_selection: list[ModelMixinRecommendation]
    ) -> list[dict[str, Any]]:
        """
        Build mixins section for LLM context.

        Args:
            mixin_selection: Mixin recommendations

        Returns:
            List of mixin dictionaries
        """
        mixins = []

        for recommendation in mixin_selection:
            mixins.append(
                {
                    "name": recommendation.mixin_name,
                    "score": recommendation.score,
                    "explanation": recommendation.explanation,
                    "features": list(recommendation.matched_features),
                    "usage_example": recommendation.usage_example,
                }
            )

        return mixins

    def _build_requirements_section(
        self, requirements: Any, operation: Any
    ) -> dict[str, Any]:
        """
        Build requirements section for LLM context.

        Args:
            requirements: Requirements object
            operation: Operation specification

        Returns:
            Requirements dictionary
        """
        req_dict = {}

        # Add operation-specific requirements
        if hasattr(operation, "input_type"):
            req_dict["input_type"] = operation.input_type
        if hasattr(operation, "output_type"):
            req_dict["output_type"] = operation.output_type

        # Add general requirements
        if hasattr(requirements, "domain"):
            req_dict["domain"] = requirements.domain

        if hasattr(requirements, "operations"):
            req_dict["operations"] = requirements.operations

        if hasattr(requirements, "dependencies"):
            req_dict["dependencies"] = requirements.dependencies

        if hasattr(requirements, "performance_requirements"):
            req_dict["performance_requirements"] = requirements.performance_requirements

        # Add features
        if hasattr(requirements, "features"):
            req_dict["features"] = requirements.features

        return req_dict

    def _build_examples_section(
        self,
        pattern_matches: list[ModelPatternMatch],
        additional_examples: list[str],
    ) -> list[str]:
        """
        Build examples section for LLM context.

        Args:
            pattern_matches: Pattern matching results
            additional_examples: Additional code examples

        Returns:
            List of code example strings
        """
        examples = []

        # Add examples from patterns (if generators have them)
        # This would be enhanced with actual pattern example code

        # Add additional examples
        examples.extend(additional_examples)

        # Limit to avoid token overflow
        if len(examples) > 5:
            examples = examples[:5]

        return examples

    def _build_best_practices_section(
        self,
        template_selection: ModelTemplateSelection,
        mixin_selection: list[ModelMixinRecommendation],
    ) -> list[str]:
        """
        Build best practices section for LLM context.

        Args:
            template_selection: Template selection result
            mixin_selection: Mixin recommendations

        Returns:
            List of best practice strings
        """
        practices = []

        # Add ONEX v2.0 best practices
        practices.extend(
            [
                "Follow ONEX v2.0 patterns (async/await, structured error handling)",
                "Use type hints throughout implementation",
                "Include comprehensive logging with structured context",
                "Implement proper error handling with ModelOnexError",
            ]
        )

        # Add template-specific practices
        if template_selection.variant.value == "production":
            practices.extend(
                [
                    "Include circuit breaker for external calls",
                    "Add retry logic with exponential backoff",
                    "Implement comprehensive health checks",
                    "Add Prometheus metrics for observability",
                ]
            )

        # Add mixin-specific practices
        for mixin in mixin_selection:
            if "health" in mixin.mixin_name.lower():
                practices.append(
                    "Implement health checks for all external dependencies"
                )
            elif "metrics" in mixin.mixin_name.lower():
                practices.append("Track key business and operational metrics")

        # Deduplicate and limit
        practices = list(dict.fromkeys(practices))  # Deduplicate
        if len(practices) > 10:
            practices = practices[:10]

        return practices

    # ========================================================================
    # Token Estimation
    # ========================================================================

    def _estimate_tokens(
        self,
        patterns: list[dict[str, Any]],
        mixins: list[dict[str, Any]],
        requirements: dict[str, Any],
        examples: list[str],
        best_practices: list[str],
    ) -> int:
        """
        Estimate token count for context.

        Rough estimation: 1 token ≈ 4 characters

        Args:
            patterns: Patterns section
            mixins: Mixins section
            requirements: Requirements section
            examples: Examples section
            best_practices: Best practices section

        Returns:
            Estimated token count
        """
        import json

        # Convert to JSON strings for character counting
        patterns_str = json.dumps(patterns)
        mixins_str = json.dumps(mixins)
        requirements_str = json.dumps(requirements)
        examples_str = "\n".join(examples)
        practices_str = "\n".join(best_practices)

        total_chars = (
            len(patterns_str)
            + len(mixins_str)
            + len(requirements_str)
            + len(examples_str)
            + len(practices_str)
        )

        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = total_chars // 4

        # Add overhead for JSON structure and formatting
        estimated_tokens = int(estimated_tokens * 1.2)

        return estimated_tokens


__all__ = [
    "EnhancedContextBuilder",
    "ModelLLMContext",
]
