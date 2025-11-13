#!/usr/bin/env python3
"""
Jinja2 Strategy - Template-based Code Generation.

Wraps the existing TemplateEngine to provide template-based code generation
through the BaseGenerationStrategy interface.

Uses Jinja2 templates to generate ONEX-compliant nodes with:
- Pre-written templates for each node type
- Proper naming conventions (NodeXxxYyy)
- Complete contract definitions
- Test scaffolding
- Documentation

Best suited for:
- Standard CRUD operations
- Well-defined patterns with established templates
- Rapid prototyping without LLM overhead
- Cost-sensitive scenarios

ONEX v2.0 Compliance:
- Inherits from BaseGenerationStrategy
- Type-safe operations
- Performance monitoring
- Comprehensive error handling

Example Usage:
    >>> strategy = Jinja2Strategy()
    >>> info = strategy.get_strategy_info()
    >>> print(f"{info['name']}: {info['description']}")
    >>>
    >>> request = ModelGenerationRequest(
    ...     requirements=requirements,
    ...     classification=classification,
    ...     output_directory=Path("./generated_nodes"),
    ... )
    >>>
    >>> result = await strategy.generate(request)
    >>> print(f"Generated {result.artifacts.node_name}")
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from ..node_classifier import EnumNodeType
from ..template_engine import TemplateEngine
from .base import (
    BaseGenerationStrategy,
    EnumStrategyType,
    ModelGenerationRequest,
    ModelGenerationResult,
)

logger = logging.getLogger(__name__)


class Jinja2Strategy(BaseGenerationStrategy):
    """
    Template-based code generation strategy using Jinja2.

    Wraps the existing TemplateEngine to provide consistent template-based
    generation through the BaseGenerationStrategy interface.

    Features:
    - Zero LLM cost (template-based only)
    - Fast generation (sub-second for most nodes)
    - Deterministic output
    - Supports all node types (Effect, Compute, Reducer, Orchestrator)

    Limitations:
    - Requires templates for each node type
    - Limited to predefined patterns
    - No custom business logic generation (use LLM strategy for that)

    Performance:
    - Generation time: <1s for standard nodes
    - Memory usage: <50MB
    - No network calls (fully offline)
    """

    def __init__(
        self,
        templates_directory: Optional[Path] = None,
        enable_inline_templates: bool = True,
        enable_validation: bool = True,
    ):
        """
        Initialize Jinja2Strategy.

        Args:
            templates_directory: Path to templates directory (uses default if None)
            enable_inline_templates: Enable inline template fallback if files not found
            enable_validation: Enable validation hooks
        """
        # Initialize base strategy
        super().__init__(
            strategy_name="Jinja2 Template Strategy",
            strategy_type=EnumStrategyType.JINJA2,
            enable_validation=enable_validation,
        )

        # Initialize template engine
        self.template_engine = TemplateEngine(
            templates_directory=templates_directory,
            enable_inline_templates=enable_inline_templates,
        )

        self.logger.info(
            f"Jinja2Strategy initialized "
            f"(templates_dir={templates_directory or 'default'}, "
            f"inline={enable_inline_templates}, "
            f"validation={enable_validation})"
        )

    def supports_node_type(self, node_type: EnumNodeType) -> bool:
        """
        Check if Jinja2 strategy supports the given node type.

        Jinja2 strategy supports all standard ONEX node types via templates.

        Args:
            node_type: Node type to check

        Returns:
            True (supports all node types)
        """
        # Jinja2 templates available for all ONEX node types
        return node_type in [
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
            EnumNodeType.ORCHESTRATOR,
        ]

    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get Jinja2 strategy information and capabilities.

        Returns:
            Dictionary with strategy metadata:
            - name: "Jinja2 Template Strategy"
            - type: EnumStrategyType.JINJA2
            - supported_node_types: All ONEX node types
            - requires_llm: False
            - performance_profile: Generation metrics
            - description: Strategy description
        """
        return {
            "name": self.strategy_name,
            "type": self.strategy_type.value,
            "supported_node_types": [
                EnumNodeType.EFFECT.value,
                EnumNodeType.COMPUTE.value,
                EnumNodeType.REDUCER.value,
                EnumNodeType.ORCHESTRATOR.value,
            ],
            "requires_llm": False,
            "performance_profile": {
                "avg_generation_time_ms": 500,
                "max_generation_time_ms": 2000,
                "memory_usage_mb": 50,
                "cost_per_generation_usd": 0.0,
            },
            "description": (
                "Template-based code generation using Jinja2. "
                "Fast, deterministic, zero-cost generation for standard patterns. "
                "Best suited for well-defined CRUD operations and standard workflows."
            ),
            "limitations": [
                "No custom business logic generation",
                "Limited to predefined templates",
                "Requires template updates for new patterns",
            ],
            "best_for": [
                "Standard CRUD operations",
                "Well-defined patterns",
                "Rapid prototyping",
                "Cost-sensitive scenarios",
            ],
        }

    async def generate(
        self,
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        """
        Generate node code using Jinja2 templates.

        Wraps TemplateEngine.generate() to provide BaseGenerationStrategy-compliant generation.

        Args:
            request: Generation request with requirements and options

        Returns:
            ModelGenerationResult with generated artifacts and metadata

        Raises:
            ValueError: If validation fails in strict mode
            RuntimeError: If generation fails

        Example:
            >>> strategy = Jinja2Strategy()
            >>> request = ModelGenerationRequest(
            ...     requirements=requirements,
            ...     classification=classification,
            ...     output_directory=Path("./nodes"),
            ...     run_tests=True,
            ... )
            >>> result = await strategy.generate(request)
            >>> print(f"Generated: {result.artifacts.node_name}")
            >>> print(f"Time: {result.generation_time_ms}ms")
        """
        # Log generation start
        self.log_generation_start(request)

        # Validate requirements if enabled
        if self.enable_validation:
            is_valid, errors = self.validate_requirements(
                request.requirements,
                request.validation_level,
            )

            if not is_valid:
                error_msg = f"Requirements validation failed: {', '.join(errors)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Track generation time
        start_time = time.time()

        try:
            # Call wrapped TemplateEngine
            self.logger.info(
                f"Generating code with Jinja2: "
                f"service={request.requirements.service_name}, "
                f"node_type={request.classification.node_type.value}"
            )

            artifacts = await self.template_engine.generate(
                requirements=request.requirements,
                classification=request.classification,
                output_directory=request.output_directory,
                run_tests=request.run_tests,
                strict_mode=request.strict_mode,
            )

            # Calculate generation time
            generation_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"Jinja2 generation complete: "
                f"{artifacts.node_name} "
                f"({len(artifacts.get_all_files())} files, "
                f"{generation_time_ms:.0f}ms)"
            )

            # Build result
            result = ModelGenerationResult(
                artifacts=artifacts,
                strategy_used=EnumStrategyType.JINJA2,
                generation_time_ms=generation_time_ms,
                validation_passed=True,
                validation_errors=[],
                llm_used=False,
                intelligence_sources=[],
                correlation_id=request.correlation_id,
            )

            # Log completion
            self.log_generation_complete(result, generation_time_ms)

            return result

        except ValueError as e:
            self.logger.error(f"Jinja2 generation failed (validation): {e}")
            raise

        except Exception as e:
            self.logger.error(f"Jinja2 generation failed: {e}")
            raise RuntimeError(f"Jinja2 code generation failed: {e}") from e


__all__ = ["Jinja2Strategy"]
