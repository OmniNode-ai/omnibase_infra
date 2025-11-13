#!/usr/bin/env python3
"""
Template Load Strategy - Hand-written Templates + LLM Enhancement.

Loads hand-written Python templates from filesystem and optionally enhances them
with LLM-generated business logic using TemplateEngineLoader + BusinessLogicGenerator pipeline.

This strategy is ideal for:
- Custom nodes with complex, hand-written template bases
- Iterative development (template skeleton + LLM fill-in)
- Nodes requiring specific patterns not in Jinja2 templates
- Prototyping with real template files

ONEX v2.0 Compliance:
- Inherits from BaseGenerationStrategy
- Async/await throughout
- Structured error handling with ModelOnexError
- Comprehensive logging and metrics
- Type-safe operations

Example Usage:
    >>> import os
    >>> from pathlib import Path
    >>> from omninode_bridge.codegen.strategies import TemplateLoadStrategy
    >>>
    >>> # Initialize strategy with LLM enhancement
    >>> os.environ["ZAI_API_KEY"] = "your_api_key"  # pragma: allowlist secret
    >>> strategy = TemplateLoadStrategy(
    ...     template_directory=Path("templates/node_templates"),
    ...     enable_llm_enhancement=True
    ... )
    >>>
    >>> # Check capabilities
    >>> info = strategy.get_strategy_info()
    >>> print(f"{info['name']}: {info['description']}")
    >>>
    >>> # Generate node
    >>> request = ModelGenerationRequest(
    ...     requirements=requirements,
    ...     classification=classification,
    ...     output_directory=Path("./generated_nodes"),
    ...     enable_llm=True,
    ... )
    >>>
    >>> result = await strategy.generate(request)
    >>> print(f"Generated {result.artifacts.node_name}")
    >>> print(f"LLM cost: ${result.artifacts.total_cost_usd:.4f}")
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from omnibase_core import ModelOnexError

from ..business_logic.generator import BusinessLogicGenerator
from ..business_logic.models import ModelEnhancedArtifacts
from ..converters import ArtifactConverter
from ..node_classifier import EnumNodeType
from ..template_engine_loader.engine import TemplateEngine, TemplateEngineError
from .base import (
    BaseGenerationStrategy,
    EnumStrategyType,
    ModelGeneratedArtifacts,
    ModelGenerationRequest,
    ModelGenerationResult,
)

logger = logging.getLogger(__name__)


class TemplateLoadStrategy(BaseGenerationStrategy):
    """
    Strategy that loads hand-written templates with optional LLM enhancement.

    Pipeline:
    1. Load template from filesystem using TemplateEngine
    2. Detect stubs in template code (methods needing implementation)
    3. Convert to BusinessLogicGenerator format using ArtifactConverter
    4. Optionally enhance stubs with LLM-generated implementations
    5. Validate enhanced code
    6. Return complete implementation

    Features:
    - Template-based skeleton (fast, deterministic)
    - Optional LLM enhancement (intelligent business logic)
    - Supports all node types (effect/compute/reducer/orchestrator)
    - Automatic stub detection and replacement
    - Cost tracking for LLM usage

    Performance:
    - Template loading: <100ms
    - LLM enhancement: Variable (depends on stubs count)
    - Total: 200ms-5s depending on complexity and LLM usage

    Limitations:
    - Requires hand-written templates in filesystem
    - LLM enhancement requires ZAI_API_KEY
    - Template must exist for specified node_type/version
    """

    def __init__(
        self,
        template_directory: Optional[Path] = None,
        enable_llm_enhancement: bool = False,
        enable_validation: bool = True,
    ):
        """
        Initialize TemplateLoadStrategy.

        Args:
            template_directory: Directory containing node templates (uses default if None)
            enable_llm_enhancement: Enable LLM enhancement of stubs (requires ZAI_API_KEY)
            enable_validation: Enable template and code validation

        Raises:
            RuntimeError: If LLM enabled but API key not available
        """
        # Initialize base strategy
        super().__init__(
            strategy_name="Template Load Strategy",
            strategy_type=EnumStrategyType.TEMPLATE_LOADING,
            enable_validation=enable_validation,
        )

        self.template_directory = template_directory
        self.enable_llm_enhancement = enable_llm_enhancement

        # Initialize TemplateEngine
        try:
            self.template_engine = TemplateEngine(
                template_root=template_directory,
                enable_validation=enable_validation,
            )
            self.logger.info(
                f"TemplateEngine initialized (template_dir={template_directory or 'default'})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TemplateEngine: {e}") from e

        # Initialize BusinessLogicGenerator if LLM enabled
        if enable_llm_enhancement:
            try:
                self.business_logic_generator: Optional[BusinessLogicGenerator] = (
                    BusinessLogicGenerator(enable_llm=True)
                )
                self.logger.info("BusinessLogicGenerator initialized (LLM enabled)")
            except ModelOnexError as e:
                raise RuntimeError(
                    f"Failed to initialize BusinessLogicGenerator: {e.message}. "
                    "Ensure ZAI_API_KEY is set in environment."
                ) from e
        else:
            self.business_logic_generator = None
            self.logger.info("LLM enhancement disabled")

        self.logger.info(
            f"TemplateLoadStrategy initialized "
            f"(template_dir={template_directory or 'default'}, "
            f"llm={enable_llm_enhancement}, "
            f"validation={enable_validation})"
        )

    async def generate(
        self,
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        """
        Generate node code using template loading + LLM enhancement.

        Pipeline execution:
        1. Validate requirements
        2. Load template from filesystem
        3. Detect stubs in template
        4. Convert to generated artifacts format
        5. Enhance with LLM if enabled
        6. Validate generated code
        7. Return result with metrics

        Args:
            request: Generation request with requirements and options

        Returns:
            ModelGenerationResult with generated artifacts and metadata

        Raises:
            ValueError: If validation fails in strict mode
            RuntimeError: If template loading or generation fails

        Example:
            >>> request = ModelGenerationRequest(
            ...     requirements=requirements,
            ...     classification=classification,
            ...     output_directory=Path("./output"),
            ...     enable_llm=True,
            ... )
            >>> result = await strategy.generate(request)
            >>> print(f"Generated: {result.artifacts.node_name}")
        """
        start_time = time.time()
        self.log_generation_start(request)

        # Step 1: Validate requirements
        is_valid, errors = self.validate_requirements(
            request.requirements, request.validation_level
        )

        if not is_valid and request.strict_mode:
            raise ValueError(f"Requirements validation failed: {errors}")

        # Step 2: Load template
        node_type_str = request.classification.node_type.value
        version = "v1_0_0"  # Default version (could be in request)
        template_name = "node"  # Default template name

        self.logger.info(f"Loading template: {node_type_str}/{version}/{template_name}")

        try:
            template_artifacts = self.template_engine.load_template(
                node_type=node_type_str,
                version=version,
                template_name=template_name,
            )

            self.logger.info(
                f"Loaded template: {template_artifacts.template_path} "
                f"({len(template_artifacts.stubs)} stubs detected)"
            )

        except TemplateEngineError as e:
            raise RuntimeError(
                f"Failed to load template {node_type_str}/{version}/{template_name}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading template: {e}") from e

        # Step 3: Convert to ModelGeneratedArtifacts format
        service_name = request.requirements.service_name
        node_class_name = self._build_node_class_name(service_name, node_type_str)

        try:
            generated_artifacts = ArtifactConverter.template_to_generated(
                template_artifacts=template_artifacts,
                service_name=service_name,
                node_class_name=node_class_name,
            )

            self.logger.info(
                f"Converted template artifacts (node_name={node_class_name})"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to convert template artifacts: {e}") from e

        # Step 4: Enhance with LLM if enabled
        use_llm = request.enable_llm and self.enable_llm_enhancement
        llm_used = False
        enhanced_artifacts: ModelGeneratedArtifacts | ModelEnhancedArtifacts = (
            generated_artifacts
        )

        if use_llm:
            if not self.business_logic_generator:
                self.logger.warning(
                    "LLM enhancement requested but generator not initialized. "
                    "Skipping LLM enhancement."
                )
            else:
                self.logger.info(
                    f"Enhancing artifacts with LLM ({len(template_artifacts.stubs)} stubs)"
                )

                try:
                    enhanced_artifacts = (
                        await self.business_logic_generator.enhance_artifacts(
                            artifacts=generated_artifacts,
                            requirements=request.requirements,
                            context_data={},  # Could pass intelligence data here
                        )
                    )

                    llm_used = True
                    self.logger.info(
                        f"LLM enhancement complete: "
                        f"{len(enhanced_artifacts.methods_generated)} methods generated, "
                        f"${enhanced_artifacts.total_cost_usd:.4f} cost, "
                        f"{enhanced_artifacts.generation_success_rate:.1%} success rate"
                    )

                except ModelOnexError as e:
                    self.logger.error(f"LLM enhancement failed: {e.message}")
                    # Continue with un-enhanced template
                    self.logger.info("Continuing with un-enhanced template")
                except Exception as e:
                    self.logger.error(f"Unexpected LLM error: {e}")
                    # Continue with un-enhanced template

        # Step 5: Build final artifacts
        # If LLM was used, enhanced_artifacts is ModelEnhancedArtifacts
        # Otherwise, it's ModelGeneratedArtifacts
        if llm_used and isinstance(enhanced_artifacts, ModelEnhancedArtifacts):
            # Update the generated_artifacts with enhanced node file
            final_artifacts = enhanced_artifacts.original_artifacts
            final_artifacts.node_file = enhanced_artifacts.enhanced_node_file
        else:
            final_artifacts = generated_artifacts

        # Step 6: Build result
        generation_time_ms = (time.time() - start_time) * 1000

        result = ModelGenerationResult(
            artifacts=final_artifacts,
            strategy_used=EnumStrategyType.TEMPLATE_LOADING,
            generation_time_ms=generation_time_ms,
            validation_passed=is_valid,
            validation_errors=errors if not is_valid else [],
            llm_used=llm_used,
            intelligence_sources=(
                ["template_engine_loader", "business_logic_generator"]
                if llm_used
                else ["template_engine_loader"]
            ),
            correlation_id=request.correlation_id,
        )

        self.log_generation_complete(result, generation_time_ms)

        return result

    def supports_node_type(self, node_type: EnumNodeType) -> bool:
        """
        Check if this strategy supports the given node type.

        This strategy supports all node types IF templates exist for them.

        Args:
            node_type: Node type to check

        Returns:
            True if supported (template exists), False otherwise
        """
        # Check if template exists for this node type
        node_type_str = node_type.value
        version = "v1_0_0"
        template_name = "node"

        template_path = self._build_template_path(node_type_str, version, template_name)
        exists = template_path.exists()

        self.logger.debug(
            f"supports_node_type({node_type_str}): {exists} (template_path={template_path})"
        )

        return exists

    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get strategy information and capabilities.

        Returns:
            Dictionary with strategy metadata:
            - name: Strategy name
            - type: Strategy type
            - description: What this strategy does
            - supported_node_types: List of supported node types
            - requires_llm: Whether LLM is required
            - requires_templates: Whether templates are required
            - performance_profile: Expected performance characteristics
        """
        # Discover which node types have templates
        templates = self.template_engine.discover_templates()
        supported_types = list({t.node_type for t in templates})

        return {
            "name": self.strategy_name,
            "type": self.strategy_type.value,
            "description": "Load hand-written Python templates with optional LLM enhancement",
            "supported_node_types": supported_types,
            "requires_llm": False,  # LLM is optional
            "requires_templates": True,  # Templates are required
            "performance_profile": {
                "template_loading_ms": "<100",
                "llm_enhancement_ms": "variable (200-5000)",
                "total_generation_ms": "200-5000 (depends on LLM usage)",
            },
            "features": [
                "Hand-written template loading",
                "Automatic stub detection",
                "Optional LLM enhancement",
                "Cost tracking",
                "Validation",
            ],
            "limitations": [
                "Requires templates in filesystem",
                "LLM enhancement requires ZAI_API_KEY",
                "Template must exist for node_type/version",
            ],
        }

    async def cleanup(self) -> None:
        """
        Cleanup strategy resources.

        Releases LLM connection resources if LLM is enabled.
        """
        if self.business_logic_generator:
            await self.business_logic_generator.cleanup()
            self.logger.info("TemplateLoadStrategy resources cleaned up")

    def _build_template_path(
        self, node_type: str, version: str, template_name: str
    ) -> Path:
        """
        Build full path to template file.

        Args:
            node_type: Node type
            version: Version string (e.g., v1_0_0)
            template_name: Template file name without extension

        Returns:
            Full path to template file
        """
        # Use TemplateEngine's logic for path building
        return self.template_engine._build_template_path(
            node_type, version, template_name
        )

    def _build_node_class_name(self, service_name: str, node_type: str) -> str:
        """
        Build node class name from service name and type.

        Args:
            service_name: Service name in snake_case (e.g., "postgres_crud")
            node_type: Node type (e.g., "effect")

        Returns:
            Node class name (e.g., "NodePostgresCRUDEffect")

        Example:
            >>> strategy._build_node_class_name("postgres_crud", "effect")
            "NodePostgresCRUDEffect"
        """
        # Convert snake_case to PascalCase
        pascal_name = "".join(word.capitalize() for word in service_name.split("_"))

        # Build node class name: Node<PascalName><Type>
        node_class_name = f"Node{pascal_name}{node_type.capitalize()}"

        return node_class_name


__all__ = ["TemplateLoadStrategy"]
