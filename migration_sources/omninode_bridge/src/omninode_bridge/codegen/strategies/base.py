#!/usr/bin/env python3
"""
Base Strategy for Code Generation.

Defines the abstract interface for all code generation strategies.
Implements the Strategy pattern for pluggable generation approaches.

ONEX v2.0 Compliance:
- Strategy pattern for extensibility
- Type-safe interfaces with Pydantic models
- Comprehensive error handling
- Performance monitoring integration
"""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ..node_classifier import EnumNodeType, ModelClassificationResult
from ..prd_analyzer import ModelPRDRequirements
from ..template_engine import ModelGeneratedArtifacts

logger = logging.getLogger(__name__)


class EnumStrategyType(str, Enum):
    """Code generation strategy types."""

    JINJA2 = "jinja2"  # Jinja2 template-based generation
    TEMPLATE_LOADING = "template_loading"  # TemplateLoader + BusinessLogicGenerator
    HYBRID = "hybrid"  # Combines both approaches
    AUTO = "auto"  # Automatically selects best strategy


class EnumValidationLevel(str, Enum):
    """Validation strictness levels."""

    NONE = "none"  # No validation
    BASIC = "basic"  # Basic syntax and structure validation
    STANDARD = "standard"  # Standard validation with type checking
    STRICT = "strict"  # Strict validation with comprehensive checks


class ModelGenerationRequest(BaseModel):
    """
    Request model for code generation.

    Contains all parameters needed for generating node code.
    """

    # Core requirements
    requirements: ModelPRDRequirements = Field(
        ..., description="Extracted PRD requirements"
    )
    classification: ModelClassificationResult = Field(
        ..., description="Node type classification"
    )

    # Output configuration
    output_directory: Path = Field(
        ..., description="Target directory for generated files"
    )

    # Generation options
    strategy: EnumStrategyType = Field(
        default=EnumStrategyType.AUTO,
        description="Code generation strategy to use",
    )
    enable_llm: bool = Field(
        default=True,
        description="Enable LLM-powered business logic generation",
    )
    validation_level: EnumValidationLevel = Field(
        default=EnumValidationLevel.STANDARD,
        description="Validation strictness level",
    )

    # Test execution
    run_tests: bool = Field(
        default=False,
        description="Execute generated tests after code generation",
    )
    strict_mode: bool = Field(
        default=False,
        description="Raise exception if tests fail (vs. attach to artifacts)",
    )

    # Metadata
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Generation timestamp",
    )


class ModelGenerationResult(BaseModel):
    """
    Result model for code generation.

    Wraps generated artifacts with metadata about the generation process.
    """

    # Generated artifacts
    artifacts: ModelGeneratedArtifacts = Field(
        ..., description="Generated code artifacts"
    )

    # Strategy metadata
    strategy_used: EnumStrategyType = Field(
        ..., description="Strategy that generated the code"
    )
    generation_time_ms: float = Field(
        ..., ge=0.0, description="Generation time in milliseconds"
    )

    # Quality metrics
    validation_passed: bool = Field(..., description="Whether validation passed")
    validation_errors: list[str] = Field(
        default_factory=list,
        description="Validation errors (if any)",
    )

    # Intelligence metadata
    llm_used: bool = Field(default=False, description="Whether LLM was used")
    intelligence_sources: list[str] = Field(
        default_factory=list,
        description="Intelligence sources used (e.g., RAG, Archon MCP)",
    )

    # Metadata
    correlation_id: UUID = Field(..., description="Correlation ID for tracing")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Generation timestamp",
    )


class BaseGenerationStrategy(ABC):
    """
    Abstract base class for code generation strategies.

    Defines the interface that all generation strategies must implement.
    Provides common utility methods and validation hooks.

    Strategy Pattern:
    - Encapsulates code generation algorithms
    - Allows runtime strategy selection
    - Enables easy addition of new generation approaches
    """

    def __init__(
        self,
        strategy_name: str,
        strategy_type: EnumStrategyType,
        enable_validation: bool = True,
    ):
        """
        Initialize base strategy.

        Args:
            strategy_name: Human-readable strategy name
            strategy_type: Strategy type enum
            enable_validation: Enable validation hooks
        """
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.enable_validation = enable_validation
        self.logger = logging.getLogger(f"{__name__}.{strategy_name}")

    @abstractmethod
    async def generate(
        self,
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        """
        Generate node code using this strategy.

        Args:
            request: Generation request with requirements and options

        Returns:
            ModelGenerationResult with generated artifacts and metadata

        Raises:
            ValueError: If validation fails in strict mode
            RuntimeError: If generation fails
        """
        pass

    @abstractmethod
    def supports_node_type(self, node_type: EnumNodeType) -> bool:
        """
        Check if this strategy supports the given node type.

        Args:
            node_type: Node type to check

        Returns:
            True if supported, False otherwise
        """
        pass

    @abstractmethod
    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get strategy information and capabilities.

        Returns:
            Dictionary with strategy metadata:
            - name: Strategy name
            - type: Strategy type
            - supported_node_types: List of supported node types
            - requires_llm: Whether LLM is required
            - performance_profile: Expected performance characteristics
        """
        pass

    def validate_requirements(
        self,
        requirements: ModelPRDRequirements,
        validation_level: EnumValidationLevel,
    ) -> tuple[bool, list[str]]:
        """
        Validate requirements before generation.

        Args:
            requirements: PRD requirements to validate
            validation_level: Validation strictness level

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self.enable_validation or validation_level == EnumValidationLevel.NONE:
            return True, []

        errors = []

        # Basic validation
        if not requirements.service_name:
            errors.append("service_name is required")

        if not requirements.node_type:
            errors.append("node_type is required")

        if not requirements.business_description:
            errors.append("business_description is required")

        # Standard validation
        if validation_level in (
            EnumValidationLevel.STANDARD,
            EnumValidationLevel.STRICT,
        ):
            if not requirements.operations:
                errors.append("At least one operation is required")

            if not requirements.domain:
                errors.append("domain is required")

        # Strict validation
        if validation_level == EnumValidationLevel.STRICT:
            if not requirements.features:
                errors.append("At least one feature should be specified")

            if requirements.extraction_confidence < 0.5:
                errors.append(
                    f"Extraction confidence too low: {requirements.extraction_confidence} (min 0.5)"
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    def log_generation_start(self, request: ModelGenerationRequest) -> None:
        """Log generation start with context."""
        self.logger.info(
            f"Starting code generation with {self.strategy_name}",
            extra={
                "strategy": self.strategy_type.value,
                "node_type": request.classification.node_type.value,
                "service_name": request.requirements.service_name,
                "correlation_id": str(request.correlation_id),
            },
        )

    def log_generation_complete(
        self,
        result: ModelGenerationResult,
        duration_ms: float,
    ) -> None:
        """Log generation completion with metrics."""
        self.logger.info(
            f"Code generation complete with {self.strategy_name}",
            extra={
                "strategy": self.strategy_type.value,
                "node_name": result.artifacts.node_name,
                "generation_time_ms": duration_ms,
                "validation_passed": result.validation_passed,
                "correlation_id": str(result.correlation_id),
            },
        )


# Export
__all__ = [
    "BaseGenerationStrategy",
    "EnumStrategyType",
    "EnumValidationLevel",
    "ModelGenerationRequest",
    "ModelGenerationResult",
]
