#!/usr/bin/env python3
"""
Hybrid Strategy - Best Quality Code Generation.

Combines Jinja2Strategy (fast template generation) with TemplateLoadStrategy's
LLM enhancement capabilities for highest quality output.

Pipeline:
1. Generate base code with Jinja2Strategy (fast, deterministic)
2. Detect stubs in generated code
3. Enhance stubs with LLM (BusinessLogicGenerator)
4. Validate with QualityGatePipeline (strict mode)
5. Retry if validation fails (max 3 attempts)

This strategy provides:
- Best quality output (combines template precision + LLM intelligence)
- Automatic stub detection and replacement
- Strict validation with retry logic
- Comprehensive metrics tracking

Trade-offs:
- Higher cost (LLM usage for enhancement)
- Slower generation (2-5s vs <1s for pure Jinja2)
- Requires ZAI_API_KEY for LLM enhancement

ONEX v2.0 Compliance:
- Inherits from BaseGenerationStrategy
- Async/await throughout
- Structured error handling with ModelOnexError
- Comprehensive logging and metrics
- Type-safe operations

Example Usage:
    >>> import os
    >>> os.environ["ZAI_API_KEY"] = "your_api_key"  # pragma: allowlist secret
    >>> strategy = HybridStrategy(
    ...     enable_llm_enhancement=True,
    ...     enable_strict_validation=True,
    ...     max_retry_attempts=3
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
    >>> print(f"Quality score: {result.quality_score:.2f}")
    >>> print(f"Time: {result.generation_time_ms:.0f}ms")
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from omnibase_core import ModelOnexError

from ..business_logic.generator import BusinessLogicGenerator
from ..business_logic.injector import CodeInjector
from ..business_logic.models import ModelEnhancedArtifacts
from ..node_classifier import EnumNodeType
from ..quality_gates import QualityGatePipeline, ValidationResult
from .base import (
    BaseGenerationStrategy,
    EnumStrategyType,
    ModelGeneratedArtifacts,
    ModelGenerationRequest,
    ModelGenerationResult,
)
from .jinja2_strategy import Jinja2Strategy

logger = logging.getLogger(__name__)


class HybridStrategy(BaseGenerationStrategy):
    """
    Hybrid strategy combining Jinja2 templates with LLM enhancement.

    Pipeline execution:
    1. **Phase 1: Base Generation** - Jinja2Strategy generates template-based code
    2. **Phase 2: Stub Detection** - CodeInjector finds stubs needing implementation
    3. **Phase 3: LLM Enhancement** - BusinessLogicGenerator enhances stubs
    4. **Phase 4: Validation** - QualityGatePipeline validates with strict checks
    5. **Phase 5: Retry** - Retry enhancement if validation fails (max 3 attempts)

    Features:
    - Combines template precision with LLM intelligence
    - Automatic stub detection and replacement
    - Strict validation with quality gates
    - Retry logic for validation failures
    - Comprehensive metrics (Jinja2 + LLM + validation time)
    - Supports all node types

    Performance:
    - Generation time: 2-5s (Jinja2=500ms + LLM=1-3s + validation=500ms)
    - Cost: Medium (LLM usage only for stubs)
    - Quality: Highest (combines best of both approaches)

    Example:
        >>> strategy = HybridStrategy(
        ...     enable_llm_enhancement=True,
        ...     enable_strict_validation=True,
        ...     max_retry_attempts=3
        ... )
        >>> result = await strategy.generate(request)
        >>> print(f"Quality: {result.quality_score:.2f}")
    """

    def __init__(
        self,
        templates_directory: Optional[Path] = None,
        enable_llm_enhancement: bool = True,
        enable_strict_validation: bool = True,
        enable_validation: bool = True,
        max_retry_attempts: int = 3,
    ):
        """
        Initialize HybridStrategy.

        Args:
            templates_directory: Path to Jinja2 templates (uses default if None)
            enable_llm_enhancement: Enable LLM enhancement of stubs (requires ZAI_API_KEY)
            enable_strict_validation: Enable strict validation with quality gates
            enable_validation: Enable base validation hooks
            max_retry_attempts: Maximum retry attempts for validation failures (1-5)

        Raises:
            ValueError: If max_retry_attempts is invalid
            RuntimeError: If LLM enhancement enabled but API key not available
        """
        # Initialize base strategy
        super().__init__(
            strategy_name="Hybrid Strategy",
            strategy_type=EnumStrategyType.HYBRID,
            enable_validation=enable_validation,
        )

        # Validate retry attempts
        if not 1 <= max_retry_attempts <= 5:
            raise ValueError(
                f"max_retry_attempts must be between 1 and 5 (got {max_retry_attempts})"
            )

        self.enable_llm_enhancement = enable_llm_enhancement
        self.enable_strict_validation = enable_strict_validation
        self.max_retry_attempts = max_retry_attempts

        # Initialize Jinja2Strategy for base generation
        self.jinja2_strategy = Jinja2Strategy(
            templates_directory=templates_directory,
            enable_inline_templates=True,
            enable_validation=enable_validation,
        )
        self.logger.info("Jinja2Strategy initialized")

        # Initialize CodeInjector for stub detection
        self.code_injector = CodeInjector()
        self.logger.info("CodeInjector initialized")

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

        # Initialize QualityGatePipeline if validation enabled
        self.quality_gate_pipeline: Optional[QualityGatePipeline]
        if enable_strict_validation:
            self.quality_gate_pipeline = QualityGatePipeline(
                validation_level="strict",
                enable_mypy=True,
            )
            self.logger.info("QualityGatePipeline initialized (strict mode)")
        else:
            self.quality_gate_pipeline = None
            self.logger.info("Strict validation disabled")

        self.logger.info(
            f"HybridStrategy initialized "
            f"(templates_dir={templates_directory or 'default'}, "
            f"llm={enable_llm_enhancement}, "
            f"strict_validation={enable_strict_validation}, "
            f"max_retries={max_retry_attempts})"
        )

    def supports_node_type(self, node_type: EnumNodeType) -> bool:
        """
        Check if this strategy supports the given node type.

        HybridStrategy supports all node types (best quality strategy).

        Args:
            node_type: Node type to check

        Returns:
            True (supports all node types)
        """
        # Hybrid strategy supports all node types via Jinja2Strategy
        return self.jinja2_strategy.supports_node_type(node_type)

    def get_strategy_info(self) -> dict[str, Any]:
        """
        Get strategy information and capabilities.

        Returns:
            Dictionary with strategy metadata:
            - name: "Hybrid Strategy"
            - type: EnumStrategyType.HYBRID
            - supported_node_types: All ONEX node types
            - requires_llm: True (for best quality)
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
            "requires_llm": True,  # LLM required for best quality
            "performance_profile": {
                "avg_generation_time_ms": 3000,  # 3s average
                "max_generation_time_ms": 8000,  # 8s max
                "jinja2_generation_ms": 500,
                "llm_enhancement_ms": 2000,
                "validation_ms": 500,
                "memory_usage_mb": 200,
                "cost_per_generation_usd": 0.01,  # Approximate (varies by stubs)
            },
            "description": (
                "Hybrid strategy combining Jinja2 templates with LLM enhancement. "
                "Provides highest quality output by using fast template generation "
                "followed by intelligent LLM enhancement and strict validation. "
                "Best suited for production-grade code requiring high quality."
            ),
            "features": [
                "Fast Jinja2 base generation",
                "Intelligent LLM enhancement",
                "Automatic stub detection",
                "Strict validation with quality gates",
                "Retry logic for validation failures",
                "Comprehensive metrics tracking",
            ],
            "limitations": [
                "Requires ZAI_API_KEY for LLM enhancement",
                "Higher cost than pure template strategy",
                "Slower than Jinja2-only strategy",
                "Not suitable for offline environments",
            ],
            "best_for": [
                "Production-grade code generation",
                "Complex business logic requirements",
                "High quality output priority",
                "Projects with LLM budget",
            ],
        }

    async def generate(
        self,
        request: ModelGenerationRequest,
    ) -> ModelGenerationResult:
        """
        Generate node code using hybrid approach.

        Pipeline execution:
        1. Validate requirements
        2. Generate base code with Jinja2Strategy
        3. Detect stubs in generated code
        4. Enhance stubs with LLM (if enabled and stubs found)
        5. Validate enhanced code with QualityGatePipeline
        6. Retry if validation fails (max attempts)
        7. Return result with comprehensive metrics

        Args:
            request: Generation request with requirements and options

        Returns:
            ModelGenerationResult with generated artifacts and metadata

        Raises:
            ValueError: If validation fails in strict mode after all retries
            RuntimeError: If generation fails

        Example:
            >>> request = ModelGenerationRequest(
            ...     requirements=requirements,
            ...     classification=classification,
            ...     output_directory=Path("./output"),
            ...     enable_llm=True,
            ... )
            >>> result = await strategy.generate(request)
            >>> print(f"Generated: {result.artifacts.node_name}")
            >>> print(f"Quality: {result.quality_score:.2f}")
        """
        start_time = time.time()
        self.log_generation_start(request)

        # Track metrics
        jinja2_time_ms = 0.0
        llm_time_ms = 0.0
        validation_time_ms = 0.0
        retry_count = 0
        llm_used = False
        validation_result: Optional[ValidationResult] = None

        try:
            # Step 1: Validate requirements
            is_valid, errors = self.validate_requirements(
                request.requirements, request.validation_level
            )

            if not is_valid and request.strict_mode:
                raise ValueError(f"Requirements validation failed: {errors}")

            # Step 2: Generate base code with Jinja2Strategy
            self.logger.info("Phase 1: Generating base code with Jinja2Strategy")
            jinja2_start = time.time()

            jinja2_result = await self.jinja2_strategy.generate(request)
            jinja2_time_ms = (time.time() - jinja2_start) * 1000

            self.logger.info(
                f"Jinja2 generation complete: {jinja2_result.artifacts.node_name} "
                f"({jinja2_time_ms:.0f}ms)"
            )

            # Step 3: Detect stubs in generated code
            self.logger.info("Phase 2: Detecting stubs in generated code")
            stubs = self.code_injector.find_stubs(
                jinja2_result.artifacts.node_file,
                file_path="",
            )

            self.logger.info(f"Found {len(stubs)} stubs to enhance")

            # Step 4: Enhance with LLM if enabled and stubs found
            enhanced_artifacts: ModelGeneratedArtifacts | ModelEnhancedArtifacts = (
                jinja2_result.artifacts
            )

            if (
                len(stubs) > 0
                and request.enable_llm
                and self.enable_llm_enhancement
                and self.business_logic_generator
            ):
                self.logger.info(f"Phase 3: Enhancing {len(stubs)} stubs with LLM")

                # Retry loop for enhancement + validation
                for attempt in range(1, self.max_retry_attempts + 1):
                    retry_count = attempt

                    try:
                        # Enhance with LLM
                        llm_start = time.time()
                        enhanced_artifacts = await self.business_logic_generator.enhance_artifacts(
                            artifacts=jinja2_result.artifacts,
                            requirements=request.requirements,
                            context_data={
                                "service_name": request.requirements.service_name,
                                "business_description": request.requirements.business_description,
                                "node_type": request.requirements.node_type,
                                "operations": request.requirements.operations,
                                "features": request.requirements.features,
                                "domain": request.requirements.domain,
                            },
                        )
                        llm_time_ms = (time.time() - llm_start) * 1000
                        llm_used = True

                        self.logger.info(
                            f"LLM enhancement complete: "
                            f"{len(enhanced_artifacts.methods_generated)} methods, "
                            f"${enhanced_artifacts.total_cost_usd:.4f}, "
                            f"{llm_time_ms:.0f}ms "
                            f"(attempt {attempt}/{self.max_retry_attempts})"
                        )

                        # Step 5: Validate enhanced code
                        if self.enable_strict_validation and self.quality_gate_pipeline:
                            self.logger.info(
                                f"Phase 4: Validating enhanced code (attempt {attempt})"
                            )
                            validation_start = time.time()

                            # Get enhanced node file content
                            if isinstance(enhanced_artifacts, ModelEnhancedArtifacts):
                                enhanced_node_file = (
                                    enhanced_artifacts.enhanced_node_file
                                )
                            else:
                                enhanced_node_file = enhanced_artifacts.node_file

                            # Create validation context
                            from ..business_logic.models import GenerationContext

                            validation_context = GenerationContext(
                                node_type=str(request.requirements.node_type),
                                service_name=request.requirements.service_name,
                                business_description=request.requirements.business_description,
                                operations=request.requirements.operations,
                                features=request.requirements.features,
                            )

                            validation_result = (
                                await self.quality_gate_pipeline.validate(
                                    generated_code=enhanced_node_file,
                                    context=validation_context,
                                )
                            )
                            validation_time_ms = (time.time() - validation_start) * 1000

                            self.logger.info(
                                f"Validation complete: "
                                f"passed={validation_result.passed}, "
                                f"score={validation_result.quality_score:.2f}, "
                                f"issues={validation_result.total_issues_count}, "
                                f"{validation_time_ms:.0f}ms"
                            )

                            # Check if validation passed
                            if validation_result.passed:
                                self.logger.info(
                                    f"Validation passed on attempt {attempt}"
                                )
                                break
                            else:
                                self.logger.warning(
                                    f"Validation failed on attempt {attempt}: "
                                    f"{validation_result.failed_stages}"
                                )

                                # If not last attempt, retry
                                if attempt < self.max_retry_attempts:
                                    self.logger.info(
                                        f"Retrying enhancement (attempt {attempt + 1}/{self.max_retry_attempts})"
                                    )
                                    continue
                                else:
                                    # Last attempt failed
                                    if request.strict_mode:
                                        error_msg = (
                                            f"Validation failed after {self.max_retry_attempts} attempts. "
                                            f"Failed stages: {validation_result.failed_stages}. "
                                            f"Issues: {validation_result.all_issues[:3]}"
                                        )
                                        raise ValueError(error_msg)
                                    else:
                                        self.logger.warning(
                                            f"Validation failed after {self.max_retry_attempts} attempts "
                                            f"(non-strict mode, continuing)"
                                        )
                        else:
                            # Validation disabled, accept enhancement
                            break

                    except ModelOnexError as e:
                        self.logger.error(
                            f"LLM enhancement failed on attempt {attempt}: {e.message}"
                        )

                        if attempt >= self.max_retry_attempts:
                            # Last attempt, fall back to Jinja2 result
                            self.logger.warning(
                                "Falling back to Jinja2-only result after enhancement failures"
                            )
                            enhanced_artifacts = jinja2_result.artifacts
                            llm_used = False
                        else:
                            self.logger.info(
                                f"Retrying enhancement (attempt {attempt + 1}/{self.max_retry_attempts})"
                            )

            else:
                self.logger.info(
                    f"Phase 3: Skipping LLM enhancement "
                    f"(stubs={len(stubs)}, enable_llm={request.enable_llm}, "
                    f"llm_available={self.business_logic_generator is not None})"
                )

            # Step 6: Build final artifacts
            final_artifacts: ModelGeneratedArtifacts
            if llm_used and isinstance(enhanced_artifacts, ModelEnhancedArtifacts):
                # Update original artifacts with enhanced node file
                final_artifacts = enhanced_artifacts.original_artifacts
                final_artifacts.node_file = enhanced_artifacts.enhanced_node_file
            else:
                # enhanced_artifacts is ModelGeneratedArtifacts (either LLM not used or enhancement failed)
                assert isinstance(enhanced_artifacts, ModelGeneratedArtifacts)
                final_artifacts = enhanced_artifacts

            # Calculate total generation time
            total_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"Hybrid generation complete: "
                f"jinja2={jinja2_time_ms:.0f}ms, "
                f"llm={llm_time_ms:.0f}ms, "
                f"validation={validation_time_ms:.0f}ms, "
                f"total={total_time_ms:.0f}ms, "
                f"retries={retry_count - 1}"
            )

            # Step 7: Build result
            result = ModelGenerationResult(
                artifacts=final_artifacts,
                strategy_used=EnumStrategyType.HYBRID,
                generation_time_ms=total_time_ms,
                validation_passed=(
                    validation_result.passed if validation_result else is_valid
                ),
                validation_errors=(
                    validation_result.all_issues if validation_result else errors
                ),
                llm_used=llm_used,
                intelligence_sources=(
                    ["jinja2", "business_logic_generator", "quality_gate_pipeline"]
                    if llm_used
                    else ["jinja2"]
                ),
                correlation_id=request.correlation_id,
            )

            # Add quality score if available
            if validation_result:
                # Store quality score in artifacts metadata (if supported)
                # For now, log it
                self.logger.info(
                    f"Quality score: {validation_result.quality_score:.2f}"
                )

            self.log_generation_complete(result, total_time_ms)

            return result

        except ValueError as e:
            self.logger.error(f"Hybrid generation failed (validation): {e}")
            raise

        except Exception as e:
            self.logger.error(f"Hybrid generation failed: {e}")
            raise RuntimeError(f"Hybrid code generation failed: {e}") from e

    async def cleanup(self) -> None:
        """
        Cleanup strategy resources.

        Releases LLM connection resources if LLM is enabled.
        """
        if self.business_logic_generator:
            await self.business_logic_generator.cleanup()
            self.logger.info("HybridStrategy resources cleaned up")


__all__ = ["HybridStrategy"]
