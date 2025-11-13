#!/usr/bin/env python3
"""
Code Generation Service - Unified Facade.

Provides a unified entry point for code generation with pluggable strategies.
Unifies two parallel code generation systems:
- Jinja2 TemplateEngine (template-based)
- TemplateEngineLoader + BusinessLogicGenerator (LLM-powered)

IMPORTANT: Generated Nodes DO NOT Include main_standalone.py
- ONEX v2.0 nodes are invoked via orchestration contracts, NOT REST APIs
- Generated artifacts contain ONLY: node.py, contract.yaml, models, tests, __init__.py
- main_standalone.py is a LEGACY pattern for backward compatibility
- New nodes should use node.py as the sole entry point
- For REST API access, use a dedicated API gateway/orchestrator layer

ONEX v2.0 Compliance:
- Strategy pattern for extensibility
- Unified API across all generation approaches
- Comprehensive validation and error handling
- Performance monitoring and metrics
- Integration with Archon MCP intelligence

Architecture:
    CodeGenerationService (Facade)
        ↓
    Strategy Registry (Strategy Selection)
        ↓
    BaseGenerationStrategy (Abstract Interface)
        ↓
    [Jinja2Strategy | TemplateLoadingStrategy | HybridStrategy | AutoStrategy]
        ↓
    Generated Artifacts

Usage:
    >>> service = CodeGenerationService()
    >>> result = await service.generate_node(
    ...     requirements=requirements,
    ...     strategy="auto",
    ...     enable_llm=True,
    ...     validation_level="strict",
    ... )
    >>> print(f"Generated: {result.artifacts.node_name}")
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from .mixin_injector import MixinInjector
from .models_contract import ModelEnhancedContract
from .node_classifier import EnumNodeType, ModelClassificationResult, NodeClassifier
from .prd_analyzer import ModelPRDRequirements
from .strategies.base import (
    BaseGenerationStrategy,
    EnumStrategyType,
    EnumValidationLevel,
    ModelGenerationRequest,
    ModelGenerationResult,
)
from .template_engine import ModelGeneratedArtifacts
from .validation.models import ModelValidationResult
from .validation.validator import NodeValidator
from .yaml_contract_parser import YAMLContractParser

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Registry for code generation strategies.

    Manages strategy registration, discovery, and selection.
    Implements strategy pattern with runtime strategy selection.
    """

    def __init__(self):
        """Initialize strategy registry."""
        self._strategies: dict[EnumStrategyType, BaseGenerationStrategy] = {}
        self._default_strategy: Optional[EnumStrategyType] = None

    def register(
        self,
        strategy: BaseGenerationStrategy,
        is_default: bool = False,
    ) -> None:
        """
        Register a generation strategy.

        Args:
            strategy: Strategy instance to register
            is_default: Set as default strategy

        Raises:
            ValueError: If strategy type already registered
        """
        if strategy.strategy_type in self._strategies:
            raise ValueError(
                f"Strategy {strategy.strategy_type.value} already registered"
            )

        self._strategies[strategy.strategy_type] = strategy

        if is_default or self._default_strategy is None:
            self._default_strategy = strategy.strategy_type

        logger.info(
            f"Registered strategy: {strategy.strategy_name} ({strategy.strategy_type.value})"
        )

    def get_strategy(
        self,
        strategy_type: EnumStrategyType,
    ) -> Optional[BaseGenerationStrategy]:
        """
        Get strategy by type.

        Args:
            strategy_type: Strategy type to retrieve

        Returns:
            Strategy instance or None if not found
        """
        return self._strategies.get(strategy_type)

    def get_default_strategy(self) -> Optional[BaseGenerationStrategy]:
        """
        Get default strategy.

        Returns:
            Default strategy instance or None if not set
        """
        if self._default_strategy:
            return self._strategies.get(self._default_strategy)
        return None

    def list_strategies(self) -> list[dict[str, str]]:
        """
        List all registered strategies.

        Returns:
            List of strategy info dictionaries
        """
        return [
            {
                "name": strategy.strategy_name,
                "type": strategy.strategy_type.value,
                "is_default": strategy.strategy_type == self._default_strategy,
            }
            for strategy in self._strategies.values()
        ]

    def select_strategy(
        self,
        node_type: EnumNodeType,
        enable_llm: bool,
        prefer_strategy: Optional[EnumStrategyType] = None,
    ) -> BaseGenerationStrategy:
        """
        Select optimal strategy based on requirements.

        Strategy selection logic:
        1. If prefer_strategy specified and supports node_type → use it
        2. If enable_llm and TemplateLoadingStrategy available → use it
        3. If Jinja2Strategy available → use it
        4. Fall back to default strategy

        Args:
            node_type: Node type being generated
            enable_llm: Whether LLM is enabled
            prefer_strategy: Preferred strategy type (optional)

        Returns:
            Selected strategy instance

        Raises:
            RuntimeError: If no suitable strategy found
        """
        # Try preferred strategy first
        if prefer_strategy:
            strategy = self.get_strategy(prefer_strategy)
            if strategy and strategy.supports_node_type(node_type):
                logger.info(f"Selected preferred strategy: {strategy.strategy_name}")
                return strategy

        # Try LLM-powered strategy if enabled
        if enable_llm:
            strategy = self.get_strategy(EnumStrategyType.TEMPLATE_LOADING)
            if strategy and strategy.supports_node_type(node_type):
                logger.info(f"Selected LLM-powered strategy: {strategy.strategy_name}")
                return strategy

        # Try Jinja2 strategy
        strategy = self.get_strategy(EnumStrategyType.JINJA2)
        if strategy and strategy.supports_node_type(node_type):
            logger.info(f"Selected template strategy: {strategy.strategy_name}")
            return strategy

        # Fall back to default
        default = self.get_default_strategy()
        if default and default.supports_node_type(node_type):
            logger.info(f"Selected default strategy: {default.strategy_name}")
            return default

        raise RuntimeError(
            f"No suitable strategy found for node_type={node_type.value}, "
            f"enable_llm={enable_llm}"
        )


class CodeGenerationService:
    """
    Unified Code Generation Service.

    Facade that provides a single entry point for all code generation.
    Uses strategy pattern to support multiple generation approaches.

    Features:
    - Unified API across all strategies
    - Runtime strategy selection
    - Automatic node type classification
    - Comprehensive validation
    - Performance monitoring
    - Intelligence integration

    Thread Safety:
    - Service instance is thread-safe
    - Strategy instances should be stateless
    """

    def __init__(
        self,
        templates_directory: Optional[Path] = None,
        archon_mcp_url: Optional[str] = None,
        enable_intelligence: bool = True,
        enable_mixin_validation: bool = True,
        enable_type_checking: bool = False,
    ):
        """
        Initialize Code Generation Service.

        Args:
            templates_directory: Path to Jinja2 templates directory
            archon_mcp_url: Archon MCP endpoint URL (e.g., http://archon:8060)
            enable_intelligence: Enable RAG intelligence gathering
            enable_mixin_validation: Enable mixin validation in generated code
            enable_type_checking: Enable mypy type checking (slower, ~2-3s overhead)
        """
        self.templates_directory = templates_directory
        self.archon_mcp_url = archon_mcp_url
        self.enable_intelligence = enable_intelligence
        self.enable_mixin_validation = enable_mixin_validation
        self.enable_type_checking = enable_type_checking

        # Initialize components
        self.node_classifier = NodeClassifier()
        self.strategy_registry = StrategyRegistry()

        # Wave 2 components for mixin-enhanced generation
        self.yaml_contract_parser = YAMLContractParser()
        self.mixin_injector = MixinInjector()
        self.node_validator = NodeValidator(
            enable_type_checking=enable_type_checking,
            enable_security_scan=True,  # Always enable security scanning
        )

        # Register strategies (lazy initialization - strategies will be imported on-demand)
        self._strategies_initialized = False

    def _initialize_strategies(self) -> None:
        """
        Initialize and register all available strategies.

        Lazy initialization to avoid circular imports and improve startup time.
        """
        if self._strategies_initialized:
            return

        logger.info("Initializing code generation strategies")

        # Register Jinja2Strategy (template-based generation)
        try:
            from .strategies.jinja2_strategy import Jinja2Strategy

            jinja2_strategy = Jinja2Strategy(
                templates_directory=self.templates_directory,
                enable_inline_templates=True,
                enable_validation=True,
            )
            self.strategy_registry.register(jinja2_strategy, is_default=True)
            logger.info("Registered Jinja2Strategy as default")
        except ImportError as e:
            logger.warning(f"Failed to import Jinja2Strategy: {e}")

        # Register TemplateLoadStrategy (LLM-powered generation)
        try:
            from .strategies.template_load_strategy import TemplateLoadStrategy

            template_load_strategy = TemplateLoadStrategy(
                template_directory=self.templates_directory,
                enable_llm_enhancement=True,
                enable_validation=True,
            )
            self.strategy_registry.register(template_load_strategy)
            logger.info("Registered TemplateLoadStrategy")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Failed to register TemplateLoadStrategy: {e}")

        # Register HybridStrategy (Jinja2 + LLM + Validation)
        try:
            from .strategies.hybrid_strategy import HybridStrategy

            hybrid_strategy = HybridStrategy(
                templates_directory=self.templates_directory,
                enable_llm_enhancement=True,
                enable_strict_validation=True,
                enable_validation=True,
            )
            self.strategy_registry.register(hybrid_strategy)
            logger.info("Registered HybridStrategy")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Failed to register HybridStrategy: {e}")

        self._strategies_initialized = True
        logger.info(
            f"Code generation strategies initialized "
            f"({len(self.strategy_registry._strategies)} strategies registered)"
        )

    async def generate_node(
        self,
        requirements: ModelPRDRequirements,
        classification: Optional[ModelClassificationResult] = None,
        output_directory: Optional[Path] = None,
        strategy: str = "auto",
        enable_llm: bool = True,
        enable_mixins: bool = True,
        validation_level: str = "standard",
        run_tests: bool = False,
        strict_mode: bool = False,
        correlation_id: Optional[UUID] = None,
        contract_path: Optional[Path] = None,
    ) -> ModelGenerationResult:
        """
        Generate ONEX node code with optional mixin support.

        Main entry point for code generation. Automatically selects optimal
        strategy based on requirements and configuration.

        Args:
            requirements: Extracted PRD requirements
            classification: Node type classification (optional, will classify if not provided)
            output_directory: Target directory (optional, will use temp dir if not provided)
            strategy: Generation strategy ("auto", "jinja2", "template_loading", "hybrid")
            enable_llm: Enable LLM-powered business logic generation
            enable_mixins: Enable mixin-enhanced generation (Wave 2 feature)
            validation_level: Validation strictness ("none", "basic", "standard", "strict")
            run_tests: Execute generated tests after code generation
            strict_mode: Raise exception if tests fail (vs. attach to artifacts)
            correlation_id: Correlation ID for tracing (optional)
            contract_path: Path to YAML contract file (optional, for mixin parsing)

        Returns:
            ModelGenerationResult with generated artifacts and metadata

        Raises:
            ValueError: If requirements invalid or validation fails
            RuntimeError: If no suitable strategy found or generation fails

        Example:
            >>> service = CodeGenerationService()
            >>> requirements = ModelPRDRequirements(
            ...     node_type="effect",
            ...     service_name="postgres_crud",
            ...     domain="database",
            ...     business_description="PostgreSQL CRUD operations",
            ...     operations=["create", "read", "update", "delete"],
            ... )
            >>> result = await service.generate_node(
            ...     requirements=requirements,
            ...     strategy="auto",
            ...     enable_llm=True,
            ...     validation_level="strict",
            ... )
            >>> assert result.artifacts.node_name == "NodePostgresCrudEffect"
        """
        # Initialize strategies if not done
        self._initialize_strategies()

        # Generate correlation ID if not provided
        correlation_id = correlation_id or uuid4()

        logger.info(
            f"Starting node generation for {requirements.service_name}",
            extra={
                "service_name": requirements.service_name,
                "node_type": requirements.node_type,
                "strategy": strategy,
                "correlation_id": str(correlation_id),
            },
        )

        start_time = time.perf_counter()

        try:
            # Step 1: Classify node type if not provided
            if classification is None:
                classification = await self._classify_node_type(requirements)

            # Step 2: Set output directory (use temp if not provided)
            if output_directory is None:
                output_directory = Path(f"/tmp/codegen/{requirements.service_name}")

            # Step 3: Build generation request
            strategy_enum = self._parse_strategy_type(strategy)
            validation_enum = self._parse_validation_level(validation_level)

            request = ModelGenerationRequest(
                requirements=requirements,
                classification=classification,
                output_directory=output_directory,
                strategy=strategy_enum,
                enable_llm=enable_llm,
                validation_level=validation_enum,
                run_tests=run_tests,
                strict_mode=strict_mode,
                correlation_id=correlation_id,
            )

            # Step 4: Validate requirements
            is_valid, errors = self._validate_request(request)
            if not is_valid:
                raise ValueError(f"Requirements validation failed: {', '.join(errors)}")

            # Step 5: Parse contract for mixins (if enabled and contract provided)
            parsed_contract: Optional[ModelEnhancedContract] = None
            if enable_mixins and contract_path and contract_path.exists():
                logger.debug(f"Parsing contract from {contract_path}")
                parsed_contract = self.yaml_contract_parser.parse_contract_file(
                    str(contract_path)
                )

                if not parsed_contract.is_valid:
                    raise ValueError(
                        f"Contract validation failed: {parsed_contract.validation_errors}"
                    )

                logger.info(
                    f"Contract parsed successfully with {len(parsed_contract.mixins)} mixins",
                    extra={
                        "contract_name": parsed_contract.name,
                        "mixins": [m.name for m in parsed_contract.mixins],
                        "correlation_id": str(correlation_id),
                    },
                )

            # Step 6: Select strategy
            selected_strategy = self._select_strategy(request)

            # Step 7: Generate code
            result = await selected_strategy.generate(request)

            # Step 7.5: Apply mixin enhancement if enabled
            if (
                enable_mixins
                and parsed_contract
                and len(parsed_contract.get_enabled_mixins()) > 0
            ):
                logger.info(
                    f"Applying mixin enhancement with {len(parsed_contract.mixins)} mixins"
                )

                # Convert parsed contract to dict for MixinInjector
                contract_dict = asdict(parsed_contract)

                # Generate mixin-enhanced node code
                mixin_enhanced_code = self.mixin_injector.generate_node_file(
                    contract_dict
                )

                # Update artifacts with mixin-enhanced code
                result.artifacts.node_file = mixin_enhanced_code

                logger.debug(
                    f"Mixin enhancement complete: {len(mixin_enhanced_code)} chars"
                )

            # Step 7.75: Validate generated code with NodeValidator
            validation_results: list[ModelValidationResult] = []
            if (
                self.enable_mixin_validation
                and enable_mixins
                and validation_level != "none"
            ):
                logger.debug("Running NodeValidator on generated code")

                validation_results = await self.node_validator.validate_generated_node(
                    node_file_content=result.artifacts.node_file,
                    contract=parsed_contract if parsed_contract else None,
                )

                # Check for validation failures
                failed_stages = [r for r in validation_results if not r.passed]

                if failed_stages:
                    logger.warning(
                        f"Validation failed for {len(failed_stages)} stages",
                        extra={
                            "failed_stages": [r.stage.value for r in failed_stages],
                            "correlation_id": str(correlation_id),
                        },
                    )

                    # In strict mode, raise exception on validation failure
                    if validation_level == "strict":
                        error_details = "\n".join(
                            [
                                f"{r.stage.value}: {', '.join(r.errors)}"
                                for r in failed_stages
                            ]
                        )
                        raise RuntimeError(
                            f"Code validation failed in strict mode:\n{error_details}"
                        )
                else:
                    logger.info("All validation stages passed")

            # Step 8: Log metrics
            end_time = time.perf_counter()
            generation_time_ms = (end_time - start_time) * 1000

            # Collect mixin metrics
            mixin_metrics = {}
            if parsed_contract:
                mixin_metrics = {
                    "mixins_applied": len(parsed_contract.mixins),
                    "mixin_names": [m.name for m in parsed_contract.mixins],
                    "has_advanced_features": parsed_contract.advanced_features
                    is not None,
                }

            # Collect validation metrics
            validation_metrics = {}
            if validation_results:
                validation_metrics = {
                    "validation_stages_run": len(validation_results),
                    "validation_stages_passed": sum(
                        1 for r in validation_results if r.passed
                    ),
                    "validation_time_ms": sum(
                        r.execution_time_ms for r in validation_results
                    ),
                    "validation_errors": sum(len(r.errors) for r in validation_results),
                    "validation_warnings": sum(
                        len(r.warnings) for r in validation_results
                    ),
                }

            logger.info(
                f"Node generation complete: {result.artifacts.node_name}",
                extra={
                    "node_name": result.artifacts.node_name,
                    "strategy": result.strategy_used.value,
                    "generation_time_ms": generation_time_ms,
                    "validation_passed": result.validation_passed,
                    "correlation_id": str(correlation_id),
                    **mixin_metrics,
                    **validation_metrics,
                },
            )

            # Step 8: Distribute files to target locations
            if output_directory:
                self._distribute_artifacts(
                    artifacts=result.artifacts,
                    base_directory=output_directory,
                    service_name=requirements.service_name,
                )

            return result

        except Exception as e:
            end_time = time.perf_counter()
            generation_time_ms = (end_time - start_time) * 1000

            logger.error(
                f"Node generation failed: {e}",
                extra={
                    "service_name": requirements.service_name,
                    "error": str(e),
                    "generation_time_ms": generation_time_ms,
                    "correlation_id": str(correlation_id),
                },
                exc_info=True,
            )
            raise

    async def _classify_node_type(
        self,
        requirements: ModelPRDRequirements,
    ) -> ModelClassificationResult:
        """
        Classify node type from requirements.

        Args:
            requirements: PRD requirements

        Returns:
            Classification result
        """
        logger.debug(f"Classifying node type for {requirements.service_name}")
        classification = self.node_classifier.classify(requirements)
        logger.info(
            f"Node classified as {classification.node_type.value} "
            f"(confidence: {classification.confidence:.2f})"
        )
        return classification

    def _distribute_artifacts(
        self,
        artifacts: ModelGeneratedArtifacts,
        base_directory: Path,
        service_name: str,
    ) -> None:
        """
        Distribute generated artifacts to multiple target directories.

        Writes artifacts to:
        1. Base directory: generated_nodes/{service_name}/
        2. Final directory: generated_nodes/{service_name}_final/
        3. LLM directory: generated_nodes/{service_name}_llm/

        Args:
            artifacts: Generated code artifacts
            base_directory: Base output directory (e.g., generated_nodes/vault_secrets_effect)
            service_name: Service name for naming directories

        Example:
            >>> self._distribute_artifacts(
            ...     artifacts=result.artifacts,
            ...     base_directory=Path("generated_nodes/vault_secrets_effect"),
            ...     service_name="vault_secrets_effect",
            ... )
            # Writes to:
            # - generated_nodes/vault_secrets_effect/node.py
            # - generated_nodes/vault_secrets_effect_final/node.py
            # - generated_nodes/vault_secrets_effect_llm/node.py
        """
        # Get all files from artifacts
        all_files = artifacts.get_all_files()

        # Define target directories
        targets = [
            base_directory,  # Base: generated_nodes/vault_secrets_effect/
            base_directory.parent / f"{service_name}_final",  # _final version
            base_directory.parent / f"{service_name}_llm",  # _llm version
        ]

        # Write to each target directory
        for target_dir in targets:
            target_dir.mkdir(parents=True, exist_ok=True)

            for filename, content in all_files.items():
                file_path = target_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)

            logger.info(f"Distributed {len(all_files)} files to {target_dir}")

        logger.info(
            f"File distribution complete: wrote to {len(targets)} directories "
            f"({len(all_files)} files each)"
        )

    def _parse_strategy_type(self, strategy: str) -> EnumStrategyType:
        """Parse strategy string to enum."""
        try:
            return EnumStrategyType(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options: {', '.join(s.value for s in EnumStrategyType)}"
            )

    def _parse_validation_level(self, level: str) -> EnumValidationLevel:
        """Parse validation level string to enum."""
        try:
            return EnumValidationLevel(level.lower())
        except ValueError:
            raise ValueError(
                f"Invalid validation_level '{level}'. "
                f"Valid options: {', '.join(v.value for v in EnumValidationLevel)}"
            )

    def _validate_request(
        self,
        request: ModelGenerationRequest,
    ) -> tuple[bool, list[str]]:
        """
        Validate generation request.

        Args:
            request: Generation request

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate requirements
        if not request.requirements.service_name:
            errors.append("service_name is required")

        if not request.requirements.node_type:
            errors.append("node_type is required")

        # Validate output directory
        if not request.output_directory:
            errors.append("output_directory is required")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _select_strategy(
        self,
        request: ModelGenerationRequest,
    ) -> BaseGenerationStrategy:
        """
        Select optimal generation strategy.

        Args:
            request: Generation request

        Returns:
            Selected strategy instance

        Raises:
            RuntimeError: If no suitable strategy found
        """
        # If AUTO strategy, let registry select
        prefer_strategy = (
            None if request.strategy == EnumStrategyType.AUTO else request.strategy
        )

        return self.strategy_registry.select_strategy(
            node_type=request.classification.node_type,
            enable_llm=request.enable_llm,
            prefer_strategy=prefer_strategy,
        )

    def list_strategies(self) -> list[dict[str, str]]:
        """
        List all registered strategies.

        Returns:
            List of strategy info dictionaries

        Example:
            >>> service = CodeGenerationService()
            >>> strategies = service.list_strategies()
            >>> for strategy in strategies:
            ...     print(f"{strategy['name']} ({strategy['type']})")
        """
        self._initialize_strategies()
        return self.strategy_registry.list_strategies()

    def get_strategy_info(self, strategy_type: str) -> dict:
        """
        Get detailed information about a strategy.

        Args:
            strategy_type: Strategy type to query

        Returns:
            Strategy information dictionary

        Raises:
            ValueError: If strategy not found
        """
        self._initialize_strategies()

        strategy_enum = self._parse_strategy_type(strategy_type)
        strategy = self.strategy_registry.get_strategy(strategy_enum)

        if not strategy:
            raise ValueError(f"Strategy '{strategy_type}' not found")

        return strategy.get_strategy_info()


# Export
__all__ = [
    "CodeGenerationService",
    "StrategyRegistry",
]
