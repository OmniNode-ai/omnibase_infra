#!/usr/bin/env python3
"""
Code Generation Pipeline - Unified Template Loading + LLM Enhancement.

Orchestrates the complete code generation workflow:
1. Load pre-written templates using TemplateEngineLoader
2. Convert to BusinessLogicGenerator format
3. Enhance stubs with LLM-generated implementations
4. Return complete, enhanced node

ONEX v2.0 Compliance:
- Async/await throughout
- Structured error handling with ModelOnexError
- Comprehensive logging
- Type-safe operations

Example Usage:
    >>> from pathlib import Path
    >>> import os
    >>>
    >>> # Initialize pipeline
    >>> pipeline = CodeGenerationPipeline(
    ...     template_dir=Path("templates/node_templates"),
    ...     enable_llm=True,
    ...     llm_api_key=os.getenv("ZAI_API_KEY")
    ... )
    >>>
    >>> # Generate enhanced node
    >>> result = await pipeline.generate_node(
    ...     node_type="effect",
    ...     version="v1_0_0",
    ...     requirements={
    ...         "service_name": "postgres_crud",
    ...         "business_description": "PostgreSQL CRUD operations",
    ...         "operations": ["create", "read", "update", "delete"],
    ...         "domain": "database",
    ...         "features": ["Async operations", "Connection pooling"]
    ...     }
    ... )
    >>>
    >>> print(f"Generated {result.node_name}")
    >>> print(f"Enhanced {len(result.methods_generated)} methods")
    >>> print(f"Total cost: ${result.total_cost_usd:.4f}")
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from omnibase_core import ModelOnexError

from .business_logic.generator import BusinessLogicGenerator
from .business_logic.models import ModelEnhancedArtifacts
from .context_builder import EnhancedContextBuilder
from .contracts.subcontract_processor import SubcontractProcessor
from .converters import ArtifactConverter
from .mixins.mixin_recommender import MixinRecommender
from .mixins.requirements_analyzer import RequirementsAnalyzer
from .pattern_library import ProductionPatternLibrary
from .prd_analyzer import ModelPRDRequirements
from .template_engine_loader.engine import TemplateEngine
from .template_selector import TemplateSelector

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when pipeline operations fail."""

    pass


class CodeGenerationPipeline:
    """
    Orchestrates template loading and LLM enhancement pipeline.

    Integrates TemplateEngineLoader and BusinessLogicGenerator to provide
    a unified code generation workflow.

    Features:
    - Template discovery and loading
    - Automatic stub detection
    - LLM-powered implementation generation
    - Validation and error handling
    - Optional LLM enhancement (can disable for template-only)
    """

    template_engine: TemplateEngine
    business_logic_generator: Optional[BusinessLogicGenerator]
    enable_llm: bool
    enable_validation: bool

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        enable_llm: bool = False,
        llm_api_key: Optional[str] = None,
        enable_validation: bool = True,
    ):
        """
        Initialize code generation pipeline.

        Args:
            template_dir: Directory containing node templates (uses default if None)
            enable_llm: Enable LLM enhancement of stubs (requires ZAI_API_KEY)
            llm_api_key: ZAI API key for LLM (optional, reads from env if not provided)
            enable_validation: Enable template validation

        Raises:
            PipelineError: If LLM enabled but API key not provided/found
        """
        # Initialize TemplateEngine
        self.template_engine = TemplateEngine(
            template_root=template_dir,
            enable_validation=enable_validation,
        )

        self.enable_llm = enable_llm
        self.enable_validation = enable_validation

        # Initialize BusinessLogicGenerator if LLM enabled
        if enable_llm:
            # Set API key in environment if provided
            if llm_api_key:
                os.environ["ZAI_API_KEY"] = llm_api_key
            elif not os.getenv("ZAI_API_KEY"):
                raise PipelineError(
                    "LLM enabled but ZAI_API_KEY not provided and not in environment. "
                    "Either pass llm_api_key parameter or set ZAI_API_KEY environment variable."
                )

            try:
                self.business_logic_generator = BusinessLogicGenerator(enable_llm=True)
            except ModelOnexError as e:
                raise PipelineError(
                    f"Failed to initialize BusinessLogicGenerator: {e.message}"
                ) from e
        else:
            self.business_logic_generator = None

        logger.info(
            f"CodeGenerationPipeline initialized "
            f"(template_dir={template_dir or 'default'}, "
            f"enable_llm={enable_llm}, "
            f"enable_validation={enable_validation})"
        )

    async def generate_node(
        self,
        node_type: str,
        version: str,
        requirements: dict[str, Any],
        template_name: str = "node",
        enable_llm: Optional[bool] = None,
        context_data: Optional[dict[str, Any]] = None,
    ) -> ModelEnhancedArtifacts:
        """
        Generate complete node implementation with optional LLM enhancement.

        Complete pipeline execution:
        1. Load template from filesystem
        2. Detect stubs in template
        3. Convert to BusinessLogicGenerator format
        4. Enhance stubs with LLM (if enabled)
        5. Validate enhanced code
        6. Return complete implementation

        Args:
            node_type: Node type (effect/compute/reducer/orchestrator)
            version: Template version (e.g., "v1_0_0")
            requirements: Requirements dict with:
                - service_name (str, required): Service name in snake_case
                - business_description (str, required): What this node does
                - operations (list[str], required): Operations to implement
                - domain (str, required): Domain (database/api/ml/etc)
                - features (list[str], optional): Key features
                - performance_requirements (dict, optional): Performance specs
                - dependencies (dict, optional): External dependencies
                - data_models (list, optional): Data models to generate
                - best_practices (list[str], optional): ONEX best practices
                - code_examples (list[str], optional): Example code patterns
            template_name: Template file name (default: "node")
            enable_llm: Override instance LLM setting for this call
            context_data: Additional context for LLM generation:
                - patterns (list[str]): Similar code patterns from RAG
                - best_practices (list[str]): ONEX best practices

        Returns:
            ModelEnhancedArtifacts with:
                - original_artifacts: Original template artifacts
                - enhanced_node_file: Node file with LLM implementations
                - methods_generated: List of generated methods with metrics
                - total_tokens_used, total_cost_usd, total_latency_ms
                - generation_success_rate

        Raises:
            PipelineError: If template loading fails
            ModelOnexError: If LLM generation or validation fails

        Example:
            >>> pipeline = CodeGenerationPipeline(enable_llm=True)
            >>> result = await pipeline.generate_node(
            ...     node_type="effect",
            ...     version="v1_0_0",
            ...     requirements={
            ...         "service_name": "postgres_crud",
            ...         "business_description": "PostgreSQL CRUD operations",
            ...         "operations": ["create", "read", "update", "delete"],
            ...         "domain": "database",
            ...         "features": ["Connection pooling", "Async operations"]
            ...     },
            ...     context_data={
            ...         "patterns": ["Use asyncpg for PostgreSQL", "Pool connections"],
            ...         "best_practices": ["Handle connection errors", "Log all queries"]
            ...     }
            ... )
            >>> print(f"Generated {result.node_name}: ${result.total_cost_usd:.4f}")
        """
        logger.info(
            f"Starting code generation pipeline: {node_type}/{version} "
            f"(service={requirements.get('service_name', 'unknown')}, "
            f"llm={'enabled' if enable_llm else 'disabled'})"
        )

        # Validate requirements
        self._validate_requirements(requirements)

        # Step 1: Load template
        try:
            template_artifacts = self.template_engine.load_template(
                node_type=node_type,
                version=version,
                template_name=template_name,
            )

            logger.info(
                f"Loaded template: {template_artifacts.template_path} "
                f"({len(template_artifacts.stubs)} stubs detected)"
            )

        except Exception as e:
            raise PipelineError(
                f"Failed to load template {node_type}/{version}/{template_name}: {e}"
            ) from e

        # Step 2: Convert to ModelPRDRequirements format
        prd_requirements = self._build_prd_requirements(requirements, node_type)

        # Step 3: Convert template artifacts to generated artifacts format
        service_name = requirements["service_name"]
        node_class_name = self._build_node_class_name(service_name, node_type)

        generated_artifacts = ArtifactConverter.template_to_generated(
            template_artifacts=template_artifacts,
            service_name=service_name,
            node_class_name=node_class_name,
        )

        logger.info("Converted template artifacts to generated format")

        # Step 4: Enhance with LLM if enabled
        use_llm = enable_llm if enable_llm is not None else self.enable_llm

        if use_llm:
            if not self.business_logic_generator:
                raise PipelineError(
                    "LLM enhancement requested but BusinessLogicGenerator not initialized. "
                    "Initialize pipeline with enable_llm=True."
                )

            logger.info("Enhancing artifacts with LLM-generated implementations")

            try:
                enhanced = await self.business_logic_generator.enhance_artifacts(
                    artifacts=generated_artifacts,
                    requirements=prd_requirements,
                    context_data=context_data or {},
                )

                logger.info(
                    f"LLM enhancement complete: "
                    f"{len(enhanced.methods_generated)} methods generated, "
                    f"${enhanced.total_cost_usd:.4f} cost, "
                    f"{enhanced.generation_success_rate:.1%} success rate"
                )

                return enhanced

            except ModelOnexError as e:
                logger.error(f"LLM enhancement failed: {e.message}")
                raise
            except Exception as e:
                raise PipelineError(
                    f"Unexpected error during LLM enhancement: {e}"
                ) from e

        else:
            # No LLM enhancement - return template as-is
            logger.info("LLM disabled - returning template without enhancement")

            enhanced = ModelEnhancedArtifacts(
                original_artifacts=generated_artifacts,
                enhanced_node_file=generated_artifacts.node_file,
                methods_generated=[],
                total_tokens_used=0,
                total_cost_usd=0.0,
                total_latency_ms=0.0,
                generation_success_rate=1.0,
            )

            return enhanced

    def discover_templates(self) -> list:
        """
        Discover all available templates.

        Returns:
            List of ModelTemplateInfo for each discovered template

        Example:
            >>> pipeline = CodeGenerationPipeline()
            >>> templates = pipeline.discover_templates()
            >>> for template in templates:
            ...     print(f"{template.node_type}/{template.version}: {template.metadata.description}")
        """
        return self.template_engine.discover_templates()

    def _validate_requirements(self, requirements: dict[str, Any]) -> None:
        """
        Validate requirements dictionary has all required fields.

        Args:
            requirements: Requirements dict to validate

        Raises:
            PipelineError: If required fields missing
        """
        required_fields = [
            "service_name",
            "business_description",
            "operations",
            "domain",
        ]

        missing = [field for field in required_fields if field not in requirements]

        if missing:
            raise PipelineError(
                f"Requirements missing required fields: {missing}. "
                f"Required: {required_fields}"
            )

        # Validate types
        if not isinstance(requirements["operations"], list):
            raise PipelineError(
                f"requirements['operations'] must be a list, got {type(requirements['operations']).__name__}"
            )

        if not requirements["operations"]:
            raise PipelineError("requirements['operations'] cannot be empty")

        logger.debug(f"Requirements validation passed: {required_fields}")

    def _build_prd_requirements(
        self, requirements: dict[str, Any], node_type: str
    ) -> ModelPRDRequirements:
        """
        Build ModelPRDRequirements from requirements dict.

        Args:
            requirements: Requirements dictionary
            node_type: Node type

        Returns:
            ModelPRDRequirements instance
        """
        return ModelPRDRequirements(
            node_type=node_type,
            service_name=requirements["service_name"],
            business_description=requirements["business_description"],
            operations=requirements["operations"],
            domain=requirements["domain"],
            features=requirements.get("features", []),
            dependencies=requirements.get("dependencies", {}),
            performance_requirements=requirements.get("performance_requirements", {}),
            data_models=requirements.get("data_models", []),
            best_practices=requirements.get("best_practices", []),
            code_examples=requirements.get("code_examples", []),
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
            >>> pipeline._build_node_class_name("postgres_crud", "effect")
            "NodePostgresCRUDEffect"
        """
        # Convert snake_case to PascalCase
        pascal_name = "".join(word.capitalize() for word in service_name.split("_"))

        # Build node class name: Node<PascalName><Type>
        node_class_name = f"Node{pascal_name}{node_type.capitalize()}"

        return node_class_name

    async def cleanup(self) -> None:
        """
        Cleanup pipeline resources.

        Should be called when done using pipeline to release LLM resources.
        """
        if self.business_logic_generator:
            await self.business_logic_generator.cleanup()
            logger.info("Pipeline resources cleaned up")


class EnhancedCodeGenerationPipeline:
    """
    Phase 3 Enhanced Pipeline with intelligent code generation.

    Integrates all Phase 3 components:
    - Template variant selection (TemplateSelector)
    - Pattern matching (ProductionPatternLibrary)
    - Mixin recommendations (MixinRecommender)
    - Contract processing (SubcontractProcessor)
    - Enhanced LLM context (EnhancedContextBuilder)

    Features:
    - Intelligent template selection based on requirements
    - Pattern-driven code generation
    - Mixin conflict resolution
    - Subcontract processing
    - Comprehensive LLM context building

    Performance Target: <20s total pipeline (including LLM)
    Quality Target: >90% first-pass success rate
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        enable_llm: bool = False,
        llm_api_key: Optional[str] = None,
        enable_validation: bool = True,
    ):
        """
        Initialize enhanced code generation pipeline.

        Args:
            template_dir: Directory containing node templates
            enable_llm: Enable LLM enhancement
            llm_api_key: ZAI API key for LLM (optional)
            enable_validation: Enable template validation
        """
        # Phase 1 + 2 components
        self.template_engine = TemplateEngine(
            template_root=template_dir,
            enable_validation=enable_validation,
        )
        self.enable_llm = enable_llm
        self.enable_validation = enable_validation

        # Phase 3 components
        self.template_selector = TemplateSelector(template_root=template_dir)
        self.pattern_library = ProductionPatternLibrary()
        self.context_builder = EnhancedContextBuilder()
        self.requirements_analyzer = RequirementsAnalyzer()
        self.mixin_recommender = MixinRecommender()
        self.subcontract_processor = SubcontractProcessor(template_dir=template_dir)

        # Business logic generator (if LLM enabled)
        if enable_llm:
            if llm_api_key:
                os.environ["ZAI_API_KEY"] = llm_api_key
            elif not os.getenv("ZAI_API_KEY"):
                raise PipelineError(
                    "LLM enabled but ZAI_API_KEY not provided. "
                    "Either pass llm_api_key or set ZAI_API_KEY environment variable."
                )

            try:
                self.business_logic_generator = BusinessLogicGenerator(enable_llm=True)
            except ModelOnexError as e:
                raise PipelineError(
                    f"Failed to initialize BusinessLogicGenerator: {e.message}"
                ) from e
        else:
            self.business_logic_generator = None

        logger.info(
            f"EnhancedCodeGenerationPipeline initialized "
            f"(enable_llm={enable_llm}, enable_validation={enable_validation})"
        )

    async def generate_node(
        self,
        contract_path: Path,
        output_dir: Path,
        target_environment: Optional[str] = None,
    ) -> ModelEnhancedArtifacts:
        """
        Execute complete Phase 1-3 pipeline with intelligent generation.

        Pipeline stages:
        1. Parse contract
        2. Select template variant (Phase 3)
        3. Analyze requirements and recommend mixins (Phase 3)
        4. Match patterns (Phase 3)
        5. Process subcontracts (Phase 3)
        6. Build LLM context (Phase 3)
        7. Render template with patterns + mixins
        8. Enhance with LLM (if enabled)
        9. Validate generated code
        10. Write artifacts

        Args:
            contract_path: Path to contract YAML
            output_dir: Output directory for generated files
            target_environment: Target environment (development/staging/production)

        Returns:
            ModelEnhancedArtifacts with generation results

        Raises:
            PipelineError: If pipeline fails
        """
        import time

        start_time = time.perf_counter()

        logger.info(
            f"Starting enhanced pipeline: {contract_path} → {output_dir} "
            f"(env={target_environment or 'default'})"
        )

        try:
            # Stage 1: Parse contract
            contract = await self._parse_contract(contract_path)
            logger.info(f"Contract parsed: {contract.metadata.name}")

            # Stage 2: Template selection (Phase 3)
            template_selection = self.template_selector.select_template(
                requirements=contract,
                node_type=contract.node_type,
                target_environment=target_environment,
            )
            logger.info(
                f"Template selected: {template_selection.variant.value} "
                f"(confidence: {template_selection.confidence:.2f})"
            )

            # Stage 3: Requirements analysis + mixin selection (Phase 3)
            requirement_analysis = self.requirements_analyzer.analyze(contract)
            mixin_recommendations = self.mixin_recommender.recommend_mixins(
                requirement_analysis, top_k=5
            )
            logger.info(
                f"Mixins recommended: {len(mixin_recommendations)} "
                f"(top: {mixin_recommendations[0].mixin_name if mixin_recommendations else 'none'})"
            )

            # Stage 4: Pattern matching (Phase 3)
            matched_patterns = self.pattern_library.find_matching_patterns(
                operation_type=contract.domain or "",
                features=set(template_selection.patterns),
                node_type=contract.node_type,
            )
            logger.info(f"Patterns matched: {len(matched_patterns)}")

            # Stage 5: Subcontract processing (Phase 3)
            subcontract_results = None
            if contract.subcontracts:
                subcontract_results = (
                    await self.subcontract_processor.process_subcontracts(contract)
                )
                if subcontract_results.has_errors:
                    logger.warning(
                        f"Subcontract processing had errors: {subcontract_results.errors}"
                    )
                else:
                    logger.info(
                        f"Subcontracts processed: {len(subcontract_results.processed_subcontracts)}"
                    )

            # Stage 6: Build LLM contexts (Phase 3)
            contexts = []
            if contract.io_operations:
                for operation in contract.io_operations:
                    context = self.context_builder.build_context(
                        requirements=contract,
                        operation=operation,
                        template_selection=template_selection,
                        mixin_selection=mixin_recommendations,
                        pattern_matches=matched_patterns,
                    )
                    contexts.append(context)
                    logger.info(
                        f"Context built for {context.operation_name}: "
                        f"{context.estimated_tokens} tokens"
                    )

            # Stage 7: Template rendering
            base_artifacts = await self._render_template(
                template_selection=template_selection,
                contract=contract,
                mixin_recommendations=mixin_recommendations,
                matched_patterns=matched_patterns,
                subcontract_results=subcontract_results,
            )
            logger.info("Template rendered with patterns and mixins")

            # Stage 8: LLM enhancement (if enabled)
            if self.enable_llm and contexts:
                logger.info("Enhancing with LLM...")
                enhanced_artifacts = (
                    await self.business_logic_generator.enhance_artifacts(
                        artifacts=base_artifacts,
                        requirements=self._build_prd_requirements(contract),
                        context_data={"contexts": [c.to_dict() for c in contexts]},
                    )
                )
                logger.info(
                    f"LLM enhancement complete: {len(enhanced_artifacts.methods_generated)} methods, "
                    f"${enhanced_artifacts.total_cost_usd:.4f}"
                )
            else:
                enhanced_artifacts = ModelEnhancedArtifacts(
                    original_artifacts=base_artifacts,
                    enhanced_node_file=base_artifacts.node_file,
                    methods_generated=[],
                    total_tokens_used=0,
                    total_cost_usd=0.0,
                    total_latency_ms=0.0,
                    generation_success_rate=1.0,
                )

            # Stage 9: Validation (if enabled)
            if self.enable_validation:
                await self._run_quality_gates(enhanced_artifacts, contract)
                logger.info("Quality gates passed")

            # Stage 10: Write artifacts
            await self._write_artifacts(enhanced_artifacts, output_dir)
            logger.info(f"Artifacts written to {output_dir}")

            # Calculate total time
            total_time_s = time.perf_counter() - start_time
            logger.info(
                f"Pipeline complete in {total_time_s:.1f}s "
                f"(variant={template_selection.variant.value}, "
                f"patterns={len(matched_patterns)}, "
                f"mixins={len(mixin_recommendations)})"
            )

            return enhanced_artifacts

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise PipelineError(f"Enhanced pipeline failed: {e}") from e

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _parse_contract(self, contract_path: Path) -> Any:
        """Parse contract YAML file."""
        from .yaml_contract_parser import YamlContractParser

        parser = YamlContractParser()
        return await parser.parse(contract_path)

    def _build_prd_requirements(self, contract: Any) -> ModelPRDRequirements:
        """Build PRD requirements from contract."""
        return ModelPRDRequirements(
            node_type=contract.node_type,
            service_name=contract.metadata.name,
            business_description=contract.metadata.description or "",
            operations=[op.name for op in (contract.io_operations or [])],
            domain=contract.domain or "general",
            features=[],
            dependencies=contract.dependencies or {},
            performance_requirements={},
            data_models=[],
            best_practices=[],
            code_examples=[],
        )

    async def _render_template(
        self,
        template_selection: Any,
        contract: Any,
        mixin_recommendations: list,
        matched_patterns: list,
        subcontract_results: Optional[Any],
    ) -> Any:
        """Render template with Phase 3 enhancements."""
        # This would integrate with template engine
        # For now, use existing template engine
        template_artifacts = self.template_engine.load_template(
            node_type=contract.node_type,
            version="v1_0_0",
            template_name="node",
        )

        service_name = contract.metadata.name
        node_class_name = self._build_node_class_name(service_name, contract.node_type)

        generated_artifacts = ArtifactConverter.template_to_generated(
            template_artifacts=template_artifacts,
            service_name=service_name,
            node_class_name=node_class_name,
        )

        return generated_artifacts

    def _build_node_class_name(self, service_name: str, node_type: str) -> str:
        """Build node class name."""
        pascal_name = "".join(word.capitalize() for word in service_name.split("_"))
        return f"Node{pascal_name}{node_type.capitalize()}"

    async def _run_quality_gates(self, artifacts: Any, contract: Any) -> None:
        """Run quality validation gates."""
        # Placeholder for quality gate validation
        pass

    async def _write_artifacts(
        self, artifacts: ModelEnhancedArtifacts, output_dir: Path
    ) -> None:
        """Write generated artifacts to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write node file
        node_file = output_dir / "node.py"
        node_file.write_text(artifacts.enhanced_node_file)

        logger.info(f"Wrote node file: {node_file}")

    async def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self.business_logic_generator:
            await self.business_logic_generator.cleanup()
            logger.info("Pipeline resources cleaned up")


__all__ = ["CodeGenerationPipeline", "EnhancedCodeGenerationPipeline", "PipelineError"]
