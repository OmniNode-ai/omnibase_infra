"""
OmniNode Code Generation Module.

Provides intelligent code generation for ONEX v2.0 nodes with:
- PRD analysis and requirement extraction
- Node type classification and template selection
- Template-based code generation (Jinja2)
- Template loading from pre-written files
- LLM-powered business logic enhancement
- Quality validation and ONEX compliance checking

Main Components:
- PRDAnalyzer: Extract requirements from natural language
- NodeClassifier: Classify node types and select templates
- TemplateEngine: Generate code from Jinja2 templates
- TemplateEngineLoader: Load pre-written Python templates
- BusinessLogicGenerator: Enhance stubs with LLM implementations
- CodeGenerationPipeline: Unified template loading + LLM enhancement
- QualityValidator: Validate generated code quality
- QualityGatePipeline: Multi-stage validation pipeline with quality gates

Usage (Unified Pipeline):
    >>> from omninode_bridge.codegen import CodeGenerationPipeline
    >>> import os
    >>>
    >>> # Initialize pipeline with LLM enhancement
    >>> pipeline = CodeGenerationPipeline(
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
    ...     }
    ... )
    >>> print(f"Generated {result.node_name}: ${result.total_cost_usd:.4f}")

Usage (Traditional Jinja2):
    >>> from omninode_bridge.codegen import (
    ...     PRDAnalyzer,
    ...     NodeClassifier,
    ...     TemplateEngine,
    ...     QualityValidator,
    ... )
    >>>
    >>> # Analyze prompt
    >>> analyzer = PRDAnalyzer()
    >>> requirements = await analyzer.analyze_prompt("Create PostgreSQL CRUD node")
    >>>
    >>> # Classify node type
    >>> classifier = NodeClassifier()
    >>> classification = classifier.classify(requirements)
    >>>
    >>> # Generate code
    >>> engine = TemplateEngine()
    >>> artifacts = await engine.generate(requirements, classification, output_dir)
    >>>
    >>> # Validate quality
    >>> validator = QualityValidator()
    >>> validation = await validator.validate(artifacts)
    >>> assert validation.passed
"""

# Import business logic components
from .business_logic.generator import BusinessLogicGenerator
from .business_logic.models import ModelEnhancedArtifacts, ModelGeneratedMethod
from .contract_inferencer import (
    ContractInferencer,
    ModelMixinConfigInference,
    ModelNodeAnalysis,
)
from .contract_introspector import ContractIntrospector
from .converters import ArtifactConverter
from .failure_analyzer import (
    EnumFailureCategory,
    EnumFailureSeverity,
    FailureAnalyzer,
    ModelFailureAnalysis,
    ModelFailureCause,
)
from .node_classifier import EnumNodeType, ModelClassificationResult, NodeClassifier
from .pipeline import CodeGenerationPipeline, PipelineError
from .prd_analyzer import ModelPRDRequirements, PRDAnalyzer
from .quality_gates import (
    QualityGatePipeline,
    StageResult,
    ValidationLevel,
    ValidationResult,
    ValidationStage,
)
from .quality_validator import ModelValidationResult, QualityValidator

# Import unified service and strategies
from .service import CodeGenerationService, StrategyRegistry
from .strategies import (
    BaseGenerationStrategy,
    EnumStrategyType,
    EnumValidationLevel,
    HybridStrategy,
    Jinja2Strategy,
    ModelGenerationRequest,
    ModelGenerationResult,
    StrategySelector,
    TemplateLoadStrategy,
)
from .template_engine import ModelGeneratedArtifacts, TemplateEngine
from .template_engine_loader.engine import TemplateEngine as TemplateEngineLoader
from .template_engine_loader.models import (
    ModelStubInfo,
    ModelTemplateArtifacts,
    ModelTemplateInfo,
    ModelTemplateMetadata,
)
from .test_executor import ModelTestResults, TestExecutionConfig, TestExecutor

__all__ = [
    # PRD Analysis
    "PRDAnalyzer",
    "ModelPRDRequirements",
    # Node Classification
    "NodeClassifier",
    "EnumNodeType",
    "ModelClassificationResult",
    # Template Engine (Jinja2)
    "TemplateEngine",
    "ModelGeneratedArtifacts",
    # Template Engine Loader (Pre-written)
    "TemplateEngineLoader",
    "ModelTemplateArtifacts",
    "ModelTemplateInfo",
    "ModelTemplateMetadata",
    "ModelStubInfo",
    # Business Logic Generation
    "BusinessLogicGenerator",
    "ModelEnhancedArtifacts",
    "ModelGeneratedMethod",
    # Unified Pipeline
    "CodeGenerationPipeline",
    "PipelineError",
    # Unified Service (New)
    "CodeGenerationService",
    "StrategyRegistry",
    # Strategies (New)
    "BaseGenerationStrategy",
    "Jinja2Strategy",
    "TemplateLoadStrategy",
    "HybridStrategy",
    "StrategySelector",
    "ModelGenerationRequest",
    "ModelGenerationResult",
    "EnumStrategyType",
    "EnumValidationLevel",
    # Converters
    "ArtifactConverter",
    # Contract Introspection
    "ContractIntrospector",
    # Contract Inference (NEW - Automated v2.0 contract generation)
    "ContractInferencer",
    "ModelNodeAnalysis",
    "ModelMixinConfigInference",
    # Quality Validation
    "QualityValidator",
    "ModelValidationResult",
    # Quality Gates Pipeline
    "QualityGatePipeline",
    "ValidationResult",
    "ValidationLevel",
    "ValidationStage",
    "StageResult",
    # Test Execution
    "TestExecutor",
    "ModelTestResults",
    "TestExecutionConfig",
    # Failure Analysis
    "FailureAnalyzer",
    "ModelFailureAnalysis",
    "ModelFailureCause",
    "EnumFailureSeverity",
    "EnumFailureCategory",
]
