"""
Business Logic Generation for ONEX Nodes.

LLM-powered business logic generation that replaces template stubs with
context-aware, intelligent implementations.

Components:
- BusinessLogicGenerator: Orchestrates LLM-based code generation
- Models: Pydantic models for generation context and results
- Config: Configuration constants and defaults

Example:
    >>> from omninode_bridge.codegen.business_logic import BusinessLogicGenerator
    >>> generator = BusinessLogicGenerator(enable_llm=True)
    >>> enhanced = await generator.enhance_artifacts(
    ...     artifacts=artifacts,
    ...     requirements=requirements,
    ...     context_data={"patterns": ["..."]}
    ... )
"""

from .config import BusinessLogicConfig
from .generator import BusinessLogicGenerator
from .models import (
    GenerationContext,
    ModelBusinessLogicContext,
    ModelEnhancedArtifacts,
    ModelGeneratedMethod,
    ModelMethodStub,
    PromptPair,
    StubInfo,
)

__all__ = [
    "BusinessLogicGenerator",
    "BusinessLogicConfig",
    "ModelMethodStub",
    "ModelBusinessLogicContext",
    "ModelGeneratedMethod",
    "ModelEnhancedArtifacts",
    "GenerationContext",
    "StubInfo",
    "PromptPair",
]
