"""
LLM Enhancement Module for Business Logic Generation.

This module provides enhanced LLM context building with Phase 3 intelligence:
- Pattern formatting for LLM consumption
- Context aggregation from multiple sources
- Response parsing and validation
- Fallback mechanisms for reliability

Phase 3 Integration:
- PatternMatcher (C10): Production pattern matching
- VariantSelector (C8): Template variant selection
- MixinRecommender (C12-C15): Mixin recommendations
- Contract Processing (C16-C18): Subcontract handling

Components:
- PatternFormatter: Format patterns for LLM prompts
- EnhancedContextBuilder: Aggregate multi-source context
- EnhancedResponseParser: Extract code from LLM responses
- EnhancedResponseValidator: Validate generated code
- FallbackStrategy: Fallback mechanisms for reliability

Models:
- ModelLLMContext: Complete LLM context structure
- ModelParsedResponse: Parsed LLM response
- ModelValidationResult: Validation results
- FallbackMetrics: Fallback operation metrics
"""

from .context_builder import EnhancedContextBuilder
from .fallback_strategies import FallbackMetrics, FallbackStrategy
from .pattern_formatter import PatternFormatter
from .response_parser import EnhancedResponseParser, ModelParsedResponse
from .response_validator import EnhancedResponseValidator, ModelValidationResult

__all__ = [
    "PatternFormatter",
    "EnhancedContextBuilder",
    "EnhancedResponseParser",
    "ModelParsedResponse",
    "EnhancedResponseValidator",
    "ModelValidationResult",
    "FallbackStrategy",
    "FallbackMetrics",
]
