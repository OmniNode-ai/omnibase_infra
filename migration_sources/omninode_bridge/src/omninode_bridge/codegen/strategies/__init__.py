#!/usr/bin/env python3
"""
Code generation strategies for ONEX nodes.

Provides different strategies for generating node implementations:
- Jinja2Strategy: Fast template-based generation
- TemplateLoadStrategy: Load hand-written templates + LLM enhancement
- HybridStrategy: Best quality (Jinja2 + LLM + strict validation)

ONEX v2.0 Compliance:
- Strategy pattern for flexible generation
- Type-safe interfaces
- Comprehensive error handling
"""

from .base import (
    BaseGenerationStrategy,
    EnumStrategyType,
    EnumValidationLevel,
    ModelGenerationRequest,
    ModelGenerationResult,
)
from .hybrid_strategy import HybridStrategy
from .jinja2_strategy import Jinja2Strategy
from .selector import StrategySelector
from .template_load_strategy import TemplateLoadStrategy

__all__ = [
    # Base
    "BaseGenerationStrategy",
    "EnumStrategyType",
    "EnumValidationLevel",
    "ModelGenerationRequest",
    "ModelGenerationResult",
    # Strategies
    "Jinja2Strategy",
    "TemplateLoadStrategy",
    "HybridStrategy",
    # Selector
    "StrategySelector",
]
