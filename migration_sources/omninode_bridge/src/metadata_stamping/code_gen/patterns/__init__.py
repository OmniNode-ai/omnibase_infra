"""
Production pattern library for code generation.

This package provides a curated library of production patterns extracted from
real ONEX nodes, enabling intelligent pattern matching and code generation.

Modules:
    models: Pydantic models for pattern metadata
    pattern_loader: Pattern loading from YAML files
    pattern_applicator: Pattern application to templates
    pattern_registry: Central registry for pattern management
    pattern_matcher: Pattern matching algorithms
    pattern_validator: Pattern validation rules

Categories:
    resilience: Circuit breakers, retries, timeouts, bulkheads
    observability: Metrics, logging, tracing, health checks
    security: Validation, sanitization, authentication, authorization
    performance: Caching, pooling, batching, streaming
    integration: Kafka, API clients, databases, message queues
"""

__version__ = "0.1.0"

from .models import (
    EnumNodeType,
    EnumPatternCategory,
    ModelPatternExample,
    ModelPatternLibraryStats,
    ModelPatternMatch,
    ModelPatternMetadata,
    ModelPatternQuery,
)
from .pattern_applicator import PatternApplicator
from .pattern_loader import PatternLoader
from .pattern_matcher import PatternMatcher
from .pattern_registry import PatternRegistry
from .pattern_validator import PatternValidator

__all__ = [
    # Enums
    "EnumNodeType",
    "EnumPatternCategory",
    # Models
    "ModelPatternExample",
    "ModelPatternLibraryStats",
    "ModelPatternMatch",
    "ModelPatternMetadata",
    "ModelPatternQuery",
    # Classes
    "PatternApplicator",
    "PatternLoader",
    "PatternMatcher",
    "PatternRegistry",
    "PatternValidator",
]
