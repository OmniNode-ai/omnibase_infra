#!/usr/bin/env python3
"""
Configuration for Business Logic Generation.

Defines constants, defaults, and configuration for LLM-based business logic generation.
"""

from dataclasses import dataclass
from typing import ClassVar

from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import EnumLLMTier


@dataclass(frozen=True)
class BusinessLogicConfig:
    """
    Configuration for BusinessLogicGenerator.

    All values are immutable after initialization.
    """

    # === LLM CONFIGURATION ===

    DEFAULT_LLM_TIER: ClassVar[EnumLLMTier] = EnumLLMTier.CLOUD_FAST
    """Default LLM tier for code generation"""

    DEFAULT_TEMPERATURE: ClassVar[float] = 0.3
    """Temperature for code generation (lower = more deterministic)"""

    DEFAULT_MAX_TOKENS: ClassVar[int] = 2000
    """Maximum tokens for method implementation"""

    DEFAULT_TOP_P: ClassVar[float] = 0.95
    """Top-p sampling for generation"""

    # === SYSTEM PROMPTS ===

    SYSTEM_PROMPT: ClassVar[str] = (
        "You are an expert Python developer specializing in ONEX v2.0 node implementations. "
        "Generate production-ready business logic that follows ONEX patterns, includes proper "
        "error handling, type hints, and docstrings. Return ONLY the method body code."
    )
    """System prompt for LLM code generation"""

    # === STUB DETECTION PATTERNS ===

    STUB_INDICATORS: ClassVar[tuple[str, ...]] = (
        "# IMPLEMENTATION REQUIRED",
        "# TODO:",
        "pass  # Stub",
        "raise NotImplementedError",
    )
    """Patterns that indicate a method stub needs implementation"""

    # === VALIDATION PATTERNS ===

    ONEX_PATTERNS: ClassVar[tuple[str, ...]] = (
        "ModelOnexError",
        "emit_log_event",
        "correlation_id",
    )
    """Required patterns for ONEX compliance"""

    SECURITY_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r'password\s*=\s*["\']', "Hardcoded password detected"),
        (r'api_key\s*=\s*["\']', "Hardcoded API key detected"),
        (r'secret\s*=\s*["\']', "Hardcoded secret detected"),
        (r'token\s*=\s*["\']', "Hardcoded token detected"),
    ]
    """Security patterns to check for in generated code"""

    # === RETRY CONFIGURATION ===

    MAX_RETRY_ATTEMPTS: ClassVar[int] = 3
    """Maximum retry attempts for LLM API calls"""

    RETRY_BACKOFF_SECONDS: ClassVar[float] = 1.0
    """Initial backoff for retries"""

    # === METRICS COLLECTION ===

    ENABLE_METRICS: ClassVar[bool] = True
    """Whether to collect and store metrics"""

    # === QUALITY THRESHOLDS ===

    MIN_QUALITY_SCORE: ClassVar[float] = 0.7
    """Minimum quality score for generated code"""

    MIN_SUCCESS_RATE: ClassVar[float] = 0.8
    """Minimum success rate for generation session"""


__all__ = ["BusinessLogicConfig"]
