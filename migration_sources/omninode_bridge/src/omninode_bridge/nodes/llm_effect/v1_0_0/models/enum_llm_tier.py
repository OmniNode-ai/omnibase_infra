#!/usr/bin/env python3
"""
LLM Tier Enum - ONEX v2.0 Compliant.

Defines available LLM inference tiers for multi-tier LLM system.
"""

from enum import Enum


class EnumLLMTier(str, Enum):
    """
    LLM inference tier selection.

    Tier Configuration:
    - LOCAL: Ollama/vLLM (future-ready, not implemented in Phase 1)
    - CLOUD_FAST: GLM-4.5 via Z.ai (baseline tier, Phase 1 implementation)
    - CLOUD_PREMIUM: GLM-4.6 via Z.ai (future-ready, not implemented in Phase 1)

    Selection Criteria:
    - LOCAL: Low-cost, high-latency, private data (future)
    - CLOUD_FAST: Balanced cost/performance, 128K context (PRIMARY)
    - CLOUD_PREMIUM: Highest quality, advanced reasoning (future)
    """

    LOCAL = "LOCAL"
    CLOUD_FAST = "CLOUD_FAST"
    CLOUD_PREMIUM = "CLOUD_PREMIUM"


__all__ = ["EnumLLMTier"]
