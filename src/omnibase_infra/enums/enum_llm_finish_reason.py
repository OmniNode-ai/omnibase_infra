# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM Finish Reason Enumeration.

Defines canonical finish reasons for LLM API responses. Used to normalize
provider-specific finish reason strings into a consistent enum across
all LLM integrations (OpenAI, vLLM, Ollama, etc.).

Finish reasons:
    - STOP: Model completed generation naturally
    - LENGTH: Output was truncated due to max_tokens limit
    - ERROR: Model or provider error during generation
    - CONTENT_FILTER: Content was filtered by safety systems
    - TOOL_CALLS: Model requested tool/function calls
    - UNKNOWN: Unrecognized provider-specific reason (fallback)
"""

from enum import Enum


class EnumLlmFinishReason(str, Enum):
    """Canonical LLM finish reasons for response normalization.

    Handlers map provider-specific finish reason strings to these
    canonical values. Unrecognized strings should map to UNKNOWN.

    Attributes:
        STOP: Natural generation completion
        LENGTH: Truncated by max_tokens limit
        ERROR: Generation error
        CONTENT_FILTER: Blocked by content safety filter
        TOOL_CALLS: Model requested tool/function execution
        UNKNOWN: Unrecognized or unmapped finish reason
    """

    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    UNKNOWN = "unknown"


__all__: list[str] = ["EnumLlmFinishReason"]
