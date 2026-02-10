# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM Operation Type Enumeration.

Defines the canonical operation types for LLM API calls. Used to classify
inference requests by their operation category for routing, metrics,
and cost tracking.

Operation types:
    - CHAT_COMPLETION: Multi-turn chat with messages array
    - COMPLETION: Single-turn text completion (legacy)
    - EMBEDDING: Vector embedding generation
"""

from enum import Enum


class EnumLlmOperationType(str, Enum):
    """Canonical LLM operation types for inference classification.

    Used to route requests to appropriate handlers and track
    usage metrics per operation category.

    Attributes:
        CHAT_COMPLETION: Chat-style multi-turn completion (/v1/chat/completions)
        COMPLETION: Legacy single-turn text completion (/v1/completions)
        EMBEDDING: Vector embedding generation (/v1/embeddings)
    """

    CHAT_COMPLETION = "chat_completion"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


__all__: list[str] = ["EnumLlmOperationType"]
