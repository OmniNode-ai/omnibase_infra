# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-eval task type enumeration.

Defines the canonical task types for autonomous LLM evaluation runs.
Each member represents a distinct evaluation category that the
ServiceAutoEvalRunner can execute against a configured LLM endpoint.

Related:
    - OMN-6795: Define eval task models and enums
    - ModelAutoEvalTask: Task definition referencing this enum
"""

from enum import Enum


class EnumAutoEvalTaskType(str, Enum):
    """Task types for autonomous LLM evaluation.

    Attributes:
        CODE_GENERATION: Evaluate code generation quality (correctness, style).
        EMBEDDING_QUALITY: Evaluate embedding similarity and retrieval accuracy.
        ROUTING_ACCURACY: Evaluate LLM routing classification correctness.
        REASONING_DEPTH: Evaluate multi-step reasoning and analysis quality.
    """

    CODE_GENERATION = "code_generation"
    EMBEDDING_QUALITY = "embedding_quality"
    ROUTING_ACCURACY = "routing_accuracy"
    REASONING_DEPTH = "reasoning_depth"


__all__: list[str] = ["EnumAutoEvalTaskType"]
