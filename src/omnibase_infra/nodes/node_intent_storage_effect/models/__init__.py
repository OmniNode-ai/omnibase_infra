# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Models for Intent Storage Effect Node.

This module exports models used by the NodeIntentStorageEffect for
capability-oriented intent storage operations in Memgraph.

Available Models:
    - ModelIntentStorageInput: Input model for intent storage (classified intent data)
    - ModelIntentStorageOutput: Output model for storage results (node ID, success)

All models are:
    - Frozen (immutable after creation)
    - Extra="forbid" (no extra fields allowed)
    - Strongly typed (no Any types)
"""

from .model_intent_storage_input import ModelIntentStorageInput
from .model_intent_storage_output import ModelIntentStorageOutput

__all__ = [
    "ModelIntentStorageInput",
    "ModelIntentStorageOutput",
]
