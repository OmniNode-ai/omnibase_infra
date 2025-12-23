# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module provides centralized type definitions to reduce union duplication
across the infrastructure codebase.

Note: JsonPrimitive and JsonValue are imported from omnibase_core.types.
EnvelopeDict and ResultDict are infra-specific type aliases for protocol methods.
"""

from omnibase_core.types import JsonPrimitive, JsonValue

# Infra-specific envelope dictionary types for protocol methods.
# Values are JSON-serializable, validated at runtime by executor implementations.
type EnvelopeDict = dict[str, JsonValue]
type ResultDict = dict[str, JsonValue]

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "EnvelopeDict",
    "ResultDict",
]
