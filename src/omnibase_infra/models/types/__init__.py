# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared type aliases for omnibase_infra models.

This module provides centralized type definitions to reduce union duplication
across the infrastructure codebase.
"""

from omnibase_infra.models.types.json_types import (
    EnvelopeDict,
    JsonPrimitive,
    JsonValue,
    ResultDict,
)

__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "EnvelopeDict",
    "ResultDict",
]
