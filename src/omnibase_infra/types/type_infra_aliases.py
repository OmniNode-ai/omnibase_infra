# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Infrastructure-specific type aliases for union complexity reduction.

These type aliases consolidate repeated union patterns found in omnibase_infra,
following the same pattern as omnibase_core.types.type_json.

When to use each alias:
    MessageOutputCategory: Use for message routing, dispatcher selection,
        and node output validation. Accepts both EnumMessageCategory (EVENT,
        COMMAND, INTENT) and EnumNodeOutputType (includes PROJECTION).

    PathInput: Use for function parameters that accept filesystem paths.
        Allows callers to pass either Path objects or string paths.

    PolicyTypeInput: Use for policy configuration APIs that need flexibility
        between enum values and string representations.

    ASTFunctionDef: Use when analyzing Python AST for function definitions,
        covering both sync and async functions.

See OMN-1358 for the union reduction initiative that drove these definitions.
"""

from __future__ import annotations

import ast
from pathlib import Path

from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType, EnumPolicyType

# Message category or node output type (for routing and validation)
# Replaces 24 occurrences of: EnumMessageCategory | EnumNodeOutputType
type MessageOutputCategory = EnumMessageCategory | EnumNodeOutputType

# Filesystem path input flexibility
# Replaces 11 occurrences of: Path | str
# Note: Named PathInput (not PathLike) to avoid collision with stdlib os.PathLike Protocol.
# PathInput represents filesystem path flexibility (Path | str), whereas os.PathLike is
# a Protocol for objects implementing __fspath__().
type PathInput = Path | str

# Policy type with string fallback for API flexibility
# Replaces 8 occurrences of: EnumPolicyType | str
# NOTE: This is an INPUT type alias for API flexibility. Pydantic validators
# using validate_policy_type_value() COERCE strings to EnumPolicyType, so
# the actual stored value is always an enum, ensuring type-safe field access.
type PolicyTypeInput = EnumPolicyType | str

# AST function definition node types
# Replaces 7 occurrences of: ast.AsyncFunctionDef | ast.FunctionDef
type ASTFunctionDef = ast.AsyncFunctionDef | ast.FunctionDef

__all__ = [
    "ASTFunctionDef",
    "MessageOutputCategory",
    "PathInput",
    "PolicyTypeInput",
]
