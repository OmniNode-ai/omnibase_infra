# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Infrastructure-specific type aliases for union complexity reduction.

These type aliases consolidate repeated union patterns found in omnibase_infra,
following the same pattern as omnibase_core.types.type_json.

See OMN-1358 for the union reduction initiative that drove these definitions.
"""

from __future__ import annotations

import ast
from pathlib import Path

from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.enums import EnumMessageCategory, EnumNodeOutputType, EnumPolicyType

# Message category or node output type (for routing and validation)
# Replaces 24 occurrences of: EnumMessageCategory | EnumNodeOutputType
type MessageOutputCategory = EnumMessageCategory | EnumNodeOutputType

# Filesystem path input flexibility
# Replaces 11 occurrences of: Path | str
# Note: Named PathInput (not PathLike) to avoid collision with stdlib os.PathLike
type PathInput = Path | str

# Policy type with string fallback for API flexibility
# Replaces 8 occurrences of: EnumPolicyType | str
type PolicyTypeInput = EnumPolicyType | str

# AST function definition node types
# Replaces 7 occurrences of: ast.AsyncFunctionDef | ast.FunctionDef
type ASTFunctionDef = ast.AsyncFunctionDef | ast.FunctionDef

# Version input flexibility
# Replaces 4 occurrences of: ModelSemVer | str
type VersionInput = ModelSemVer | str

__all__ = [
    "ASTFunctionDef",
    "MessageOutputCategory",
    "PathInput",
    "PolicyTypeInput",
    "VersionInput",
]
