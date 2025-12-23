# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Type aliases for validation-related unions.

These type aliases consolidate commonly used union patterns in validators
to reduce union count and improve code readability.

Related:
    - OMN-1001: Union Reduction Phase 1
"""

import ast

# Type alias for function definitions (sync or async)
# Used in AST analysis validators
type AnyFunctionDef = ast.FunctionDef | ast.AsyncFunctionDef

__all__ = ["AnyFunctionDef"]
