# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""
Message Category Enumeration — canonical definition lives in omnibase_core.

Re-exported from omnibase_core.enums for backwards-compatible infra imports.
Do NOT define EnumMessageCategory here; the single source of truth is
omnibase_core.enums.enum_execution_shape.EnumMessageCategory.
"""

from omnibase_core.enums import EnumMessageCategory

__all__ = ["EnumMessageCategory"]
