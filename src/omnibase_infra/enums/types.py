# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Type aliases for enum unions.

These type aliases consolidate commonly used enum union patterns
to reduce union count and improve code readability.

Related:
    - OMN-1001: Union Reduction Phase 1
    - EnumMessageCategory: Message routing categories
    - EnumNodeOutputType: Node output validation types
"""

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType

# Type alias for message/output type unions used in validators
# Combines message routing (EVENT, COMMAND, INTENT) with node output (includes PROJECTION)
type MessageOutputType = EnumMessageCategory | EnumNodeOutputType

__all__ = ["MessageOutputType"]
