# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Selection Strategy Enumeration - Re-export for backward compatibility.

The canonical location for EnumSelectionStrategy is now omnibase_infra.enums.
This module re-exports it for backward compatibility with existing imports.

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery

Example:
    >>> from omnibase_infra.enums import EnumSelectionStrategy  # Preferred
    >>> # Or for backward compatibility:
    >>> from omnibase_infra.services import EnumSelectionStrategy
"""

from omnibase_infra.enums.enum_selection_strategy import EnumSelectionStrategy

__all__: list[str] = ["EnumSelectionStrategy"]
