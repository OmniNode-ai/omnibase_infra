# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for consumption count sources.

This module provides the Protocol definition for consumption count sources.
Used by WiringHealthChecker for dependency injection.

See Also:
    - OMN-1895: Wiring health monitor implementation
    - WiringHealthChecker: Checker that uses this protocol
"""

from __future__ import annotations

from typing import Protocol


class ProtocolConsumptionCountSource(Protocol):
    """Protocol for consumption count sources (EventBusSubcontractWiring).

    Any class that provides consumption counts must implement this protocol.
    Used by WiringHealthChecker for dependency injection.

    Example:
        >>> class EventBusSubcontractWiring(ProtocolConsumptionCountSource):
        ...     def get_consumption_counts(self) -> dict[str, int]:
        ...         return {"topic1": 98, "topic2": 195}
    """

    def get_consumption_counts(self) -> dict[str, int]:
        """Get consumption counts per topic.

        Returns:
            Dictionary mapping topic name to consumption count.
        """
        ...


__all__ = ["ProtocolConsumptionCountSource"]
