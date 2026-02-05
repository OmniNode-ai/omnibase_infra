# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for emission count sources.

This module provides the Protocol definition for emission count sources.
Used by WiringHealthChecker for dependency injection.

See Also:
    - OMN-1895: Wiring health monitor implementation
    - WiringHealthChecker: Checker that uses this protocol
"""

from __future__ import annotations

from typing import Protocol


class ProtocolEmissionCountSource(Protocol):
    """Protocol for emission count sources (EventBusKafka).

    Any class that provides emission counts must implement this protocol.
    Used by WiringHealthChecker for dependency injection.

    Example:
        >>> class EventBusKafka(ProtocolEmissionCountSource):
        ...     def get_emission_counts(self) -> dict[str, int]:
        ...         return {"topic1": 100, "topic2": 200}
    """

    def get_emission_counts(self) -> dict[str, int]:
        """Get emission counts per topic.

        Returns:
            Dictionary mapping topic name to emission count.
        """
        ...


__all__ = ["ProtocolEmissionCountSource"]
