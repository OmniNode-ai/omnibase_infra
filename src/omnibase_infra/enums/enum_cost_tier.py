# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Cost tier enumeration for LLM routing decisions.

Defines the canonical cost tier levels used by the bifrost gateway to
select backend endpoints based on cost budget constraints.

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
"""

from enum import Enum


class EnumCostTier(str, Enum):
    """Cost tier classification for LLM backend routing.

    Used by the bifrost gateway to match routing rules against
    incoming request cost preferences.

    Attributes:
        LOW: Lowest cost tier — prefer small/quantized local models.
        MID: Medium cost tier — balanced quality and cost.
        HIGH: Highest cost tier — prefer large/quality models.
    """

    LOW = "low"
    MID = "mid"
    HIGH = "high"


__all__: list[str] = ["EnumCostTier"]
