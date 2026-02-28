# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Re-export canonical ModelRewardAssignedEvent from omnibase_core.

The local definition was replaced in OMN-2928 (gap:164320af CONTRACT_DRIFT fix).
Canonical model lives at omnibase_core.models.objective.model_reward_assigned_event.

Published to: ``onex.evt.omnimemory.reward-assigned.v1``
"""

from __future__ import annotations

from omnibase_core.models.objective.model_reward_assigned_event import (
    ModelRewardAssignedEvent,
)

__all__: list[str] = ["ModelRewardAssignedEvent"]
