# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-target replay bound for the contract-native DLQ replay node (OMN-24177).

At most ``max_replays`` replays per (original topic, target consumer group) in
any window of ``window_seconds``; the next record for that pair is quarantined
instead (OMN-24177).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDlqTargetReplayBound(BaseModel):
    """Bound on how many times one (original topic, target consumer group)
    pair may be replayed within a sliding window.

    At most ``max_replays`` replays per (original topic, target consumer
    group) in any window of ``window_seconds``; the next record for that pair
    is quarantined instead (OMN-24177).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_replays: int = Field(..., gt=0)
    window_seconds: float = Field(..., gt=0.0)


__all__ = ["ModelDlqTargetReplayBound"]
