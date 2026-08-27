# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Per-topic verdict for the DLQ depth/arrival evaluation (OMN-16769)."""

from __future__ import annotations

from enum import Enum


class EnumDlqDepthVerdict(str, Enum):
    """Outcome of evaluating one DLQ topic against its declared bounds.

    ``ALERT_ARRIVALS`` is the PRIMARY alert signal. ``ALERT_DEPTH`` is
    secondary and disabled by default — see
    :class:`ModelDlqThresholdPolicy.max_retained_depth` for why a depth
    bound cannot be the primary signal on a topic carrying a standing
    backlog (OMN-16769 AC4).
    """

    OK = "ok"
    ALERT_ARRIVALS = "alert_arrivals"
    ALERT_DEPTH = "alert_depth"


__all__ = ["EnumDlqDepthVerdict"]
