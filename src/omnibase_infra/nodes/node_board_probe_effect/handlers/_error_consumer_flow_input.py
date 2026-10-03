# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An unreadable consumer-flow surface."""


class ConsumerFlowInputError(RuntimeError):
    """Could not look: return read_ok=False, never a negative finding."""


class ConsumerFlowLaneUnsettledError(ConsumerFlowInputError):
    """One settle window ran out; wait one more before grading (OMN-20410)."""
