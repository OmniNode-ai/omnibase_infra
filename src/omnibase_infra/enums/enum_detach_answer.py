# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Broker answer to a shared-listener detach request (OMN-22377)."""

from __future__ import annotations

from enum import Enum


class EnumDetachAnswer(str, Enum):
    """Broker answer to a detach request (rule A6)."""

    GRANTED = "granted"
    """Another confirmed session of the pool is attached: leave now."""

    WAIT = "wait"
    """A handover is in progress and no other confirmed session is attached."""

    ALONE = "alone"
    """No handover in progress and no other session."""
