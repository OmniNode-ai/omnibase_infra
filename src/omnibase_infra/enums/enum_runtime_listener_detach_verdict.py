# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The broker's three answers to a detach request (rule A6, OMN-22379)."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class EnumRuntimeListenerDetachVerdict(StrEnum):
    """What a runtime may do after asking the broker to leave the listener.

    GRANTED: another confirmed session of the pool is attached and holds no
        outstanding grant, so this session may stop accepting.
    WAIT: the agent has a handover in progress for the pool and no other
        confirmed session is attached right now.
    ALONE: no handover is in progress and no other session is attached.
    """

    GRANTED = "granted"
    WAIT = "wait"
    ALONE = "alone"


__all__: list[str] = ["EnumRuntimeListenerDetachVerdict"]
