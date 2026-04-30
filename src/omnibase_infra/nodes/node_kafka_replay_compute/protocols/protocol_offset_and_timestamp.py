# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka offset timestamp protocol for replay compute."""

from __future__ import annotations

from typing import Protocol


class ProtocolOffsetAndTimestamp(Protocol):
    """Kafka offset lookup result for wall-clock replay bounds."""

    offset: int


__all__ = ["ProtocolOffsetAndTimestamp"]
