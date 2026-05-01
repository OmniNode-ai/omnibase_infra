# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka message protocol for replay compute."""

from __future__ import annotations

from typing import Protocol


class ProtocolKafkaMessage(Protocol):
    """Kafka message fields consumed by the replay handler."""

    topic: str
    partition: int
    offset: int
    value: bytes | None


__all__ = ["ProtocolKafkaMessage"]
