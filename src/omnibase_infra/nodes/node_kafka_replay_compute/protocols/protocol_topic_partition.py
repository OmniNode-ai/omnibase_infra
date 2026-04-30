# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka topic-partition protocol for replay compute."""

from __future__ import annotations

from typing import Protocol


class ProtocolTopicPartition(Protocol):
    """Structural topic-partition key used by Kafka replay consumers."""

    topic: str
    partition: int


__all__ = ["ProtocolTopicPartition"]
