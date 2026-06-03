# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocols for the topic-migration lag surface (OMN-12632)."""

from omnibase_infra.migration.protocols.protocol_kafka_lag_consumer import (
    ProtocolKafkaLagConsumer,
)

__all__ = ["ProtocolKafkaLagConsumer"]
