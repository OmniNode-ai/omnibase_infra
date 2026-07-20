# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deserializer support for node_kafka_replay_compute."""

from omnibase_infra.nodes.node_kafka_replay_compute.deserializers.deserializer_default import (
    default_envelope_deserializer,
)

__all__ = ["default_envelope_deserializer"]
