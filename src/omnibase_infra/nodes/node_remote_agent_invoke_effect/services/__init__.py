# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Service boundaries for node_remote_agent_invoke_effect."""

from omnibase_infra.nodes.node_remote_agent_invoke_effect.services.lifecycle_event_sink import (
    EventBusLifecycleEventSink,
    ProtocolLifecycleEventSink,
)

__all__ = ["EventBusLifecycleEventSink", "ProtocolLifecycleEventSink"]
