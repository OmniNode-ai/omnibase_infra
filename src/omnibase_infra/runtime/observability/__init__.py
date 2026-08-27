# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime self-observability seams (OMN-16777, epic OMN-16776).

Throughput truth for the platform's own execution: counters accumulated in
process and drained onto the heartbeat the runtime already emits.  No daemon,
no poller, no scraper, no metrics endpoint.
"""

from omnibase_infra.runtime.observability.consumer_flow_counters import (
    ConsumerFlowCounters,
    active_flow_key,
    get_consumer_flow_counters,
    record_active_dlq,
    record_active_error,
    record_active_out,
    record_produced_topic,
    reset_consumer_flow_counters,
)

__all__ = [
    "ConsumerFlowCounters",
    "active_flow_key",
    "get_consumer_flow_counters",
    "record_active_dlq",
    "record_active_error",
    "record_active_out",
    "record_produced_topic",
    "reset_consumer_flow_counters",
]
