# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the OMN-9266 runtime health aggregate contract."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.models import ModelRuntimeAggregateHealth

pytestmark = pytest.mark.integration


def test_runtime_aggregate_health_contract_accepts_service_health_shape() -> None:
    """The exported model validates a representative :8085/health details shape."""
    payload = {
        "healthy": True,
        "degraded": False,
        "startup_in_progress": False,
        "is_running": True,
        "is_draining": False,
        "pending_message_count": 0,
        "max_concurrent_handlers": 10,
        "handler_pool_size": 5,
        "in_flight_tasks": 0,
        "batch_response_enabled": False,
        "batch_response_pending": 0,
        "event_bus": {
            "healthy": True,
            "started": True,
            "environment": "test",
            "bootstrap_servers": "redpanda:9092",
            "circuit_state": "closed",
            "subscriber_count": 3,
            "topic_count": 2,
            "consumer_count": 1,
        },
        "event_bus_healthy": True,
        "failed_handlers": {},
        "skipped_handlers": {},
        "registered_handlers": ["HandlerNodeHeartbeat"],
        "handlers": {"HandlerNodeHeartbeat": {"healthy": True}},
        "handler_pools": {},
        "no_handlers_registered": False,
        "config_prefetch_status": "ok",
        "local_ingress": {"enabled": False, "running": False},
        "components": [],
    }

    model = ModelRuntimeAggregateHealth.model_validate(payload, strict=False)

    assert model.is_running is True
    assert model.event_bus.circuit_state == "closed"
    assert model.model_dump(mode="json")["event_bus"]["topic_count"] == 2
