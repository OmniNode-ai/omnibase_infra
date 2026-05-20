# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelRuntimeAggregateHealth and ModelEventBusAggregateHealth.

Validates the canonical typed model for the :8085/health aggregate response
shape introduced by OMN-9266.  Tests cover:

- Construction from a raw dict matching the real RuntimeHostProcess.health_check()
  return shape.
- Field presence and type constraints for previously-provisional fields
  (is_running, is_draining, pending_message_count, handler_pools,
  event_bus.circuit_state, event_bus.subscriber_count,
  event_bus.topic_count, event_bus.consumer_count).
- JSON round-trip serialisation.
- Rejection of invalid circuit_state values.
- extra="allow" semantics — unknown fields do not raise.

Related Tickets:
    - OMN-9266: Cite ServiceHealth aggregate response-shape model for :8085/health

.. versionadded:: OMN-9266
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_runtime_aggregate_health import (
    ModelEventBusAggregateHealth,
    ModelRuntimeAggregateHealth,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_HEALTHY_EVENT_BUS: dict[str, object] = {
    "healthy": True,
    "started": True,
    "environment": "prod",
    "bootstrap_servers": "redpanda:9092",
    "circuit_state": "closed",
    "subscriber_count": 374,
    "topic_count": 42,
    "consumer_count": 12,
}

_HEALTHY_RUNTIME: dict[str, object] = {
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
    "event_bus": _HEALTHY_EVENT_BUS,
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


# ---------------------------------------------------------------------------
# ModelEventBusAggregateHealth
# ---------------------------------------------------------------------------


class TestModelEventBusAggregateHealth:
    """Tests for ModelEventBusAggregateHealth — the event_bus sub-shape."""

    def test_construction_from_valid_dict(self) -> None:
        """Happy-path construction from a real EventBusKafka.health_check() dict."""
        model = ModelEventBusAggregateHealth.model_validate(
            _HEALTHY_EVENT_BUS, strict=False
        )
        assert model.healthy is True
        assert model.started is True
        assert model.circuit_state == "closed"
        assert model.subscriber_count == 374
        assert model.topic_count == 42
        assert model.consumer_count == 12

    def test_circuit_state_open(self) -> None:
        """circuit_state='open' is valid (circuit breaker tripped)."""
        data = {**_HEALTHY_EVENT_BUS, "healthy": False, "circuit_state": "open"}
        model = ModelEventBusAggregateHealth.model_validate(data, strict=False)
        assert model.circuit_state == "open"
        assert model.healthy is False

    def test_circuit_state_invalid_rejected(self) -> None:
        """circuit_state values other than 'open'/'closed' are rejected."""
        data = {**_HEALTHY_EVENT_BUS, "circuit_state": "half-open"}
        with pytest.raises(ValidationError) as exc_info:
            ModelEventBusAggregateHealth.model_validate(data, strict=False)
        assert "circuit_state" in str(exc_info.value)

    def test_subscriber_count_negative_rejected(self) -> None:
        """subscriber_count must be >= 0."""
        data = {**_HEALTHY_EVENT_BUS, "subscriber_count": -1}
        with pytest.raises(ValidationError):
            ModelEventBusAggregateHealth.model_validate(data, strict=False)

    def test_extra_fields_allowed(self) -> None:
        """Unknown fields do not raise (extra='allow' semantics)."""
        data = {**_HEALTHY_EVENT_BUS, "future_field": "some_value"}
        model = ModelEventBusAggregateHealth.model_validate(data, strict=False)
        assert model.circuit_state == "closed"

    def test_json_roundtrip(self) -> None:
        """JSON round-trip preserves all known fields."""
        model = ModelEventBusAggregateHealth.model_validate(
            _HEALTHY_EVENT_BUS, strict=False
        )
        json_str = model.model_dump_json()
        restored = ModelEventBusAggregateHealth.model_validate_json(json_str)
        assert restored.subscriber_count == model.subscriber_count
        assert restored.circuit_state == model.circuit_state

    def test_unhealthy_not_started(self) -> None:
        """healthy=False, started=False is a valid (pre-start) state."""
        data = {
            "healthy": False,
            "started": False,
            "environment": "dev",
            "bootstrap_servers": "localhost:9092",
            "circuit_state": "closed",
            "subscriber_count": 0,
            "topic_count": 0,
            "consumer_count": 0,
        }
        model = ModelEventBusAggregateHealth.model_validate(data, strict=False)
        assert model.healthy is False
        assert model.started is False
        assert model.subscriber_count == 0


# ---------------------------------------------------------------------------
# ModelRuntimeAggregateHealth
# ---------------------------------------------------------------------------


class TestModelRuntimeAggregateHealth:
    """Tests for ModelRuntimeAggregateHealth — the full /health response shape."""

    def test_construction_from_healthy_dict(self) -> None:
        """Happy-path construction from a healthy runtime health_check() return dict."""
        model = ModelRuntimeAggregateHealth.model_validate(
            _HEALTHY_RUNTIME, strict=False
        )
        assert model.healthy is True
        assert model.is_running is True
        assert model.is_draining is False
        assert model.pending_message_count == 0
        assert model.no_handlers_registered is False
        assert model.config_prefetch_status == "ok"

    def test_event_bus_sub_model_parsed(self) -> None:
        """The nested event_bus dict is parsed into ModelEventBusAggregateHealth."""
        model = ModelRuntimeAggregateHealth.model_validate(
            _HEALTHY_RUNTIME, strict=False
        )
        assert isinstance(model.event_bus, ModelEventBusAggregateHealth)
        assert model.event_bus.subscriber_count == 374
        assert model.event_bus.circuit_state == "closed"

    def test_is_running_field_cited(self) -> None:
        """is_running=False is valid (runtime not yet started)."""
        data = {
            **_HEALTHY_RUNTIME,
            "healthy": False,
            "is_running": False,
            "event_bus_healthy": False,
            "event_bus": {**_HEALTHY_EVENT_BUS, "healthy": False, "started": False},
        }
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert model.is_running is False
        assert model.healthy is False

    def test_is_draining_field_cited(self) -> None:
        """is_draining=True represents graceful-shutdown drain phase."""
        data = {**_HEALTHY_RUNTIME, "is_draining": True}
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert model.is_draining is True

    def test_pending_message_count_field_cited(self) -> None:
        """pending_message_count reflects in-flight messages."""
        data = {**_HEALTHY_RUNTIME, "pending_message_count": 12}
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert model.pending_message_count == 12

    def test_pending_message_count_negative_rejected(self) -> None:
        """pending_message_count must be >= 0."""
        data = {**_HEALTHY_RUNTIME, "pending_message_count": -1}
        with pytest.raises(ValidationError):
            ModelRuntimeAggregateHealth.model_validate(data, strict=False)

    def test_handler_pools_field_cited(self) -> None:
        """handler_pools contains per-pool health metrics."""
        data = {
            **_HEALTHY_RUNTIME,
            "handler_pools": {"effect_pool": {"running": True, "size": 5}},
        }
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert "effect_pool" in model.handler_pools

    def test_degraded_state(self) -> None:
        """degraded=True, healthy=False with failed_handlers is a valid state."""
        data = {
            **_HEALTHY_RUNTIME,
            "healthy": False,
            "degraded": True,
            "failed_handlers": {"HandlerFoo": "import error: module not found"},
        }
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert model.degraded is True
        assert model.healthy is False
        assert "HandlerFoo" in model.failed_handlers

    def test_startup_in_progress_state(self) -> None:
        """startup_in_progress=True represents bus-live-but-handlers-not-ready."""
        data = {
            **_HEALTHY_RUNTIME,
            "healthy": False,
            "degraded": True,
            "startup_in_progress": True,
            "is_running": False,
        }
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert model.startup_in_progress is True

    def test_extra_fields_allowed(self) -> None:
        """Unknown fields do not raise (forward-compat extra='allow')."""
        data = {**_HEALTHY_RUNTIME, "future_runtime_field": "some_value"}
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert model.healthy is True

    def test_json_roundtrip(self) -> None:
        """JSON round-trip preserves previously-provisional fields."""
        model = ModelRuntimeAggregateHealth.model_validate(
            _HEALTHY_RUNTIME, strict=False
        )
        json_str = model.model_dump_json()
        restored = ModelRuntimeAggregateHealth.model_validate_json(json_str)
        assert restored.is_running == model.is_running
        assert restored.is_draining == model.is_draining
        assert restored.pending_message_count == model.pending_message_count
        assert restored.event_bus.subscriber_count == model.event_bus.subscriber_count
        assert restored.event_bus.circuit_state == model.event_bus.circuit_state

    def test_registered_handlers_list(self) -> None:
        """registered_handlers is a list of handler type strings."""
        model = ModelRuntimeAggregateHealth.model_validate(
            _HEALTHY_RUNTIME, strict=False
        )
        assert model.registered_handlers == ["HandlerNodeHeartbeat"]

    def test_no_handlers_registered_critical_state(self) -> None:
        """no_handlers_registered=True indicates a critical configuration error."""
        data = {
            **_HEALTHY_RUNTIME,
            "healthy": False,
            "no_handlers_registered": True,
            "registered_handlers": [],
        }
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert model.no_handlers_registered is True
        assert model.registered_handlers == []

    def test_config_prefetch_status_values(self) -> None:
        """config_prefetch_status accepts all documented values."""
        for status in (
            "pending",
            "skipped",
            "ok",
            "degraded_no_requirements",
            "degraded_error",
        ):
            data = {**_HEALTHY_RUNTIME, "config_prefetch_status": status}
            model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
            assert model.config_prefetch_status == status

    def test_components_list(self) -> None:
        """components contains per-component health dicts."""
        data = {
            **_HEALTHY_RUNTIME,
            "components": [
                {"name": "published_events_map", "status": "healthy"},
            ],
        }
        model = ModelRuntimeAggregateHealth.model_validate(data, strict=False)
        assert len(model.components) == 1
        assert model.components[0]["name"] == "published_events_map"


# ---------------------------------------------------------------------------
# Module-level __all__ export coverage
# ---------------------------------------------------------------------------


class TestExportsFromModelsInit:
    """Verify the models package exports the new classes."""

    def test_models_init_exports_aggregate_health(self) -> None:
        """ModelRuntimeAggregateHealth and ModelEventBusAggregateHealth are exported."""
        from omnibase_infra.runtime.models import (
            ModelEventBusAggregateHealth as ImportedEventBusAggregateHealth,
        )
        from omnibase_infra.runtime.models import (
            ModelRuntimeAggregateHealth as ImportedRuntimeAggregateHealth,
        )

        assert ImportedRuntimeAggregateHealth is ModelRuntimeAggregateHealth
        assert ImportedEventBusAggregateHealth is ModelEventBusAggregateHealth
