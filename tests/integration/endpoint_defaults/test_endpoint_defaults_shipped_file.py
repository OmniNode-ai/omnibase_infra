# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped endpoint defaults file keeps the values consumers had as literals (OMN-21690).

The expected values are written out here on purpose: a test that compares a
consumer to the loader would pass after any edit of the file.
"""

from __future__ import annotations

import ipaddress

import pytest

from omnibase_infra.cli.artifact_reconcile import _get_bootstrap_servers
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.models.endpoint_defaults import load_endpoint_defaults
from omnibase_infra.observability.handlers.model_metrics_handler_config import (
    ModelMetricsHandlerConfig,
)
from omnibase_infra.runtime.models.model_runtime_scheduler_config import (
    ModelRuntimeSchedulerConfig,
)
from omnibase_infra.services.health_checker import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
)

pytestmark = pytest.mark.integration


def test_the_shipped_file_holds_the_values_the_code_used_to_hard_code() -> None:
    defaults = load_endpoint_defaults()

    assert defaults.loopback_address == "127.0.0.1"
    assert ipaddress.ip_address(defaults.bind_all_address).is_unspecified
    assert defaults.loopback_host_names == ("localhost", "127.0.0.1", "::1")
    assert defaults.local_host == "localhost"
    assert defaults.local_kafka_bootstrap_servers == "localhost:19092"
    assert defaults.local_valkey_port == 6379
    assert defaults.runtime_health_port == 8085


def test_consumers_resolve_the_previous_literal_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("KAFKA_BOOTSTRAP_SERVERS", "VALKEY_HOST", "VALKEY_PORT"):
        monkeypatch.delenv(name, raising=False)

    assert _get_bootstrap_servers() == "localhost:19092"
    assert ModelKafkaEventBusConfig.default().bootstrap_servers == "localhost:19092"
    scheduler = ModelRuntimeSchedulerConfig.default()
    assert scheduler.valkey_host == "localhost"
    assert scheduler.valkey_port == 6379
    assert ModelMetricsHandlerConfig().host == "127.0.0.1"
    assert load_endpoint_defaults().bind_all_address == DEFAULT_HTTP_HOST
    assert ipaddress.ip_address(DEFAULT_HTTP_HOST).is_unspecified
    assert DEFAULT_HTTP_PORT == 8085
