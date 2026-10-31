# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every consumer of a host, address or port default carries the declared value (OMN-21690)."""

from __future__ import annotations

import pytest

from omnibase_infra.cli.artifact_reconcile import _get_bootstrap_servers
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.handlers.handler_ledger_write_guard import (
    dsn_is_canonical,
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

pytestmark = pytest.mark.unit


def test_the_artifact_reconcile_cli_defaults_to_the_declared_local_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    assert (
        _get_bootstrap_servers()
        == load_endpoint_defaults().local_kafka_bootstrap_servers
    )


def test_the_kafka_default_config_names_the_declared_local_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    config = ModelKafkaEventBusConfig.default()

    assert (
        config.bootstrap_servers
        == load_endpoint_defaults().local_kafka_bootstrap_servers
    )


def test_the_scheduler_default_config_names_the_declared_valkey(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VALKEY_HOST", raising=False)
    monkeypatch.delenv("VALKEY_PORT", raising=False)
    defaults = load_endpoint_defaults()

    config = ModelRuntimeSchedulerConfig.default()

    assert config.valkey_host == defaults.local_host
    assert config.valkey_port == defaults.local_valkey_port


def test_the_metrics_config_binds_the_declared_loopback_address() -> None:
    assert ModelMetricsHandlerConfig().host == load_endpoint_defaults().loopback_address


def test_the_health_server_defaults_are_the_declared_bind_all_address_and_port() -> (
    None
):
    defaults = load_endpoint_defaults()

    assert defaults.bind_all_address == DEFAULT_HTTP_HOST
    assert defaults.runtime_health_port == DEFAULT_HTTP_PORT


@pytest.mark.parametrize("host", sorted(load_endpoint_defaults().loopback_host_names))
def test_the_ledger_guard_treats_each_declared_loopback_host_as_local(
    host: str,
) -> None:
    name = f"[{host}]" if ":" in host else host

    assert dsn_is_canonical(f"postgresql://user@{name}/ledger") is False


def test_the_ledger_guard_treats_another_host_as_the_shared_database() -> None:
    assert dsn_is_canonical("postgresql://user@db.example.test/ledger") is True
