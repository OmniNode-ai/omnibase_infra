# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime timeouts are read from the declared file, not from call-site literals (OMN-21719)."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.mixins import mixin_llm_http_transport, mixin_node_introspection
from omnibase_infra.models.runtime_timeouts import (
    RUNTIME_TIMEOUTS,
    ModelRuntimeTimeouts,
    load_runtime_timeouts,
    model_runtime_timeouts,
)
from omnibase_infra.probes import capability_probe
from omnibase_infra.projectors import snapshot_publisher_registration
from omnibase_infra.runtime import (
    node_invocation_adapter,
    protocol_lifecycle_executor,
    request_response_wiring,
)

pytestmark = pytest.mark.unit

_DECLARED = Path(model_runtime_timeouts.__file__).parent / "runtime_timeouts.yaml"


def _block() -> dict[str, object]:
    document = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))
    return dict(document["timeouts"])


def _default(function: Callable[..., object], name: str) -> object:
    return inspect.signature(function).parameters[name].default


def test_loaded_timeouts_equal_the_declared_file() -> None:
    assert load_runtime_timeouts() is RUNTIME_TIMEOUTS
    assert ModelRuntimeTimeouts.model_validate(_block()) == RUNTIME_TIMEOUTS


@pytest.mark.parametrize("key", sorted(_block()))
def test_a_block_missing_a_value_fails_to_load(key: str) -> None:
    block = _block()
    del block[key]
    with pytest.raises(ValidationError):
        ModelRuntimeTimeouts.model_validate(block)


def test_a_file_without_a_timeouts_mapping_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "runtime_timeouts.yaml"
    path.write_text("other: 1\n", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        ModelRuntimeTimeouts.from_file(path)


@pytest.mark.parametrize(
    "override",
    [
        {"health_check_default_seconds": 61.0},
        {"handler_shutdown_default_seconds": 0.5},
        {"kafka_heartbeat_interval_ms": 45000},
    ],
)
def test_inconsistent_values_fail_to_load(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelRuntimeTimeouts.model_validate({**_block(), **override})


def test_call_sites_carry_the_declared_values() -> None:
    probe = RUNTIME_TIMEOUTS.capability_probe_seconds
    assert (
        RUNTIME_TIMEOUTS.capability_probe_ttl_seconds
        == capability_probe.PROBE_TTL_SECONDS
    )
    assert _default(capability_probe.socket_check, "timeout") == probe
    assert _default(capability_probe.kafka_reachable, "timeout") == probe
    assert _default(capability_probe.http_health_check, "timeout") == probe
    assert _default(capability_probe.probe_platform_tier, "kafka_timeout") == probe
    assert _default(capability_probe.probe_platform_tier, "intel_timeout") == probe

    lifecycle = protocol_lifecycle_executor
    assert (
        RUNTIME_TIMEOUTS.health_check_min_seconds == lifecycle.MIN_HEALTH_CHECK_TIMEOUT
    )
    assert (
        RUNTIME_TIMEOUTS.health_check_max_seconds == lifecycle.MAX_HEALTH_CHECK_TIMEOUT
    )
    assert (
        RUNTIME_TIMEOUTS.health_check_default_seconds
        == lifecycle.DEFAULT_HEALTH_CHECK_TIMEOUT
    )
    assert (
        RUNTIME_TIMEOUTS.handler_shutdown_min_seconds
        == lifecycle.MIN_HANDLER_SHUTDOWN_TIMEOUT
    )
    assert (
        RUNTIME_TIMEOUTS.handler_shutdown_max_seconds
        == lifecycle.MAX_HANDLER_SHUTDOWN_TIMEOUT
    )
    assert (
        RUNTIME_TIMEOUTS.handler_shutdown_default_seconds
        == lifecycle.DEFAULT_HANDLER_SHUTDOWN_TIMEOUT
    )

    for module in (request_response_wiring, snapshot_publisher_registration):
        source = inspect.getsource(module)
        for key in (
            "kafka_session_timeout_ms",
            "kafka_heartbeat_interval_ms",
            "kafka_max_poll_interval_ms",
        ):
            assert f"=RUNTIME_TIMEOUTS.{key}" in source
    assert (
        _default(
            snapshot_publisher_registration.SnapshotPublisherRegistration.__init__,
            "consumer_timeout_ms",
        )
        == RUNTIME_TIMEOUTS.snapshot_consumer_timeout_ms
    )

    assert (
        _default(
            node_invocation_adapter.NodeInvocationAdapter.__init__,
            "health_probe_timeout",
        )
        == RUNTIME_TIMEOUTS.health_probe_seconds
    )
    assert (
        _default(
            node_invocation_adapter.NodeInvocationAdapter.dispatch, "timeout_seconds"
        )
        == RUNTIME_TIMEOUTS.local_dispatch_seconds
    )

    transport = mixin_llm_http_transport.MixinLlmHttpTransport
    assert (
        _default(transport._init_llm_http_transport, "max_timeout_seconds")
        == RUNTIME_TIMEOUTS.llm_transport_max_seconds
    )
    assert (
        _default(transport._execute_llm_http_call, "timeout_seconds")
        == RUNTIME_TIMEOUTS.llm_post_seconds
    )

    introspection = mixin_node_introspection.MixinNodeIntrospection
    backoff = RUNTIME_TIMEOUTS.introspection_base_backoff_seconds
    assert _default(introspection._ack_listener_loop, "base_backoff_seconds") == backoff
    assert (
        _default(introspection._registry_listener_loop, "base_backoff_seconds")
        == backoff
    )
    assert (
        _default(introspection.start_introspection_tasks, "heartbeat_interval_seconds")
        == RUNTIME_TIMEOUTS.introspection_heartbeat_interval_seconds
    )
