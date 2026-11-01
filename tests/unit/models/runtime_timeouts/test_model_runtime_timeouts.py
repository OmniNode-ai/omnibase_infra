# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime timeouts are read from the declared file, not from call-site literals (OMN-21719)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.runtime_timeouts import (
    RUNTIME_TIMEOUTS,
    ModelRuntimeTimeouts,
    load_runtime_timeouts,
)
from omnibase_infra.models.runtime_timeouts import (
    model_runtime_timeouts as model_module,
)

pytestmark = pytest.mark.unit

_DECLARED = Path(model_module.__file__).parent / "runtime_timeouts.yaml"
_SRC = Path(model_module.__file__).resolve().parents[2]

# (module path under src/omnibase_infra, called name, keyword) -> declared key.
_KEYWORD_WIRING: dict[tuple[str, str, str], str] = {
    ("runtime/request_response_wiring.py", "AIOKafkaConsumer", "session_timeout_ms"): "kafka_session_timeout_ms",
    ("runtime/request_response_wiring.py", "AIOKafkaConsumer", "heartbeat_interval_ms"): "kafka_heartbeat_interval_ms",
    ("runtime/request_response_wiring.py", "AIOKafkaConsumer", "max_poll_interval_ms"): "kafka_max_poll_interval_ms",
    ("runtime/request_response_wiring.py", "_init_circuit_breaker", "reset_timeout"): "circuit_breaker_reset_seconds",
    ("projectors/snapshot_publisher_registration.py", "AIOKafkaConsumer", "session_timeout_ms"): "kafka_session_timeout_ms",
    ("projectors/snapshot_publisher_registration.py", "AIOKafkaConsumer", "heartbeat_interval_ms"): "kafka_heartbeat_interval_ms",
    ("projectors/snapshot_publisher_registration.py", "AIOKafkaConsumer", "max_poll_interval_ms"): "kafka_max_poll_interval_ms",
    ("mixins/mixin_llm_http_transport.py", "_init_circuit_breaker", "reset_timeout"): "circuit_breaker_reset_seconds",
    ("runtime/node_invocation_adapter.py", "wait_for", "timeout"): "terminal_seconds",
}  # fmt: skip


def _block() -> dict[str, object]:
    document = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))
    return dict(document["timeouts"])


def _wired_keywords(relative_path: str) -> set[tuple[str, str, str]]:
    """Return (called name, keyword, RUNTIME_TIMEOUTS attribute) for each wired keyword."""
    tree = ast.parse((_SRC / relative_path).read_text(encoding="utf-8"))
    found: set[tuple[str, str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        for keyword in node.keywords:
            value = keyword.value
            if (
                keyword.arg is not None
                and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "RUNTIME_TIMEOUTS"
            ):
                found.add((name, keyword.arg, value.attr))
    return found


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
        {"health_check_min_seconds": 90.0, "health_check_default_seconds": 95.0},
        {
            "handler_shutdown_min_seconds": 400.0,
            "handler_shutdown_default_seconds": 350.0,
        },
        {"kafka_heartbeat_interval_ms": 45000},
        {"kafka_max_poll_interval_ms": 30000},
    ],
)
def test_inconsistent_values_fail_to_load(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelRuntimeTimeouts.model_validate({**_block(), **override})


@pytest.mark.parametrize(
    "key",
    [
        "capability_probe_ttl_seconds",
        "kafka_session_timeout_ms",
        "kafka_heartbeat_interval_ms",
        "kafka_max_poll_interval_ms",
        "snapshot_consumer_timeout_ms",
        "local_dispatch_seconds",
    ],
)
@pytest.mark.parametrize("bad", [1.5, 120.0])
def test_a_fractional_or_float_value_for_an_integer_key_is_refused(
    key: str, bad: float
) -> None:
    with pytest.raises(ValidationError):
        ModelRuntimeTimeouts.model_validate({**_block(), key: bad})


@pytest.mark.parametrize(
    ("relative_path", "called", "keyword", "key"),
    [(*wire[:3], key) for wire, key in sorted(_KEYWORD_WIRING.items())],
)
def test_keyword_call_sites_are_wired_to_their_declared_key(
    relative_path: str, called: str, keyword: str, key: str
) -> None:
    assert (called, keyword, key) in _wired_keywords(relative_path)


_SENTINEL_SCRIPT = textwrap.dedent(
    """
    import inspect
    import sys
    from pathlib import Path

    from omnibase_infra.models.runtime_timeouts import model_runtime_timeouts as m
    import omnibase_infra.models.runtime_timeouts as pkg

    sentinel = m.ModelRuntimeTimeouts.from_file(Path(sys.argv[1]))
    m._TIMEOUTS_PATH = Path(sys.argv[1])
    m.load_runtime_timeouts.cache_clear()
    pkg.RUNTIME_TIMEOUTS = m.load_runtime_timeouts()
    assert pkg.RUNTIME_TIMEOUTS == sentinel

    from omnibase_infra.mixins import mixin_llm_http_transport, mixin_node_introspection
    from omnibase_infra.probes import capability_probe
    from omnibase_infra.projectors import snapshot_publisher_registration
    from omnibase_infra.runtime import (
        node_invocation_adapter,
        protocol_lifecycle_executor as lifecycle,
    )

    def default(function, name):
        return inspect.signature(function).parameters[name].default

    probe = sentinel.capability_probe_seconds
    expected = [
        (capability_probe.PROBE_TTL_SECONDS, sentinel.capability_probe_ttl_seconds),
        (default(capability_probe.socket_check, "timeout"), probe),
        (default(capability_probe.kafka_reachable, "timeout"), probe),
        (default(capability_probe.http_health_check, "timeout"), probe),
        (default(capability_probe.probe_platform_tier, "kafka_timeout"), probe),
        (default(capability_probe.probe_platform_tier, "intel_timeout"), probe),
        (lifecycle.MIN_HEALTH_CHECK_TIMEOUT, sentinel.health_check_min_seconds),
        (lifecycle.MAX_HEALTH_CHECK_TIMEOUT, sentinel.health_check_max_seconds),
        (lifecycle.DEFAULT_HEALTH_CHECK_TIMEOUT, sentinel.health_check_default_seconds),
        (lifecycle.MIN_HANDLER_SHUTDOWN_TIMEOUT, sentinel.handler_shutdown_min_seconds),
        (lifecycle.MAX_HANDLER_SHUTDOWN_TIMEOUT, sentinel.handler_shutdown_max_seconds),
        (
            lifecycle.DEFAULT_HANDLER_SHUTDOWN_TIMEOUT,
            sentinel.handler_shutdown_default_seconds,
        ),
        (
            default(
                snapshot_publisher_registration.SnapshotPublisherRegistration.__init__,
                "consumer_timeout_ms",
            ),
            sentinel.snapshot_consumer_timeout_ms,
        ),
        (
            default(
                node_invocation_adapter.NodeInvocationAdapter.__init__,
                "health_probe_timeout",
            ),
            sentinel.health_probe_seconds,
        ),
        (
            default(
                node_invocation_adapter.NodeInvocationAdapter.dispatch,
                "timeout_seconds",
            ),
            sentinel.local_dispatch_seconds,
        ),
    ]
    transport = mixin_llm_http_transport.MixinLlmHttpTransport
    expected += [
        (
            default(transport._init_llm_http_transport, "max_timeout_seconds"),
            sentinel.llm_transport_max_seconds,
        ),
        (
            default(transport._execute_llm_http_call, "timeout_seconds"),
            sentinel.llm_post_seconds,
        ),
    ]
    introspection = mixin_node_introspection.MixinNodeIntrospection
    backoff = sentinel.introspection_base_backoff_seconds
    expected += [
        (default(introspection._ack_listener_loop, "base_backoff_seconds"), backoff),
        (default(introspection._registry_listener_loop, "base_backoff_seconds"), backoff),
        (
            default(introspection.start_introspection_tasks, "heartbeat_interval_seconds"),
            sentinel.introspection_heartbeat_interval_seconds,
        ),
    ]
    wrong = [pair for pair in expected if pair[0] != pair[1]]
    assert not wrong, wrong
    print(len(expected))
    """
)


def test_call_sites_follow_the_declared_file_not_a_literal(tmp_path: Path) -> None:
    """Import every call site against a doubled file; a surviving literal keeps its old value."""
    doubled = {key: value * 2 for key, value in _block().items()}
    path = tmp_path / "runtime_timeouts.yaml"
    path.write_text(yaml.safe_dump({"timeouts": doubled}), encoding="utf-8")
    assert ModelRuntimeTimeouts.from_file(path) != RUNTIME_TIMEOUTS

    completed = subprocess.run(
        [sys.executable, "-c", _SENTINEL_SCRIPT, str(path)],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == "20"
