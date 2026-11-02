# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime timeouts are read from the declared file, not from call-site literals (OMN-21719)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import zipfile
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

# (module path under src/omnibase_infra, enclosing scope, called name, keyword) -> declared key.
_KEYWORD_WIRING: dict[tuple[str, str, str, str], str] = {
    ("runtime/request_response_wiring.py", "RequestResponseWiring._wire_instance", "AIOKafkaConsumer", "session_timeout_ms"): "kafka_session_ms",
    ("runtime/request_response_wiring.py", "RequestResponseWiring._wire_instance", "AIOKafkaConsumer", "heartbeat_interval_ms"): "kafka_heartbeat_interval_ms",
    ("runtime/request_response_wiring.py", "RequestResponseWiring._wire_instance", "AIOKafkaConsumer", "max_poll_interval_ms"): "kafka_max_poll_interval_ms",
    ("runtime/request_response_wiring.py", "RequestResponseWiring.__init__", "_init_circuit_breaker", "reset_timeout"): "circuit_breaker_reset_seconds",
    ("projectors/snapshot_publisher_registration.py", "SnapshotPublisherRegistration._load_cache_from_topic", "AIOKafkaConsumer", "session_timeout_ms"): "kafka_session_ms",
    ("projectors/snapshot_publisher_registration.py", "SnapshotPublisherRegistration._load_cache_from_topic", "AIOKafkaConsumer", "heartbeat_interval_ms"): "kafka_heartbeat_interval_ms",
    ("projectors/snapshot_publisher_registration.py", "SnapshotPublisherRegistration._load_cache_from_topic", "AIOKafkaConsumer", "max_poll_interval_ms"): "kafka_max_poll_interval_ms",
    ("mixins/mixin_llm_http_transport.py", "MixinLlmHttpTransport._init_llm_http_transport", "_init_circuit_breaker", "reset_timeout"): "circuit_breaker_reset_seconds",
    ("runtime/node_invocation_adapter.py", "NodeInvocationAdapter._run_local_bus_round_trip", "wait_for", "timeout"): "terminal_seconds",
}  # fmt: skip

# One distinct value per declared key, none equal to the shipped value: a call site
# bound to the wrong key, or still holding a literal, then reads a different number.
_SENTINEL: dict[str, float] = {
    "capability_probe_ttl_seconds": 601,
    "capability_probe_seconds": 7.25,
    "health_check_min_seconds": 2.5,
    "health_check_max_seconds": 61.5,
    "health_check_default_seconds": 9.5,
    "handler_shutdown_min_seconds": 3.5,
    "handler_shutdown_max_seconds": 301.5,
    "handler_shutdown_default_seconds": 13.5,
    "kafka_session_ms": 46001,
    "kafka_heartbeat_interval_ms": 15003,
    "kafka_max_poll_interval_ms": 300007,
    "snapshot_consumer_wait_ms": 5011,
    "circuit_breaker_reset_seconds": 61.25,
    "health_probe_seconds": 2.75,
    "local_dispatch_seconds": 121,
    "terminal_seconds": 0.625,
    "llm_transport_max_seconds": 123.5,
    "llm_post_seconds": 31.5,
    "introspection_base_backoff_seconds": 1.125,
    "introspection_heartbeat_interval_seconds": 33.75,
}  # fmt: skip


def _block() -> dict[str, object]:
    document = yaml.safe_load(_DECLARED.read_text(encoding="utf-8"))
    return dict(document["timeouts"])


class _KeywordSites(ast.NodeVisitor):
    """Collect (scope, called name, keyword, RUNTIME_TIMEOUTS attribute or "") per keyword argument."""

    def __init__(self) -> None:
        self._scope: list[str] = []
        self.sites: list[tuple[str, str, str, str]] = []

    def _enter(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._enter(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        for keyword in node.keywords:
            value = keyword.value
            bound = (
                value.attr
                if isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "RUNTIME_TIMEOUTS"
                else ""
            )
            if keyword.arg is not None:
                self.sites.append((".".join(self._scope), name, keyword.arg, bound))
        self.generic_visit(node)


def _keyword_sites(relative_path: str) -> list[tuple[str, str, str, str]]:
    visitor = _KeywordSites()
    visitor.visit(ast.parse((_SRC / relative_path).read_text(encoding="utf-8")))
    return visitor.sites


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
        "kafka_session_ms",
        "kafka_heartbeat_interval_ms",
        "kafka_max_poll_interval_ms",
        "snapshot_consumer_wait_ms",
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
    ("relative_path", "scope", "called", "keyword", "key"),
    [(*wire, key) for wire, key in sorted(_KEYWORD_WIRING.items())],
)
def test_keyword_call_sites_are_wired_to_their_declared_key(
    relative_path: str, scope: str, called: str, keyword: str, key: str
) -> None:
    """Every call of ``called`` with ``keyword`` in the file is the one expected site, bound to ``key``."""
    sites = [
        site for site in _keyword_sites(relative_path) if site[1:3] == (called, keyword)
    ]
    assert sites == [(scope, called, keyword, key)]


def test_the_sentinel_gives_every_declared_key_its_own_value() -> None:
    assert set(_SENTINEL) == set(_block())
    assert len(set(_SENTINEL.values())) == len(_SENTINEL)
    assert not [key for key, value in _SENTINEL.items() if value == _block()[key]]
    ModelRuntimeTimeouts.model_validate(_SENTINEL)


_SENTINEL_SCRIPT = textwrap.dedent(
    """
    import inspect
    import sys
    from pathlib import Path

    from omnibase_infra.models.runtime_timeouts import model_runtime_timeouts as m
    import omnibase_infra.models.runtime_timeouts as pkg

    sentinel = m.ModelRuntimeTimeouts.from_file(Path(sys.argv[1]))
    m._TIMEOUTS_RESOURCE = Path(sys.argv[1])
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
            sentinel.snapshot_consumer_wait_ms,
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
    """Import every call site against a file with a distinct value per key; a surviving literal or a crossed key reads a different value."""
    path = tmp_path / "runtime_timeouts.yaml"
    path.write_text(yaml.safe_dump({"timeouts": _SENTINEL}), encoding="utf-8")
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


def test_the_declared_file_loads_from_inside_a_zip(tmp_path: Path) -> None:
    """A package imported from a zipped wheel has no directory to open; the file still loads."""
    archive = tmp_path / "wheel.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(_DECLARED, "runtime_timeouts.yaml")
    loaded = ModelRuntimeTimeouts.from_file(
        zipfile.Path(archive, "runtime_timeouts.yaml")
    )
    assert loaded == RUNTIME_TIMEOUTS


def test_the_resource_beside_a_module_inside_a_zip_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_declared_resource`` walks up from a module path inside an archive to the archive."""
    archive = tmp_path / "wheel.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(_DECLARED, "pkg/runtime_timeouts.yaml")
    monkeypatch.setattr(
        model_module, "__file__", str(archive / "pkg" / "model_runtime_timeouts.py")
    )
    resource = model_module._declared_resource()
    assert isinstance(resource, zipfile.Path)
    assert ModelRuntimeTimeouts.from_file(resource) == RUNTIME_TIMEOUTS


def test_a_malformed_file_is_refused_as_a_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "runtime_timeouts.yaml"
    path.write_text("timeouts: [unclosed\n", encoding="utf-8")
    with pytest.raises(ProtocolConfigurationError):
        ModelRuntimeTimeouts.from_file(path)


def test_a_value_outside_its_bounds_in_the_file_is_a_configuration_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime_timeouts.yaml"
    path.write_text(
        yaml.safe_dump(
            {"timeouts": {**_block(), "health_check_default_seconds": 61.0}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolConfigurationError):
        ModelRuntimeTimeouts.from_file(path)


@pytest.mark.parametrize(
    "key", sorted(k for k, v in _block().items() if isinstance(v, float))
)
def test_a_boolean_is_not_a_float_timeout(key: str) -> None:
    with pytest.raises(ValidationError):
        ModelRuntimeTimeouts.model_validate({**_block(), key: True})


def test_an_llm_post_wait_above_the_transport_ceiling_fails_to_load() -> None:
    with pytest.raises(ValidationError):
        ModelRuntimeTimeouts.model_validate(
            {**_block(), "llm_post_seconds": 121.0, "llm_transport_max_seconds": 120.0}
        )
