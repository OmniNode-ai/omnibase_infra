# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the S6 kernel glue (OMN-14758): handler resolution parity (R-7),
fail-closed DLQ provisioning (R-6), and the readiness snapshot (§c.5/§d)."""

from __future__ import annotations

import asyncio

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.dispatch.model_handler_ref import ModelHandlerRef
from omnibase_infra.runtime.core_runtime.composition import CoreRuntimeHandle
from omnibase_infra.runtime.core_runtime.kernel_glue import (
    _construct_handler,
    _provision_dlq_topics,
    build_kernel_handler_resolver,
)
from omnibase_infra.runtime.core_runtime.phantom_alarm import PhantomAlarmMonitor

# ---------------------------------------------------------------------------
# Module-level handler classes (importable via ``module=__name__``).
# ---------------------------------------------------------------------------


class HandlerNoArg:
    """def-B handler with a no-arg constructor."""

    def handle(self, request: object) -> object:  # pragma: no cover - not invoked
        return request


class HandlerContainer:
    """def-B handler taking the ONEX container."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self.container = container

    def handle(self, request: object) -> object:  # pragma: no cover - not invoked
        return request


class HandlerBadSignature:
    """def-B handler with a required non-container argument (unresolvable)."""

    def __init__(self, dependency: object) -> None:
        self.dependency = dependency

    def handle(self, request: object) -> object:  # pragma: no cover - not invoked
        return request


class NotAHandler:
    """No ``handle`` — must be rejected as not-def-B."""


def _ref(name: str) -> ModelHandlerRef:
    return ModelHandlerRef(name=name, module=__name__)


# ---------------------------------------------------------------------------
# _construct_handler / resolver (R-7 + MINOR fail-closed construction).
# ---------------------------------------------------------------------------


def test_construct_handler_no_arg() -> None:
    container = ModelONEXContainer()
    instance = _construct_handler(HandlerNoArg, container)
    assert isinstance(instance, HandlerNoArg)


def test_construct_handler_binds_container_by_name() -> None:
    container = ModelONEXContainer()
    instance = _construct_handler(HandlerContainer, container)
    assert isinstance(instance, HandlerContainer)
    assert instance.container is container


def test_construct_handler_fails_closed_on_unknown_signature() -> None:
    container = ModelONEXContainer()
    with pytest.raises(ModelOnexError, match="unsupported constructor signature"):
        _construct_handler(HandlerBadSignature, container)


def test_resolver_rejects_non_def_b_class() -> None:
    resolver = build_kernel_handler_resolver(ModelONEXContainer())
    with pytest.raises(ModelOnexError, match="not a def-B handler"):
        resolver(_ref("NotAHandler"))


def test_resolver_constructs_when_no_shared_instance() -> None:
    resolver = build_kernel_handler_resolver(ModelONEXContainer())
    instance = resolver(_ref("HandlerContainer"))
    assert isinstance(instance, HandlerContainer)


async def test_resolver_reuses_shared_instance_from_registry() -> None:
    container = ModelONEXContainer()
    shared = HandlerContainer(container)
    assert container.service_registry is not None
    await container.service_registry.register_instance(HandlerContainer, shared)

    resolver = build_kernel_handler_resolver(container)
    resolved = resolver(_ref("HandlerContainer"))
    # R-7 parity: the SAME instance the legacy path wired is returned, not a fresh build.
    assert resolved is shared


# ---------------------------------------------------------------------------
# _provision_dlq_topics (R-6 fail-closed DLQ provisioning before loop start).
# ---------------------------------------------------------------------------


class _RecordingProvisioner:
    def __init__(self, *, result: bool = True) -> None:
        self.result = result
        self.created: list[str] = []

    async def ensure_topic_exists(
        self, topic_name: str, spec: object = None, correlation_id: object = None
    ) -> bool:
        self.created.append(topic_name)
        return self.result


async def test_provision_dlq_topics_creates_each() -> None:
    provisioner = _RecordingProvisioner()
    topics = frozenset({"onex.dlq.omnibase-infra.a.v1", "onex.dlq.omnibase-infra.b.v1"})
    await _provision_dlq_topics(topics, provisioner=provisioner, correlation_id=None)
    assert sorted(provisioner.created) == sorted(topics)


async def test_provision_dlq_topics_empty_is_noop() -> None:
    provisioner = _RecordingProvisioner()
    await _provision_dlq_topics(
        frozenset(), provisioner=provisioner, correlation_id=None
    )
    assert provisioner.created == []


async def test_provision_dlq_topics_fails_closed_without_provisioner() -> None:
    with pytest.raises(ModelOnexError, match="no topic provisioner is available"):
        await _provision_dlq_topics(
            frozenset({"onex.dlq.omnibase-infra.a.v1"}),
            provisioner=None,
            correlation_id=None,
        )


async def test_provision_dlq_topics_fails_closed_on_creation_failure() -> None:
    provisioner = _RecordingProvisioner(result=False)
    with pytest.raises(ModelOnexError, match="failed to provision DLQ topic"):
        await _provision_dlq_topics(
            frozenset({"onex.dlq.omnibase-infra.a.v1"}),
            provisioner=provisioner,
            correlation_id=None,
        )


# ---------------------------------------------------------------------------
# CoreRuntimeHandle.readiness_snapshot (§c.5 loop-health + §d phantom).
# ---------------------------------------------------------------------------


class _NoopConsumer:
    async def start(self) -> None: ...
    async def close(self) -> None: ...

    async def poll(self, *, max_messages: int, timeout_ms: int):
        return []

    async def commit(self, message: object) -> None: ...
    async def nack(self, message: object) -> None: ...


def _handle() -> CoreRuntimeHandle:
    monitor = PhantomAlarmMonitor(
        _NoopConsumer(), core_runtime_topics=frozenset({"onex.cmd.x.v1"})
    )
    return CoreRuntimeHandle(
        dispatch=object(),  # type: ignore[arg-type]
        transport=object(),
        monitor=monitor,
        dlq_provision_topics=frozenset(),
        core_runtime_topics=frozenset({"onex.cmd.x.v1"}),
    )


def test_readiness_snapshot_not_ready_before_start() -> None:
    handle = _handle()
    ready, detail = handle.readiness_snapshot()
    assert ready is False
    assert detail["loop_healthy"] is False


async def test_readiness_snapshot_ready_when_loop_live() -> None:
    handle = _handle()

    async def _live() -> None:
        await asyncio.sleep(10)

    handle._task = asyncio.create_task(_live())
    try:
        ready, detail = handle.readiness_snapshot()
        assert ready is True
        assert detail["loop_healthy"] is True
        assert detail["phantom_subscription_topics"] == []
    finally:
        handle._task.cancel()


async def test_readiness_snapshot_fails_on_phantom() -> None:
    handle = _handle()

    async def _live() -> None:
        await asyncio.sleep(10)

    handle._task = asyncio.create_task(_live())
    handle._last_phantom_topics = frozenset({"onex.cmd.x.v1"})
    try:
        ready, detail = handle.readiness_snapshot()
        assert ready is False
        assert detail["phantom_subscription_topics"] == ["onex.cmd.x.v1"]
    finally:
        handle._task.cancel()
