# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A request on a dormant contract's HTTP mount wakes it through the front door (OMN-26316).

Measured on the dev runtime at 2026-11-23T19:19:43Z: ``GET /projection-api/health``, ``/projection-api/ready`` and
``/projection-api/projections`` answered ``404 Not Found`` and no log line named
``node_projection_query_api_effect`` after boot. The contract declares an
``http_mount`` (prefix ``/projection-api``) but a lazy-start lane leaves it out of the
boot manifest, so the listener mounted nothing for it, and the request never reached
the front door.

Real aiohttp listener, real FastAPI application, real HTTP requests, the real ingress
handler over the real front door. Only the start transaction is replaced, by one that
records build and attach, so no broker or handler wiring is needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import pytest
from fastapi import FastAPI

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.handlers import handler_lazy_start_ingress as ingress_module
from omnibase_infra.handlers.handler_lazy_start_ingress import HandlerLazyStartIngress
from omnibase_infra.handlers.lazy_start_boot_split import StoreLazyStartDormant
from omnibase_infra.models.lazy_start import ModelLazyStartPolicy
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.unit

FIXTURES = "tests.unit.handlers.http_mount.test_dormant_http_wake_omn26316"
LIFESPAN_LOG: list[str] = []
BOOT = "node_boot_mount"
DORMANT = "node_dormant_mount"
CONSUMER = "node_dormant_consumer"
CONSUMER_TOPIC = "onex.cmd.test.dormant-consumer.v1"
OVERLAPPING = "node_dormant_overlap"
CALLS: list[tuple[str, str]] = []
FAIL_ATTACH: set[str] = set()


class RecordingTransaction:
    """The start transaction, recording build and attach and failing on demand."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def build(self, name: str) -> None:
        CALLS.append(("build", name))

    async def attach(self, name: str) -> None:
        await asyncio.sleep(0.05)
        if name in FAIL_ATTACH:
            FAIL_ATTACH.discard(name)
            CALLS.append(("attach_failed", name))
            raise ConnectionError("join refused")
        CALLS.append(("attach", name))

    async def rollback(self, name: str) -> None:
        CALLS.append(("rollback", name))

    async def deliver(self, name: str, topic: str, envelope: object) -> object:
        raise AssertionError("an HTTP request is never delivered as a bus envelope")


class FakeKafkaBus(EventBusKafka):
    """Kafka-typed stand-in; never opens a transport."""

    def __init__(self) -> None:
        self._environment = "local"

    async def publish(self, *args: object, **kwargs: object) -> None:
        return None


class NoWatch:
    """The dormant-topic watch is the bus test's concern, not this one's."""

    def __init__(self, _container: object) -> None:
        raise TypeError("no broker in this test")


def _app_of(prefix: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        LIFESPAN_LOG.append("startup")
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get(f"{prefix}/ping")
    async def ping() -> dict[str, str]:
        return {"source": "mounted-app", "prefix": prefix}

    return app


def create_boot_app() -> FastAPI:
    return _app_of("/boot")


def create_dormant_app() -> FastAPI:
    return _app_of("/dormant")


def create_consumer_app() -> FastAPI:
    return _app_of("/consumer")


def create_overlap_app() -> FastAPI:
    return _app_of("/boot/inner")


def _contract(tmp_path: Path, name: str, block: str) -> ModelDiscoveredContract:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"name: {name}\n"
        "node_type: EFFECT_GENERIC\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        f"{block}\n",
        encoding="utf-8",
    )
    return _parse_contract(
        contract_path=path,
        entry_point_name=name,
        package_name="fixture",
        package_version="0.0.0",
    )


def _consumer_block() -> str:
    return (
        _mount("/consumer", "create_consumer_app")
        + f'\nevent_bus:\n  subscribe_topics: ["{CONSUMER_TOPIC}"]\n  publish_topics: []'
    )


def _mount(prefix: str, factory: str) -> str:
    return f'http_mount:\n  factory: "{FIXTURES}:{factory}"\n  path_prefix: "{prefix}"'


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    CALLS.clear()
    FAIL_ATTACH.clear()
    LIFESPAN_LOG.clear()
    monkeypatch.setattr(
        ingress_module, "LazyStartRuntimeTransaction", RecordingTransaction
    )
    monkeypatch.setattr(
        ingress_module, "LazyStartDormantTopicWatch", NoWatch, raising=False
    )
    yield
    StoreLazyStartDormant.clear()


async def _booted(
    tmp_path: Path, *dormant_blocks: tuple[str, str]
) -> tuple[ServiceHealth, HandlerLazyStartIngress, int]:
    """Boot as the kernel does: the boot set is mounted, the rest is recorded dormant."""
    boot = _contract(tmp_path, BOOT, _mount("/boot", "create_boot_app"))
    dormant = tuple(_contract(tmp_path, n, b) for n, b in dormant_blocks)
    StoreLazyStartDormant.record(
        ModelLazyStartPolicy(
            ingress_contract="node_lazy_start_ingress", boot_contracts=(BOOT,)
        ),
        dormant,
        None,
        (boot,),
    )
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, FakeKafkaBus())
    handler = HandlerLazyStartIngress(container)
    server = ServiceHealth(runtime=MagicMock(), port=0, host="127.0.0.1")
    await server.start()
    server.attach_manifest(ModelAutoWiringManifest(contracts=(boot,)))
    assert server._mount_task is not None
    await server._mount_task
    index = server._http_mounts._dormant_index_task
    assert index is not None
    await index
    assert server._site is not None
    return server, handler, int(server._site.port)


async def _get(port: int, path: str) -> tuple[int, str]:
    async with (
        aiohttp.ClientSession() as session,
        session.get(f"http://127.0.0.1:{port}{path}") as response,
    ):
        return response.status, await response.text()


async def test_first_request_on_a_dormant_mount_is_served_by_a_contract_with_no_subscription(
    tmp_path: Path,
) -> None:
    """The projection-api shape: an http_mount and no event_bus, so nothing to start."""
    server, handler, port = await _booted(
        tmp_path, (DORMANT, _mount("/dormant", "create_dormant_app"))
    )
    try:
        assert LIFESPAN_LOG == ["startup"]
        status, body = await _get(port, "/dormant/ping")
        assert (status, '"mounted-app"' in body) == (200, True), (status, body)
        # Its lifespan ran once, beside the boot mount's; the transaction never ran,
        # because the wiring skips a contract that subscribes to no topic.
        assert LIFESPAN_LOG == ["startup", "startup"]
        assert CALLS == []
        front_door = handler._front_door
        assert front_door is not None
        assert front_door.serving == {DORMANT}
        status, _ = await _get(port, "/dormant/ping")
        assert status == 200
        assert LIFESPAN_LOG == ["startup", "startup"]
    finally:
        await server.stop()


async def test_first_request_on_a_dormant_mount_starts_a_contract_that_subscribes(
    tmp_path: Path,
) -> None:
    server, _, port = await _booted(tmp_path, (CONSUMER, _consumer_block()))
    try:
        assert CALLS == []
        status, body = await _get(port, "/consumer/ping")
        assert (status, '"mounted-app"' in body) == (200, True), (status, body)
        assert CALLS == [("build", CONSUMER), ("attach", CONSUMER)]
        status, _ = await _get(port, "/consumer/ping")
        assert status == 200
        assert CALLS == [("build", CONSUMER), ("attach", CONSUMER)]
    finally:
        await server.stop()


async def test_positive_control_boot_mount_and_unrelated_paths_wake_nothing(
    tmp_path: Path,
) -> None:
    server, _, port = await _booted(
        tmp_path,
        (DORMANT, _mount("/dormant", "create_dormant_app")),
        (CONSUMER, _consumer_block()),
    )
    try:
        assert (await _get(port, "/boot/ping"))[0] == 200
        assert (await _get(port, "/v1/introspection/manifest"))[0] == 200
        assert (await _get(port, "/dormantly/ping"))[0] == 404
        assert (await _get(port, "/nowhere"))[0] == 404
        assert CALLS == []
        assert LIFESPAN_LOG == ["startup"]
    finally:
        await server.stop()


async def test_concurrent_first_requests_start_the_contract_once(
    tmp_path: Path,
) -> None:
    server, _, port = await _booted(tmp_path, (CONSUMER, _consumer_block()))
    try:
        answers = await asyncio.gather(
            *(_get(port, "/consumer/ping") for _ in range(8))
        )
        assert [status for status, _ in answers] == [200] * 8
        assert CALLS == [("build", CONSUMER), ("attach", CONSUMER)]
        assert LIFESPAN_LOG == ["startup", "startup"]
    finally:
        await server.stop()


async def test_a_start_that_fails_once_is_retried_inside_the_wake_and_served(
    tmp_path: Path,
) -> None:
    FAIL_ATTACH.add(CONSUMER)
    server, _, port = await _booted(tmp_path, (CONSUMER, _consumer_block()))
    try:
        # The policy's retry budget: the front door retries the start inside one wake.
        status, body = await _get(port, "/consumer/ping")
        assert status == 200, (status, body)
        assert CALLS.count(("attach_failed", CONSUMER)) == 1
        assert ("attach", CONSUMER) in CALLS
    finally:
        await server.stop()


async def test_a_wake_that_raises_leaves_the_mount_dormant_and_the_server_up(
    tmp_path: Path,
) -> None:
    server, _, port = await _booted(
        tmp_path, (DORMANT, _mount("/dormant", "create_dormant_app"))
    )

    async def refuse(_name: str) -> None:
        raise ConnectionError("join refused for the node")

    StoreLazyStartDormant.record_waker(refuse)
    try:
        status, body = await _get(port, "/dormant/ping")
        assert status == 503
        assert '"unavailable"' in body
        # One contract's failed wake never takes the listener out of service.
        assert (await _get(port, "/boot/ping"))[0] == 200
        assert (await _get(port, "/v1/introspection/manifest"))[0] == 200
    finally:
        await server.stop()


async def test_a_dormant_mount_that_overlaps_the_boot_set_is_never_served(
    tmp_path: Path,
) -> None:
    server, _, port = await _booted(
        tmp_path,
        (OVERLAPPING, _mount("/boot/inner", "create_overlap_app")),
        (DORMANT, _mount("/dormant", "create_dormant_app")),
    )
    try:
        await _get(port, "/boot/inner/ping")
        assert LIFESPAN_LOG == ["startup"]
        assert (await _get(port, "/dormant/ping"))[0] == 200
        assert LIFESPAN_LOG == ["startup", "startup"]
    finally:
        await server.stop()
