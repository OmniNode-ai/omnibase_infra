# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A mount factory receives the dependencies its contract block declares (OMN-21894)."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from pydantic import ValidationError

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.http_mount import AsgiMounts
from omnibase_infra.models.model_http_mount import ModelHttpMount
from tests.unit.handlers.http_mount import asgi_fixtures
from tests.unit.handlers.http_mount.test_asgi_mounts_omn21288 import (
    FIXTURES,
    _manifest,
    _mounted,
    _server,
)

pytestmark = pytest.mark.unit


def _block(dependencies: str) -> str:
    return (
        f'http_mount:\n  factory: "{FIXTURES}:create_app_with_dependencies"\n'
        f'  path_prefix: "/dep"\n  dependencies: {dependencies}'
    )


def test_model_defaults_to_no_dependencies() -> None:
    mount = ModelHttpMount(factory="a.b:c", path_prefix="/x")
    assert mount.dependencies == ()


@pytest.mark.parametrize("bad", [["Bad-Name"], ["bus", "bus"], [""]])
def test_model_refuses_malformed_or_repeated_dependency_names(bad: list[str]) -> None:
    with pytest.raises(ValidationError):
        ModelHttpMount(factory="a.b:c", path_prefix="/x", dependencies=tuple(bad))


def test_factory_receives_declared_dependencies(tmp_path: Path) -> None:
    asgi_fixtures.DEPENDENCY_LOG.clear()
    bus, reader = object(), object()
    mounts = AsgiMounts()
    mounts.provide_dependency("bus", bus)
    mounts.provide_dependency("reader", reader)
    manifest = _manifest(tmp_path, _block("[bus, reader]"))
    assert mounts.prepare(manifest.contracts, web.UrlDispatcher())
    assert [{"bus": bus, "reader": reader}] == asgi_fixtures.DEPENDENCY_LOG


def test_missing_dependency_refuses_the_mount_by_name(tmp_path: Path) -> None:
    asgi_fixtures.DEPENDENCY_LOG.clear()
    mounts = AsgiMounts()
    mounts.provide_dependency("bus", object())
    manifest = _manifest(tmp_path, _block("[bus, reader]"))
    with pytest.raises(RuntimeHostError, match="reader"):
        mounts.prepare(manifest.contracts, web.UrlDispatcher())
    assert asgi_fixtures.DEPENDENCY_LOG == []


def test_undeclared_provided_dependency_is_not_passed(tmp_path: Path) -> None:
    """A provided value the block does not name never reaches the factory."""
    mounts = AsgiMounts()
    mounts.provide_dependency("bus", object())
    mounts.provide_dependency("reader", object())
    block = _block("[bus]")
    manifest = _manifest(tmp_path, block)
    with pytest.raises(RuntimeHostError, match="factory raised"):
        mounts.prepare(manifest.contracts, web.UrlDispatcher())


async def test_runtime_listener_hands_provided_dependencies_to_the_mount(
    tmp_path: Path,
) -> None:
    asgi_fixtures.DEPENDENCY_LOG.clear()
    bus, reader = object(), object()
    server = await _server()
    try:
        server._http_mounts.provide_dependency("bus", bus)
        server._http_mounts.provide_dependency("reader", reader)
        await _mounted(server, _manifest(tmp_path, _block("[bus, reader]")))
        assert [{"bus": bus, "reader": reader}] == asgi_fixtures.DEPENDENCY_LOG
    finally:
        await server.stop()
