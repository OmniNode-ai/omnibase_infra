# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerResourceManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.resources.handler_resource_manager import HandlerResourceManager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_or_create_client_creates_new_client() -> None:
    """get_or_create_client returns a new client when none exists for handler_id."""
    manager = HandlerResourceManager()
    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient"
    ) as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        client = await manager.get_or_create_client("handler-a")
    assert client is mock_client
    mock_cls.assert_called_once_with()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_or_create_client_reuses_existing_client() -> None:
    """get_or_create_client returns the same client on repeated calls for the same handler_id."""
    manager = HandlerResourceManager()
    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient"
    ) as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        client_first = await manager.get_or_create_client("handler-b")
        client_second = await manager.get_or_create_client("handler-b")
    assert client_first is client_second
    mock_cls.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_or_create_client_isolates_by_handler_id() -> None:
    """get_or_create_client creates distinct clients for distinct handler_ids."""
    manager = HandlerResourceManager()
    clients: list[AsyncMock] = [AsyncMock(), AsyncMock()]
    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient",
        side_effect=clients,
    ):
        client_a = await manager.get_or_create_client("handler-a")
        client_b = await manager.get_or_create_client("handler-b")
    assert client_a is not client_b
    assert client_a is clients[0]
    assert client_b is clients[1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_or_create_client_passes_base_url() -> None:
    """get_or_create_client forwards base_url when creating a new client."""
    manager = HandlerResourceManager()
    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient"
    ) as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        await manager.get_or_create_client("handler-c", base_url="http://example.com")
    mock_cls.assert_called_once_with(base_url="http://example.com")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_client_closes_and_removes_client() -> None:
    """release_client closes the client and removes it from the internal registry."""
    manager = HandlerResourceManager()
    mock_client = AsyncMock()
    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient",
        return_value=mock_client,
    ):
        await manager.get_or_create_client("handler-d")

    await manager.release_client("handler-d")

    mock_client.aclose.assert_awaited_once()
    assert "handler-d" not in manager._clients


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_client_noop_for_unknown_handler_id() -> None:
    """release_client is a no-op when no client exists for the given handler_id."""
    manager = HandlerResourceManager()
    # Should not raise
    await manager.release_client("nonexistent-handler")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_all_closes_all_clients() -> None:
    """shutdown_all calls aclose() on every registered client."""
    manager = HandlerResourceManager()
    clients = [AsyncMock(), AsyncMock(), AsyncMock()]
    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient",
        side_effect=clients,
    ):
        await manager.get_or_create_client("handler-x")
        await manager.get_or_create_client("handler-y")
        await manager.get_or_create_client("handler-z")

    await manager.shutdown_all()

    for mock_client in clients:
        mock_client.aclose.assert_awaited_once()
    assert manager._clients == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_all_is_idempotent() -> None:
    """shutdown_all is safe to call multiple times."""
    manager = HandlerResourceManager()
    mock_client = AsyncMock()
    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient",
        return_value=mock_client,
    ):
        await manager.get_or_create_client("handler-once")

    await manager.shutdown_all()
    await manager.shutdown_all()  # second call — no error, no double-close

    mock_client.aclose.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_all_continues_after_close_error() -> None:
    """shutdown_all closes remaining clients even if one raises during aclose()."""
    manager = HandlerResourceManager()
    client_fail = AsyncMock()
    client_fail.aclose.side_effect = Exception("close failed")
    client_ok = AsyncMock()

    with patch(
        "omnibase_infra.resources.handler_resource_manager.httpx.AsyncClient",
        side_effect=[client_fail, client_ok],
    ):
        await manager.get_or_create_client("handler-fail")
        await manager.get_or_create_client("handler-ok")

    # Should not raise — errors are logged and remaining clients are still closed
    await manager.shutdown_all()

    client_fail.aclose.assert_awaited_once()
    client_ok.aclose.assert_awaited_once()
    assert manager._clients == {}
