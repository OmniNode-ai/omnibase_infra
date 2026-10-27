# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Owned transport clients and materialized clients share HTTP construction."""

from __future__ import annotations

import httpx
import pytest

from omnibase_infra.runtime.models.model_http_client_config import ModelHttpClientConfig
from omnibase_infra.runtime.providers.provider_http_client import ProviderHttpClient

pytestmark = pytest.mark.unit


async def test_owned_transport_client_keeps_subsecond_timeout_and_no_redirects() -> (
    None
):
    client = ProviderHttpClient.create_client(timeout_seconds=0.25)
    try:
        assert client.timeout == httpx.Timeout(0.25)
        assert client.follow_redirects is False
    finally:
        await ProviderHttpClient.close(client)
    assert client.is_closed


@pytest.mark.parametrize("follow_redirects", [False, True])
async def test_materialized_client_uses_configured_timeout_and_redirects(
    follow_redirects: bool,
) -> None:
    provider = ProviderHttpClient(
        ModelHttpClientConfig(timeout_seconds=7.0, follow_redirects=follow_redirects)
    )
    client = await provider.create()
    try:
        assert client.timeout == httpx.Timeout(7.0)
        assert client.follow_redirects is follow_redirects
    finally:
        await provider.close(client)
    assert client.is_closed
