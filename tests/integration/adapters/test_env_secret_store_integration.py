# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the env-backed ProtocolSecretStore adapter."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.adapter_env_secret_store import AdapterEnvSecretStore
from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_env_secret_store_protocol_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "_ONEX_INTEGRATION_ENV_SECRET_STORE"
    monkeypatch.delenv(key, raising=False)

    store: ProtocolSecretStore = AdapterEnvSecretStore()

    assert await store.get_secret(key) is None
    assert await store.set_secret(key, "integration-value") is True
    assert await store.get_secret(key) == "integration-value"
    assert key in await store.list_keys(prefix="_ONEX_INTEGRATION_")
    assert await store.delete_secret(key) is True
    assert await store.get_secret(key) is None
    assert await store.health_check() is True
    await store.close()
