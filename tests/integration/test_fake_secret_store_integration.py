# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the fake secret store test utility (OMN-10559)."""

from __future__ import annotations

import pytest

from omnibase_infra.test_utils.fake_secret_store import FakeSecretStore
from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore


@pytest.mark.integration
async def test_fake_secret_store_protocol_lifecycle() -> None:
    store: ProtocolSecretStore = FakeSecretStore(initial={"db/url": "postgres"})

    assert await store.health_check() is True
    assert await store.get_secret("db/url") == "postgres"
    assert await store.set_secret("llm/key", "secret") is True
    assert await store.get_secret("llm/key") == "secret"
    assert await store.list_keys(prefix="llm/") == ["llm/key"]
    assert await store.delete_secret("db/url") is True
    assert await store.get_secret("db/url") is None

    await store.close()
    assert await store.health_check() is False
