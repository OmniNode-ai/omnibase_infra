# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the env-backed secret store (OMN-10558)."""

from __future__ import annotations

import pytest

from omnibase_infra.secret_stores.adapter_env_secret_store import (
    AdapterEnvSecretStore,
)
from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore


@pytest.mark.integration
async def test_adapter_env_secret_store_protocol_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMN_INT_SECRET_A", "alpha")
    monkeypatch.setenv("OMN_INT_SECRET_B", "bravo")
    monkeypatch.delenv("OMN_INT_SECRET_MISSING", raising=False)

    store: ProtocolSecretStore = AdapterEnvSecretStore()

    assert await store.health_check() is True
    assert await store.get_secret("OMN_INT_SECRET_A") == "alpha"
    assert await store.get_secret("OMN_INT_SECRET_MISSING") is None
    assert sorted(await store.list_keys(prefix="OMN_INT_SECRET_")) == [
        "OMN_INT_SECRET_A",
        "OMN_INT_SECRET_B",
    ]

    with pytest.raises(RuntimeError, match="read-only"):
        await store.set_secret("OMN_INT_SECRET_C", "charlie")

    await store.close()
