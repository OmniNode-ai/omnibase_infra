# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``AdapterEnvSecretStore`` (OMN-10558).

Asserts the env-backed implementation honours the OMN-10556
``ProtocolSecretStore`` semantics: nullable lookup, read-only writes,
prefix listing.
"""

from __future__ import annotations

import pytest

from omnibase_infra.secret_stores.adapter_env_secret_store import (
    AdapterEnvSecretStore,
)
from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore


@pytest.mark.unit
def test_satisfies_protocol_secret_store() -> None:
    assert isinstance(AdapterEnvSecretStore(), ProtocolSecretStore)


@pytest.mark.unit
async def test_get_secret_returns_value_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMN_TEST_SECRET", "hunter2")
    store = AdapterEnvSecretStore()
    assert await store.get_secret("OMN_TEST_SECRET") == "hunter2"


@pytest.mark.unit
async def test_get_secret_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMN_TEST_ABSENT_KEY", raising=False)
    store = AdapterEnvSecretStore()
    assert await store.get_secret("OMN_TEST_ABSENT_KEY") is None


@pytest.mark.unit
async def test_set_secret_raises_runtime_error() -> None:
    store = AdapterEnvSecretStore()
    with pytest.raises(RuntimeError, match="read-only"):
        await store.set_secret("OMN_TEST_SECRET", "value")


@pytest.mark.unit
async def test_delete_secret_raises_runtime_error() -> None:
    store = AdapterEnvSecretStore()
    with pytest.raises(RuntimeError, match="read-only"):
        await store.delete_secret("OMN_TEST_SECRET")


@pytest.mark.unit
async def test_list_keys_no_prefix_returns_all_env_var_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMN_TEST_AAA", "1")
    monkeypatch.setenv("OMN_TEST_BBB", "2")
    store = AdapterEnvSecretStore()

    keys = await store.list_keys()
    assert "OMN_TEST_AAA" in keys
    assert "OMN_TEST_BBB" in keys


@pytest.mark.unit
async def test_list_keys_filters_by_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMN_T12_FOO", "1")
    monkeypatch.setenv("OMN_T12_BAR", "2")
    monkeypatch.setenv("UNRELATED_KEY", "3")
    store = AdapterEnvSecretStore()

    keys = await store.list_keys(prefix="OMN_T12_")
    assert sorted(keys) == ["OMN_T12_BAR", "OMN_T12_FOO"]


@pytest.mark.unit
async def test_health_check_always_true() -> None:
    store = AdapterEnvSecretStore()
    assert await store.health_check() is True


@pytest.mark.unit
async def test_close_is_noop() -> None:
    store = AdapterEnvSecretStore()
    await store.close()
    await store.close(timeout_seconds=5.0)
