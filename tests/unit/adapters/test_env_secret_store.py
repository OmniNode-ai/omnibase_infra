# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for AdapterEnvSecretStore."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.adapter_env_secret_store import AdapterEnvSecretStore


@pytest.fixture
def store() -> AdapterEnvSecretStore:
    return AdapterEnvSecretStore()


class TestAdapterEnvSecretStoreGet:
    @pytest.mark.asyncio
    async def test_get_missing_key_returns_none(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("_ONEX_TEST_KEY_MISSING", raising=False)
        result = await store.get_secret("_ONEX_TEST_KEY_MISSING")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_present_key_returns_value(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("_ONEX_TEST_KEY", "secret-value")
        result = await store.get_secret("_ONEX_TEST_KEY")
        assert result == "secret-value"


class TestAdapterEnvSecretStoreSet:
    @pytest.mark.asyncio
    async def test_set_and_get_roundtrip(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("_ONEX_SET_KEY", raising=False)
        success = await store.set_secret("_ONEX_SET_KEY", "my-value")
        assert success is True
        result = await store.get_secret("_ONEX_SET_KEY")
        assert result == "my-value"


class TestAdapterEnvSecretStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_existing_key_returns_true(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("_ONEX_DEL_KEY", "val")
        result = await store.delete_secret("_ONEX_DEL_KEY")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_missing_key_returns_false(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("_ONEX_DEL_MISSING", raising=False)
        result = await store.delete_secret("_ONEX_DEL_MISSING")
        assert result is False


class TestAdapterEnvSecretStoreListKeys:
    @pytest.mark.asyncio
    async def test_list_keys_no_prefix_returns_all(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("_ONEX_LIST_A", "1")
        keys = await store.list_keys()
        assert "_ONEX_LIST_A" in keys

    @pytest.mark.asyncio
    async def test_list_keys_prefix_filters(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("_ONEX_PFXA_1", "1")
        monkeypatch.setenv("_ONEX_PFXA_2", "2")
        monkeypatch.setenv("_ONEX_OTHER", "3")
        keys = await store.list_keys(prefix="_ONEX_PFXA")
        assert "_ONEX_PFXA_1" in keys
        assert "_ONEX_PFXA_2" in keys
        assert "_ONEX_OTHER" not in keys

    @pytest.mark.asyncio
    async def test_list_keys_prefix_no_match_returns_empty(
        self, store: AdapterEnvSecretStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keys = await store.list_keys(prefix="__ONEX_NO_MATCH_XYZ__")
        assert keys == []


class TestAdapterEnvSecretStoreHealthAndClose:
    @pytest.mark.asyncio
    async def test_health_check_always_true(self, store: AdapterEnvSecretStore) -> None:
        assert await store.health_check() is True

    @pytest.mark.asyncio
    async def test_close_is_noop(self, store: AdapterEnvSecretStore) -> None:
        await store.close()
