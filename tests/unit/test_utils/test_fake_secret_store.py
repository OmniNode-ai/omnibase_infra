# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``FakeSecretStore`` (OMN-10559)."""

from __future__ import annotations

import pytest

from omnibase_infra.test_utils.fake_secret_store import FakeSecretStore
from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore


@pytest.mark.unit
def test_satisfies_protocol_secret_store() -> None:
    assert isinstance(FakeSecretStore(), ProtocolSecretStore)


@pytest.mark.unit
async def test_empty_constructor_yields_empty_store() -> None:
    store = FakeSecretStore()
    assert await store.list_keys() == []
    assert await store.get_secret("anything") is None


@pytest.mark.unit
async def test_initial_data_preloaded() -> None:
    store = FakeSecretStore(initial={"a": "1", "b": "2"})
    assert await store.get_secret("a") == "1"
    assert await store.get_secret("b") == "2"


@pytest.mark.unit
async def test_initial_data_is_copied_not_referenced() -> None:
    source = {"a": "1"}
    store = FakeSecretStore(initial=source)
    source["a"] = "mutated"
    assert await store.get_secret("a") == "1"


@pytest.mark.unit
async def test_set_then_get_round_trip() -> None:
    store = FakeSecretStore()
    assert await store.set_secret("key", "value") is True
    assert await store.get_secret("key") == "value"


@pytest.mark.unit
async def test_set_overwrites_existing_value() -> None:
    store = FakeSecretStore(initial={"key": "original"})
    await store.set_secret("key", "updated")
    assert await store.get_secret("key") == "updated"


@pytest.mark.unit
async def test_delete_existing_returns_true_and_removes() -> None:
    store = FakeSecretStore(initial={"key": "value"})
    assert await store.delete_secret("key") is True
    assert await store.get_secret("key") is None


@pytest.mark.unit
async def test_delete_missing_returns_false() -> None:
    store = FakeSecretStore()
    assert await store.delete_secret("absent") is False


@pytest.mark.unit
async def test_list_keys_no_prefix_returns_all_sorted() -> None:
    store = FakeSecretStore(initial={"b": "2", "a": "1", "c": "3"})
    assert await store.list_keys() == ["a", "b", "c"]


@pytest.mark.unit
async def test_list_keys_filters_by_prefix() -> None:
    store = FakeSecretStore(
        initial={"llm/openai": "k1", "db/pg": "k2", "llm/anthropic": "k3"}
    )
    assert await store.list_keys(prefix="llm/") == ["llm/anthropic", "llm/openai"]


@pytest.mark.unit
async def test_health_check_true_when_open() -> None:
    store = FakeSecretStore()
    assert await store.health_check() is True


@pytest.mark.unit
async def test_health_check_false_after_close() -> None:
    store = FakeSecretStore()
    await store.close()
    assert await store.health_check() is False


@pytest.mark.unit
async def test_close_accepts_default_timeout() -> None:
    store = FakeSecretStore()
    await store.close()
    await store.close(timeout_seconds=5.0)
