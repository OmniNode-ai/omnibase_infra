# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``InfisicalSecretStore`` (OMN-10557).

Mocks ``AdapterInfisical`` so the suite is hermetic; the wrapper's job is
to translate the protocol surface to the SDK-shaped adapter calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock
from uuid import UUID

import pytest
from pydantic import SecretStr

from omnibase_infra.adapters._internal.adapter_infisical import (
    AdapterInfisical,
    ModelInfisicalSecretResult,
)
from omnibase_infra.errors import InfraConnectionError, SecretResolutionError
from omnibase_infra.secret_stores.infisical_secret_store import InfisicalSecretStore
from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore

_PROJECT_ID = UUID("00000000-0000-0000-0000-000000000123")
_ENV = "dev"
_PATH = "/services/omnimarket/llm"


def _make_store(adapter: MagicMock) -> InfisicalSecretStore:
    return InfisicalSecretStore(
        adapter,
        project_id=_PROJECT_ID,
        environment_slug=_ENV,
        secret_path=_PATH,
    )


def _result(key: str, value: str) -> ModelInfisicalSecretResult:
    return ModelInfisicalSecretResult(
        key=key,
        value=SecretStr(value),
        version=1,
        secret_path=_PATH,
        environment=_ENV,
    )


@pytest.mark.unit
def test_satisfies_protocol_secret_store() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    assert isinstance(_make_store(adapter), ProtocolSecretStore)


@pytest.mark.unit
async def test_get_secret_returns_value_on_hit() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    adapter.get_secret.return_value = _result("API_KEY", "topsecret")

    store = _make_store(adapter)
    assert await store.get_secret("API_KEY") == "topsecret"

    adapter.get_secret.assert_called_once_with(
        secret_name="API_KEY",
        project_id=str(_PROJECT_ID),
        environment_slug=_ENV,
        secret_path=_PATH,
    )


@pytest.mark.unit
async def test_get_secret_returns_none_on_miss() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    adapter.get_secret.side_effect = SecretResolutionError("not found")

    store = _make_store(adapter)
    assert await store.get_secret("MISSING") is None


@pytest.mark.unit
async def test_set_secret_uses_update_when_key_exists() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    adapter.update_secret.return_value = None

    store = _make_store(adapter)
    assert await store.set_secret("API_KEY", "newval") is True

    adapter.update_secret.assert_called_once_with(
        "API_KEY",
        "newval",
        project_id=str(_PROJECT_ID),
        environment_slug=_ENV,
        secret_path=_PATH,
    )
    adapter.create_secret.assert_not_called()


@pytest.mark.unit
async def test_set_secret_falls_back_to_create_when_update_fails() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    adapter.update_secret.side_effect = SecretResolutionError("not found")
    adapter.create_secret.return_value = None

    store = _make_store(adapter)
    assert await store.set_secret("NEW_KEY", "val") is True

    adapter.update_secret.assert_called_once()
    adapter.create_secret.assert_called_once_with(
        "NEW_KEY",
        "val",
        project_id=str(_PROJECT_ID),
        environment_slug=_ENV,
        secret_path=_PATH,
    )


@pytest.mark.unit
async def test_set_secret_falls_back_to_create_on_connection_error() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    adapter.update_secret.side_effect = InfraConnectionError("update failed")
    adapter.create_secret.return_value = None

    store = _make_store(adapter)
    assert await store.set_secret("NEW_KEY", "val") is True

    adapter.create_secret.assert_called_once()


@pytest.mark.unit
async def test_delete_secret_raises_runtime_error() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    store = _make_store(adapter)
    with pytest.raises(RuntimeError, match="OMN-2286"):
        await store.delete_secret("ANY")


@pytest.mark.unit
async def test_list_keys_no_prefix_returns_all() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    adapter.list_secrets.return_value = [
        _result("LLM_CODER_URL", "x"),
        _result("LLM_CODER_MODEL_ID", "y"),
        _result("DB_DSN", "z"),
    ]
    store = _make_store(adapter)

    keys = await store.list_keys()
    assert keys == ["LLM_CODER_URL", "LLM_CODER_MODEL_ID", "DB_DSN"]


@pytest.mark.unit
async def test_list_keys_filters_by_prefix() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    adapter.list_secrets.return_value = [
        _result("LLM_CODER_URL", "x"),
        _result("LLM_CODER_MODEL_ID", "y"),
        _result("DB_DSN", "z"),
    ]
    store = _make_store(adapter)

    keys = await store.list_keys(prefix="LLM_")
    assert keys == ["LLM_CODER_URL", "LLM_CODER_MODEL_ID"]


@pytest.mark.unit
async def test_health_check_reflects_adapter_authentication_state() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    type(adapter).is_authenticated = PropertyMock(return_value=True)
    store = _make_store(adapter)
    assert await store.health_check() is True


@pytest.mark.unit
async def test_health_check_returns_false_when_not_authenticated() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    type(adapter).is_authenticated = PropertyMock(return_value=False)
    store = _make_store(adapter)
    assert await store.health_check() is False


@pytest.mark.unit
async def test_close_calls_adapter_shutdown() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    store = _make_store(adapter)
    await store.close()
    adapter.shutdown.assert_called_once()


@pytest.mark.unit
async def test_close_accepts_timeout_for_protocol_compatibility() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    store = _make_store(adapter)
    await store.close(timeout_seconds=5.0)
    adapter.shutdown.assert_called_once()


@pytest.mark.unit
def test_adapter_source_unchanged_uses_sdk_shaped_kwargs() -> None:
    """Sanity: the wrapper does not require modifying AdapterInfisical."""
    import inspect

    sig = inspect.signature(AdapterInfisical.get_secret)
    assert "secret_name" in sig.parameters
    assert "project_id" in sig.parameters
    assert "environment_slug" in sig.parameters
    assert "secret_path" in sig.parameters
