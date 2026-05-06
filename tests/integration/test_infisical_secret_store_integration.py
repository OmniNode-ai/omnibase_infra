# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the Infisical protocol wrapper (OMN-10557)."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock
from uuid import UUID

import pytest
from pydantic import SecretStr

from omnibase_infra.adapters._internal.adapter_infisical import (
    AdapterInfisical,
    ModelInfisicalSecretResult,
)
from omnibase_infra.secret_stores.infisical_secret_store import InfisicalSecretStore
from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore


@pytest.mark.integration
async def test_infisical_secret_store_adapts_sdk_calls_to_protocol() -> None:
    adapter = MagicMock(spec=AdapterInfisical)
    type(adapter).is_authenticated = PropertyMock(return_value=True)
    adapter.get_secret.return_value = ModelInfisicalSecretResult(
        key="API_KEY",
        value=SecretStr("secret-value"),
        version=1,
        secret_path="/services/test",
        environment="dev",
    )
    adapter.list_secrets.return_value = [
        ModelInfisicalSecretResult(
            key="API_KEY",
            value=SecretStr("secret-value"),
            version=1,
            secret_path="/services/test",
            environment="dev",
        )
    ]

    project_id = UUID("00000000-0000-0000-0000-000000001557")
    store: ProtocolSecretStore = InfisicalSecretStore(
        adapter,
        project_id=project_id,
        environment_slug="dev",
        secret_path="/services/test",
    )

    assert await store.health_check() is True
    assert await store.get_secret("API_KEY") == "secret-value"
    assert await store.list_keys(prefix="API_") == ["API_KEY"]
    assert await store.set_secret("API_KEY", "new-value") is True

    adapter.get_secret.assert_called_once_with(
        secret_name="API_KEY",
        project_id=str(project_id),
        environment_slug="dev",
        secret_path="/services/test",
    )
    adapter.update_secret.assert_called_once_with(
        "API_KEY",
        "new-value",
        project_id=str(project_id),
        environment_slug="dev",
        secret_path="/services/test",
    )

    await store.close()
    adapter.shutdown.assert_called_once()
