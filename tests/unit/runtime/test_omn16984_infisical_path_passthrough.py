# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16984: an Infisical mapping's FOLDER must reach the handler.

``_read_infisical_secret_sync`` / ``_read_infisical_secret_async`` split a
mapping's ``source_path`` on ``/`` and kept only the last segment as the secret
key -- the folder was computed and then discarded. Every Infisical read
therefore landed in whatever ``secret_path`` the handler happened to be
configured with, so two mappings in different folders were indistinguishable
and a path-qualified mapping was silently misaddressed.

There were zero ``source_type: infisical`` mappings anywhere in the platform
when this was found, so nothing depended on the discard. The BYOK read path
needs it: house provider keys live in ``/dev/onex-runtime`` and tenant-minted
credentials in ``/tenant-inference-credentials`` on the same lane.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr

from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec
from omnibase_infra.runtime.secret_resolver import SecretResolver


class _RecordingInfisicalHandler:
    """Records the addressing arguments each read is made with."""

    def __init__(self, secrets: dict[tuple[str, str | None], str]) -> None:
        self._secrets = secrets
        self.sync_calls: list[dict[str, Any]] = []
        self.async_calls: list[dict[str, Any]] = []

    def get_secret_sync(
        self,
        *,
        secret_name: str,
        project_id: str | None = None,
        environment_slug: str | None = None,
        secret_path: str | None = None,
    ) -> SecretStr | None:
        self.sync_calls.append({"secret_name": secret_name, "secret_path": secret_path})
        value = self._secrets.get((secret_name, secret_path))
        return SecretStr(value) if value is not None else None

    async def execute(self, envelope: dict[str, Any]) -> Any:
        payload = envelope["payload"]
        self.async_calls.append(dict(payload))
        value = self._secrets.get((payload["secret_name"], payload.get("secret_path")))

        class _Result:
            result = {"value": value} if value is not None else {}

        return _Result()


def _config(logical_name: str, source_path: str) -> ModelSecretResolverConfig:
    return ModelSecretResolverConfig(
        enable_convention_fallback=False,
        mappings=[
            ModelSecretMapping(
                logical_name=logical_name,
                source=ModelSecretSourceSpec(
                    source_type="infisical", source_path=source_path
                ),
            )
        ],
    )


def test_sync_read_carries_the_mapping_folder() -> None:
    handler = _RecordingInfisicalHandler(
        {("LLM_GLM_API_KEY", "/dev/onex-runtime"): "value-from-the-right-folder"}
    )
    resolver = SecretResolver(
        config=_config("llm.glm.api_key", "/dev/onex-runtime/LLM_GLM_API_KEY"),
        infisical_handler=handler,  # type: ignore[arg-type]
    )

    resolved = resolver.get_secret("llm.glm.api_key", required=False)

    assert resolved is not None
    assert resolved.get_secret_value() == "value-from-the-right-folder"
    assert handler.sync_calls == [
        {"secret_name": "LLM_GLM_API_KEY", "secret_path": "/dev/onex-runtime"}
    ]


async def test_async_read_carries_the_mapping_folder() -> None:
    handler = _RecordingInfisicalHandler(
        {("CRED_X", "/tenant-inference-credentials"): "tenant-scoped-value"}
    )
    resolver = SecretResolver(
        config=_config("cred.x", "/tenant-inference-credentials/CRED_X"),
        infisical_handler=handler,  # type: ignore[arg-type]
    )

    resolved = await resolver.get_secret_async("cred.x", required=False)

    assert resolved is not None
    assert resolved.get_secret_value() == "tenant-scoped-value"
    assert handler.async_calls[0]["secret_path"] == "/tenant-inference-credentials"


@pytest.mark.parametrize("reader", ["sync", "async"])
async def test_unqualified_mapping_leaves_the_handler_default_in_place(
    reader: str,
) -> None:
    """A bare secret name declares no folder, so the handler's configured
    default stays authoritative -- passing an empty path would silently
    re-root the read."""
    handler = _RecordingInfisicalHandler({("LLM_GLM_API_KEY", None): "default-folder"})
    resolver = SecretResolver(
        config=_config("llm.glm.api_key", "LLM_GLM_API_KEY"),
        infisical_handler=handler,  # type: ignore[arg-type]
    )

    if reader == "sync":
        resolved = resolver.get_secret("llm.glm.api_key", required=False)
        recorded = handler.sync_calls[0]
    else:
        resolved = await resolver.get_secret_async("llm.glm.api_key", required=False)
        recorded = handler.async_calls[0]

    assert resolved is not None
    assert recorded.get("secret_path") is None


def test_field_fragment_is_stripped_before_the_folder_split() -> None:
    handler = _RecordingInfisicalHandler(
        {("DB_CREDENTIALS", "/dev/onex-runtime"): '{"password": "p"}'}
    )
    resolver = SecretResolver(
        config=_config("db.password", "/dev/onex-runtime/DB_CREDENTIALS#password"),
        infisical_handler=handler,  # type: ignore[arg-type]
    )

    resolved = resolver.get_secret("db.password", required=False)

    assert resolved is not None
    assert resolved.get_secret_value() == "p"
    assert handler.sync_calls == [
        {"secret_name": "DB_CREDENTIALS", "secret_path": "/dev/onex-runtime"}
    ]


def test_correlation_id_is_accepted_unchanged() -> None:
    """Guards the call-signature edit itself."""
    handler = _RecordingInfisicalHandler({})
    resolver = SecretResolver(
        config=_config("llm.glm.api_key", "/dev/onex-runtime/LLM_GLM_API_KEY"),
        infisical_handler=handler,  # type: ignore[arg-type]
    )

    assert (
        resolver.get_secret("llm.glm.api_key", required=False, correlation_id=uuid4())
        is None
    )
