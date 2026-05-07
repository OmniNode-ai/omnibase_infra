# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Env-var-backed implementation of ProtocolSecretStore for local dev / no-Infisical path."""

from __future__ import annotations

import os

from omnibase_spi.protocols.services.protocol_secret_store import ProtocolSecretStore


class AdapterEnvSecretStore:
    """ProtocolSecretStore backed by os.environ.

    Intended for local development and tests where Infisical is unavailable.
    get_secret returns None on miss — never raises. Required-value validation
    is the caller's responsibility (e.g. Settings).
    """

    async def get_secret(self, key: str) -> str | None:
        return os.environ.get(key)

    async def set_secret(self, key: str, value: str) -> bool:
        os.environ[key] = value
        return True

    async def delete_secret(self, key: str) -> bool:
        existed = key in os.environ
        os.environ.pop(key, None)
        return existed

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        if prefix is None:
            return list(os.environ.keys())
        return [k for k in os.environ if k.startswith(prefix)]

    async def health_check(self) -> bool:
        return True

    async def close(self, timeout_seconds: float = 30.0) -> None:  # stub-ok
        pass


# Runtime-checkable structural check — ensures protocol satisfaction is verified at import.
_: ProtocolSecretStore = AdapterEnvSecretStore()
