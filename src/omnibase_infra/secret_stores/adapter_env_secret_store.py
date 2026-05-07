# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``ProtocolSecretStore`` adapter backed by ``os.environ`` (OMN-10558).

Used by local development and any runtime profile where Infisical is
unavailable. ``get_secret`` is a *nullable lookup* — a missing key
returns ``None``. Required-service validation is the caller's
responsibility (e.g., ``omnimarket.config.Settings`` raises if a
mandatory secret is unset). The protocol semantics (OMN-10556) are the
contract.

Mutating env vars at runtime is unsafe in multi-threaded/async
processes, so ``set_secret`` and ``delete_secret`` raise ``RuntimeError``.
"""

from __future__ import annotations

import os


class AdapterEnvSecretStore:
    """``ProtocolSecretStore`` adapter that reads from ``os.environ``.

    Read-only by design. ``health_check`` always returns ``True``;
    ``close`` is a no-op.
    """

    async def get_secret(self, key: str) -> str | None:
        """Return ``os.environ[key]`` or ``None`` if not set."""
        return os.environ.get(key)

    async def set_secret(self, key: str, value: str) -> bool:
        raise RuntimeError("AdapterEnvSecretStore is read-only")

    async def delete_secret(self, key: str) -> bool:
        raise RuntimeError("AdapterEnvSecretStore is read-only")

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        """Return env var names; filter by ``prefix`` when supplied."""
        keys = list(os.environ.keys())
        if prefix is None:
            return keys
        return [k for k in keys if k.startswith(prefix)]

    async def health_check(self) -> bool:
        return True

    async def close(self, timeout_seconds: float = 30.0) -> None:
        return None


__all__: list[str] = ["AdapterEnvSecretStore"]
