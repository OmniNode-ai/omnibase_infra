# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deterministic in-memory ``ProtocolSecretStore`` for tests (OMN-10559).

Used by the canonical conftest in omnimarket and other consumers to
exercise code that depends on a ``ProtocolSecretStore`` without standing
up a real backend.

Packaging note: this module ships in the omnibase_infra wheel (under
``omnibase_infra.test_utils``) because downstream test suites import it.
The package is internal-platform-only; shipping test helpers is
deliberate. Heavier fixtures requiring infrastructure (Postgres, Kafka)
belong in ``omnibase_infra.testing`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping


class FakeSecretStore:
    """In-memory ``ProtocolSecretStore`` impl backed by a ``dict``.

    Fully implements every protocol method (gets, sets, deletes, lists,
    health, close). Constructor accepts an optional ``initial`` mapping
    to preload values.
    """

    def __init__(self, initial: Mapping[str, str] | None = None) -> None:
        self._data: dict[str, str] = dict(initial) if initial else {}
        self._closed: bool = False

    async def get_secret(self, key: str) -> str | None:
        return self._data.get(key)

    async def set_secret(self, key: str, value: str) -> bool:
        self._data[key] = value
        return True

    async def delete_secret(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        if prefix is None:
            return sorted(self._data.keys())
        return sorted(k for k in self._data if k.startswith(prefix))

    async def health_check(self) -> bool:
        return not self._closed

    async def close(self, timeout_seconds: float = 30.0) -> None:
        del timeout_seconds
        self._closed = True


__all__: list[str] = ["FakeSecretStore"]
