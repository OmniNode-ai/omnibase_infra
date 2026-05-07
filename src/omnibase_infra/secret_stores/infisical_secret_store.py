# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``ProtocolSecretStore`` wrapper over ``AdapterInfisical`` (OMN-10557).

This wrapper adapts the SDK-shaped (`secret_name=`, `project_id=`,
`environment_slug=`, `secret_path=`) ``AdapterInfisical`` API to the
async, key-only ``ProtocolSecretStore`` contract. ``AdapterInfisical`` is
*not* mutated — its public methods retain their SDK kwargs because
existing call sites (HandlerInfisical, seed scripts, provisioning) still
depend on them.

Architecture notes:
    - Adapter calls are synchronous; this wrapper bridges to the async
      protocol via ``asyncio.to_thread``. The Infisical SDK is HTTP-bound
      so blocking the event loop is the wrong default.
    - ``get_secret`` translates the adapter's ``SecretResolutionError``
      to ``None`` (per protocol semantics: nullable lookup). Authentication
      and connection issues surface via ``health_check`` returning
      ``False``, not via exceptions during ``get_secret``.
    - ``set_secret`` tries ``update_secret`` first (the common case after
      first write) and falls back to ``create_secret`` if the SDK reports
      the secret does not yet exist.
    - ``delete_secret`` raises ``RuntimeError``: ``AdapterInfisical``
      deliberately exposes no delete operation per the OMN-2286 read-only
      policy.
    - ``close`` shuts down the underlying SDK client. ``timeout_seconds``
      is accepted for protocol compatibility; the SDK shutdown is
      synchronous and effectively immediate.
"""

from __future__ import annotations

import asyncio
import logging

from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical
from omnibase_infra.errors import InfraConnectionError, SecretResolutionError

logger = logging.getLogger(__name__)


class InfisicalSecretStore:
    """``ProtocolSecretStore`` wrapper composed over ``AdapterInfisical``.

    The wrapper does not own the adapter's lifecycle; callers must
    ``initialize()`` the adapter before passing it in (keeping auth
    failures localised to bootstrap).
    """

    def __init__(
        self,
        adapter: AdapterInfisical,
        *,
        project_id: str,
        environment_slug: str,
        secret_path: str,
    ) -> None:
        self._adapter = adapter
        self._project_id = str(project_id)
        self._environment_slug = environment_slug
        self._secret_path = secret_path

    async def get_secret(self, key: str) -> str | None:
        """Retrieve a secret value, or ``None`` if not present."""
        try:
            result = await asyncio.to_thread(
                self._adapter.get_secret,
                secret_name=key,
                project_id=self._project_id,
                environment_slug=self._environment_slug,
                secret_path=self._secret_path,
            )
        except SecretResolutionError:
            return None
        return result.value.get_secret_value()

    async def set_secret(self, key: str, value: str) -> bool:
        """Store or update a secret. Tries update first, then create."""
        try:
            await asyncio.to_thread(
                self._adapter.update_secret,
                key,
                value,
                project_id=self._project_id,
                environment_slug=self._environment_slug,
                secret_path=self._secret_path,
            )
            return True
        except (SecretResolutionError, InfraConnectionError):
            await asyncio.to_thread(
                self._adapter.create_secret,
                key,
                value,
                project_id=self._project_id,
                environment_slug=self._environment_slug,
                secret_path=self._secret_path,
            )
            return True

    async def delete_secret(self, key: str) -> bool:
        """Always raises — AdapterInfisical exposes no delete (OMN-2286)."""
        raise RuntimeError("Infisical adapter is read-only by OMN-2286 policy")

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        """List secret keys at the configured path, optionally filtered."""
        results = await asyncio.to_thread(
            self._adapter.list_secrets,
            project_id=self._project_id,
            environment_slug=self._environment_slug,
            secret_path=self._secret_path,
        )
        keys = [r.key for r in results]
        if prefix is None:
            return keys
        return [k for k in keys if k.startswith(prefix)]

    async def health_check(self) -> bool:
        """Return ``True`` iff the adapter has authenticated successfully."""
        return self._adapter.is_authenticated

    async def close(self, timeout_seconds: float = 30.0) -> None:
        """Release the underlying SDK client.

        ``timeout_seconds`` is accepted for ``ProtocolSecretStore``
        compatibility but is unused: ``AdapterInfisical.shutdown()`` is
        synchronous and clears in-memory references only.
        """
        del timeout_seconds
        await asyncio.to_thread(self._adapter.shutdown)


__all__: list[str] = ["InfisicalSecretStore"]
