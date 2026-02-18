# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Minimal protocol for secret resolution used by ConfigPrefetcher.

This protocol avoids circular imports between the config_discovery
subpackage and the handler layer by defining only the interface
that ConfigPrefetcher needs.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import SecretStr


@runtime_checkable
class ProtocolSecretResolver(Protocol):
    """Minimal protocol for a secret resolver that supports synchronous reads.

    Any object that implements ``get_secret_sync`` with the expected
    signature satisfies this protocol, including ``HandlerInfisical``.
    """

    def get_secret_sync(
        self,
        *,
        secret_name: str,
        secret_path: str,
    ) -> SecretStr | None:
        """Fetch a single secret synchronously.

        Args:
            secret_name: The key name of the secret.
            secret_path: The folder/path where the secret resides.

        Returns:
            The secret value, or ``None`` if not found.
        """
        ...
