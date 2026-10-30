# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Redpanda Admin API calls the broker principal effect needs (OMN-20822)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import SecretStr


@runtime_checkable
class ProtocolBrokerPrincipalAdmin(Protocol):
    """List the broker's SCRAM users and set one user's password."""

    def list_users(self) -> tuple[str, ...]:
        """Every SCRAM user the broker holds. Raises when the Admin API is unreachable."""
        ...

    def set_password(self, name: str, password: SecretStr, *, exists: bool) -> None:
        """Create the user, or replace its password when ``exists``."""
        ...
