# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Authenticate to the broker as a developer principal (OMN-20822)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import SecretStr


@runtime_checkable
class ProtocolBrokerPrincipalSession(Protocol):
    """Log in as the principal and list the topics it can see."""

    def list_topics_as(self, principal: str, password: SecretStr) -> tuple[str, ...]:
        """Topics visible to the principal. Raises when the login is refused."""
        ...
