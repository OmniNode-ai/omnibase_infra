# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Names-only slice of ``ProtocolSecretStore`` for the required-secret gate.

The auto-wiring gate (OMN-21423 AC3) decides whether a runtime joins a
contract's consumer group from the presence of the contract's required secret.
It is typed against this protocol, which carries ``list_keys`` and nothing
else, so the gate cannot read a secret value. ``AdapterEnvSecretStore`` and
``InfisicalSecretStore`` both satisfy it.
"""

from __future__ import annotations

from typing import Protocol


class ProtocolSecretNameLister(Protocol):
    """List the secret names a store holds, never their values."""

    async def list_keys(self, prefix: str | None = None) -> list[str]: ...
