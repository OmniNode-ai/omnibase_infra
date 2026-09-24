# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Public identity exposed by a configured broker transport."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolAddressedBrokerTransport(Protocol):
    """A broker transport whose configured endpoint and environment are known."""

    @property
    def bootstrap_servers(self) -> str: ...

    @property
    def environment(self) -> str: ...


__all__ = ["ProtocolAddressedBrokerTransport"]
