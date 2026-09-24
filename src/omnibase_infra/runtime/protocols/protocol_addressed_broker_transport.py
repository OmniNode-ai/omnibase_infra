# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Public identity exposed by a configured broker transport (OMN-18933).

Carried from the Codex draft omnibase_infra#3951 (OMN-18925 capture root).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolAddressedBrokerTransport(Protocol):
    """A broker transport whose configured endpoint and environment are known.

    ``EventBusKafka`` satisfies it. An in-process bus has an environment but no
    broker address, so it does not, and the bounded delegation route gate has
    nothing to bound for it.
    """

    @property
    def bootstrap_servers(self) -> str: ...

    @property
    def environment(self) -> str: ...


__all__ = ["ProtocolAddressedBrokerTransport"]
