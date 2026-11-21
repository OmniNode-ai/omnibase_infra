# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The single-contract start transaction the lazy-start front door drives (OMN-21592).

This is the smallest seam plan task A4 needs from H8 (the whole single-contract
transaction: eligibility, registration, provision with declared specs, confirm,
attach), which is not built yet. The front door calls, for one contract,
``build`` (import the handler and prepare its dispatchers) and then ``attach``
(provision its topics, confirm them, then attach its subscriptions, in that
order), and then ``deliver`` (the held request to the node's own dispatcher).
On a failure in either step it calls ``rollback``, which removes every
dispatcher and subscription the start created and never deletes a topic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__: list[str] = ["ProtocolLazyStartTransaction"]


@runtime_checkable
class ProtocolLazyStartTransaction(Protocol):
    """One runtime's start steps for a dormant contract."""

    async def build(self, contract_name: str) -> None:
        """Import the contract's handler and prepare its dispatchers."""
        ...

    async def attach(self, contract_name: str) -> None:
        """Provision, confirm, then attach every subscription, or raise."""
        ...

    async def rollback(self, contract_name: str) -> None:
        """Remove every dispatcher and subscription this start created; keep topics."""
        ...

    async def deliver(self, contract_name: str, topic: str, envelope: object) -> object:
        """Dispatch ``envelope`` to the started node's own dispatcher."""
        ...
