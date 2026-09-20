# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol interface for delegation dispatch ports.

Defines the structural interface that RuntimeDelegationDispatchPort conforms to,
enabling handler_wiring to inject the dispatch port without a concrete dependency.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from omnibase_core.models.delegation.wire import ModelDelegationProvenance


class ProtocolDelegationDispatchPort(Protocol):
    async def dispatch(
        self,
        *,
        prompt: str,
        task_type: str,
        correlation_id: UUID,
        max_tokens: int | None,
        source_file_path: str | None,
        source_session_id: str | None,
        wait: bool,
        execution_timeout_seconds: int,
        terminal_delivery_margin_seconds: int,
        quality_contract_mode: str,
        acceptance_criteria: tuple[str, ...],
        tenant_id: str | None = None,
        # OMN-18321: declared here because the OmniMarket consumer protocol
        # declares it and its handler passes it on EVERY delegation (OMN-18172,
        # omnimarket#2494). It was added on that side alone, and the resulting
        # TypeError on the deployed bus path was swallowed by the consumer's own
        # `except Exception` into a delegate-skill-failed terminal -- so the
        # dev-lane chain died silently for a day and wrote no FSM row at all.
        # Parity is held mechanically by
        # tests/integration/runtime/test_delegation_dispatch_port_consumer_kwarg_parity.py,
        # which reads the consumer's declaration rather than a list kept here.
        provenance: ModelDelegationProvenance | None = None,
        backend_id: str | None = None,
        response_contract: dict[str, object] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, object] | None = None,
    ) -> dict[str, object]: ...


__all__ = ["ProtocolDelegationDispatchPort"]
