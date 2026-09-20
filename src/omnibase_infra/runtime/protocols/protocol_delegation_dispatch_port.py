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

#: The execution budget a caller that passes neither argument resolves to
#: (OMN-18924). Mirrors `omnibase_infra.cli.task_class_selection`'s
#: DEFAULT_EXECUTION_BUDGET; the two are pinned equal by
#: tests/unit/runtime/test_dispatch_port_budget_defaults_omn18924.py rather
#: than shared by an import, because the runtime layer does not depend on the
#: CLI layer and adding that edge to share two integers would be the wrong
#: trade.
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 240
DEFAULT_TERMINAL_DELIVERY_MARGIN_SECONDS = 60


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
        # OMN-18924: defaulted, not required. See the implementation's note --
        # these landed as required while the deployed caller passed neither,
        # and every dev-lane delegation terminalized `provider_error` on the
        # resulting TypeError. Exactly the shape the OMN-18321 comment below
        # records, one incident later.
        execution_timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        terminal_delivery_margin_seconds: int = DEFAULT_TERMINAL_DELIVERY_MARGIN_SECONDS,
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
