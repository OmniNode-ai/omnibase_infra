# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope-enforcing wrapper adapter for ProtocolTicketService.

Wraps AdapterTicketLinear with an allowed-action set. Any call to an action
not in the set raises InvariantViolation before the underlying adapter is
reached.

Related Tickets:
    - OMN-8065: Task 7 — Scoped wrapper adapters for TicketService, EventBus, LLMProvider
"""

from __future__ import annotations

from omnibase_infra.adapters.ticket.adapter_ticket_service_linear import (
    AdapterTicketLinear,
)
from omnibase_infra.errors.error_invariant_violation import InvariantViolation

_PROTOCOL_DOMAIN = "ticket_service"

_ACTION_CREATE_TICKET = "create_ticket"
_ACTION_GET_TICKET = "get_ticket"
_ACTION_UPDATE_TICKET_STATUS = "update_ticket_status"
_ACTION_ADD_COMMENT = "add_comment"
_ACTION_LIST_TICKETS = "list_tickets"
_ACTION_GET_TICKET_STATUS = "get_ticket_status"
_ACTION_HEALTH_CHECK = "health_check"
_ACTION_CLOSE = "close"

# Mirrors AdapterTicketLinear's unimplemented create_ticket metadata passthrough.
# ONEX_EXCLUDE: dict_str_any - no ticket metadata domain type exists yet.
_TicketMetadata = dict[str, object]


class AdapterTicketLinearScoped:
    """Scope-enforcing wrapper around AdapterTicketLinear.

    Delegates all method calls to the wrapped adapter after checking whether
    the action is present in ``allowed_actions``.  Raises InvariantViolation
    on any disallowed action before any network I/O occurs.

    Args:
        inner: The AdapterTicketLinear to delegate to.
        allowed_actions: Frozenset of action name strings that callers may
            invoke.  Pass an empty frozenset to deny all actions.
    """

    def __init__(
        self,
        inner: AdapterTicketLinear,
        allowed_actions: frozenset[str],
    ) -> None:
        self._inner = inner
        self._allowed_actions = allowed_actions

    def _check(self, action_name: str) -> None:
        if action_name not in self._allowed_actions:
            raise InvariantViolation(
                action_name=action_name,
                protocol_domain=_PROTOCOL_DOMAIN,
                allowed_actions=tuple(sorted(self._allowed_actions)),
            )

    async def create_ticket(
        self,
        title: str,
        description: str,
        labels: list[str] | None = None,
        assignee: str | None = None,
        metadata: _TicketMetadata | None = None,
    ) -> str:
        self._check(_ACTION_CREATE_TICKET)
        return await self._inner.create_ticket(
            title=title,
            description=description,
            labels=labels,
            assignee=assignee,
            metadata=metadata,
        )

    async def get_ticket(self, ticket_id: str) -> dict[str, object]:
        self._check(_ACTION_GET_TICKET)
        return await self._inner.get_ticket(ticket_id)

    async def update_ticket_status(self, ticket_id: str, status: str) -> bool:
        self._check(_ACTION_UPDATE_TICKET_STATUS)
        return await self._inner.update_ticket_status(ticket_id, status)

    async def add_comment(self, ticket_id: str, body: str) -> str:
        self._check(_ACTION_ADD_COMMENT)
        return await self._inner.add_comment(ticket_id, body)

    async def list_tickets(
        self,
        filters: dict[str, object] | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        self._check(_ACTION_LIST_TICKETS)
        return await self._inner.list_tickets(filters=filters, limit=limit)

    async def get_ticket_status(self, ticket_id: str) -> str:
        self._check(_ACTION_GET_TICKET_STATUS)
        return await self._inner.get_ticket_status(ticket_id)

    async def health_check(self) -> bool:
        self._check(_ACTION_HEALTH_CHECK)
        return await self._inner.health_check()

    async def close(self, timeout_seconds: float = 30.0) -> None:
        self._check(_ACTION_CLOSE)
        await self._inner.close(timeout_seconds=timeout_seconds)


__all__: list[str] = ["AdapterTicketLinearScoped"]
