# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol interface for delegation dispatch ports.

Defines the structural interface that RuntimeDelegationDispatchPort conforms to,
enabling handler_wiring to inject the dispatch port without a concrete dependency.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class ProtocolDelegationDispatchPort(Protocol):
    async def dispatch(
        self,
        *,
        prompt: str,
        task_type: str,
        correlation_id: UUID,
        max_tokens: int,
        source_file_path: str | None,
        source_session_id: str | None,
        wait: bool,
        quality_contract_mode: str,
        acceptance_criteria: tuple[str, ...],
    ) -> dict[str, object]: ...


__all__ = ["ProtocolDelegationDispatchPort"]
