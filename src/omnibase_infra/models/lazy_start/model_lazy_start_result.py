# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The front door's answer to one held request (OMN-21592)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome

__all__: list[str] = ["ModelLazyStartResult"]


class ModelLazyStartResult(BaseModel):
    """``answer`` is what the node's dispatcher returned for a delivered request.

    ``START_TIMEOUT`` keeps the request held: it is delivered once the start
    completes (``held=True``), never dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    contract_name: str
    correlation_id: UUID
    outcome: EnumLazyStartOutcome
    held: bool = False
    answer: object = None
    detail: str = ""
