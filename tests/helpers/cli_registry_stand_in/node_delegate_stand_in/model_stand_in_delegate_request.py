# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The stand-in delegate contract's input model.

Its closed fields carry values chosen for this fixture. ``claude-code`` is
among the sources because it is the identity ``onex delegate`` stamps by
default; the other values exist nowhere in production, so a CLI that offered
a copied production list instead of reading this model would fail the tests.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The criterion slugs this stand-in admits.
STAND_IN_CRITERIA = frozenset({"probe_criterion_alpha", "probe_criterion_beta"})


class ModelStandInDelegateRequest(BaseModel):
    """Stand-in for the delegate node's declared input model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    source: Literal["claude-code", "stand-in-source"]
    correlation_id: UUID = Field(default_factory=uuid4)
    max_tokens: int | None = Field(default=None, gt=0)
    quality_contract_mode: Literal["probe_extend", "probe_replace"] = "probe_extend"
    acceptance_criteria: tuple[str, ...] = ()
    response_contract: dict[str, object] | None = None
    system_prompt: str | None = None
    requested_timeout_seconds: int | None = None
    backend_id: str | None = None

    @field_validator("acceptance_criteria")
    @classmethod
    def _declared_criteria_only(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unsupported = sorted(set(value) - STAND_IN_CRITERIA)
        if unsupported:
            raise ValueError(
                f"unsupported acceptance criteria: {', '.join(unsupported)}; "
                f"allowed: {', '.join(sorted(STAND_IN_CRITERIA))}"
            )
        return value
