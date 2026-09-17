# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegation terminal itself -- one run's answer and its route evidence (OMN-18569).

Read-side mirror of ``omnimarket``'s ``ModelDelegateSkillResponse``. See
:mod:`omnibase_infra.cli.delegate_terminal_resolver` for why this CLI mirrors
the wire contract instead of importing it, and for how the two carrier shapes
this object arrives in are resolved.

.. versionadded:: OMN-18569
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli.model_delegate_attempt import ModelDelegateAttempt
from omnibase_infra.cli.model_delegate_terminal_metrics import (
    ModelDelegateTerminalMetrics,
)

__all__ = ["ModelDelegateTerminal"]


class ModelDelegateTerminal(BaseModel):
    """One delegation's terminal state: what it answered and which rungs it tried.

    ``attempts`` is the only REQUIRED field, and deliberately so: it is what
    identifies this object as a delegation terminal rather than some other
    node's terminal payload, and it is what the run files' route attribution is
    resolved from. Everything else is optional for the same reason the attempt
    record's fields are -- this reads a recording, and a recording missing a
    field must still be reportable.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    attempts: tuple[ModelDelegateAttempt, ...] = Field(...)
    response: str = Field(default="")
    model_name: str | None = Field(default=None)
    provider: str | None = Field(default=None)
    status: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    terminal_failure_cause: str | None = Field(default=None)
    quality_gate_passed: bool | None = Field(default=None)
    quality_score: float | None = Field(default=None)
    quality_gates_failed: tuple[str, ...] = Field(default=())
    metrics: ModelDelegateTerminalMetrics | None = Field(default=None)

    @property
    def accepted_attempt(self) -> ModelDelegateAttempt | None:
        """The rung whose output this run accepted, or ``None`` when unproven."""
        for attempt in self.attempts:
            if attempt.is_accepted:
                return attempt
        return None
