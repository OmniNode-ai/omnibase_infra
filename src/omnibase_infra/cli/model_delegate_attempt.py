# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One routing rung a delegation attempted, with that rung's own verdict (OMN-18569).

Read-side mirror of ``omnimarket``'s ``ModelDelegateSkillAttemptRecord``. See
:mod:`omnibase_infra.cli.delegate_terminal_resolver` for why this CLI mirrors
the wire contract instead of importing it.

.. versionadded:: OMN-18569
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDelegateAttempt"]


class ModelDelegateAttempt(BaseModel):
    """One rung this delegation attempted, and what that rung decided.

    Every field is optional even where the wire model declares it required.
    This model reads a RECORDING of what a runtime produced: a rung record
    missing a field has to be reportable as such, because the customer artifact
    it feeds is the only place a customer can see which backend refused them.
    An absent field must read as ``null`` there, never vanish, and never make
    the whole recording unparseable.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    tier: str | None = Field(default=None)
    backend_id: str | None = Field(default=None)
    model_id: str | None = Field(default=None)
    quality_gate_passed: bool | None = Field(default=None)
    quality_score: float | None = Field(default=None)
    cost_usd: float | None = Field(default=None)
    failure_class: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    acceptance_decision: str | None = Field(default=None)
    acceptance_reason: str | None = Field(default=None)
    input_tokens_measured: int | None = Field(default=None)
    input_token_budget: int | None = Field(default=None)
    # OMN-19765: the pinned or house backend the local BYOK route (omnimarket
    # ``substitute_local_byok_route``) replaced to produce THIS attempt's
    # ``backend_id``, or ``None`` when no substitution occurred. Read alongside
    # ``backend_id`` by the pin check so an in-process BYOK-substituted answer
    # is not mistaken for an escalation off the pin.
    substituted_from_backend_id: str | None = Field(default=None)

    @property
    def is_accepted(self) -> bool:
        """Whether THIS rung is the one whose output the run accepted.

        An explicit ``acceptance_decision`` is authoritative when present. Only
        when the runtime recorded no decision at all does the quality verdict
        stand in for one -- the pre-OMN-18569 behaviour, preserved deliberately
        so an older recording still attributes its route rather than silently
        attributing none.
        """
        decision = (self.acceptance_decision or "").strip().lower()
        if decision:
            return decision == "accept"
        return bool(self.quality_gate_passed)
