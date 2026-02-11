# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Auth gate decision model — output of the authorization decision cascade.

Carries the allow/deny/soft_deny verdict, the cascade step that produced it,
a human-readable reason, and an optional banner for soft-deny decisions.

Ticket: OMN-2125
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_auth_decision import EnumAuthDecision


class ModelAuthGateDecision(BaseModel):
    """Output model for the auth gate compute handler.

    Attributes:
        decision: The authorization verdict (allow, deny, soft_deny).
        step: The 1-indexed cascade step that produced the decision (1-10).
        reason: Human-readable explanation of the decision.
        reason_code: Machine-readable reason code (e.g., "whitelisted_path").
        banner: Warning banner text for soft_deny decisions. Empty for others.

    Warning:
        **Non-standard __bool__ behavior**: Returns ``True`` only when the
        decision permits the tool invocation (ALLOW or SOFT_DENY). Differs
        from typical Pydantic behavior where any non-empty model is truthy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    decision: EnumAuthDecision = Field(..., description="Authorization verdict.")
    step: int = Field(
        ..., ge=1, le=10, description="Cascade step that produced this decision."
    )
    reason: str = Field(..., min_length=1, description="Human-readable explanation.")
    reason_code: str = Field(
        ..., min_length=1, description="Machine-readable reason code."
    )
    banner: str = Field(
        default="",
        description="Warning banner for soft_deny decisions.",
    )

    def __bool__(self) -> bool:
        """Allow using decision in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            the decision permits the tool invocation (ALLOW or SOFT_DENY).
            Differs from typical Pydantic behavior.
        """
        return self.decision.is_permitted()


__all__: list[str] = ["ModelAuthGateDecision"]
