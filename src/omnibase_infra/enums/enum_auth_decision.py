# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Authorization decision outcomes for the auth gate compute node.

Used by HandlerAuthGate to express the result of the 10-step authorization
cascade. Each tool invocation is either allowed, denied, or soft-denied
(allowed with a warning banner).

Ticket: OMN-2125
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumAuthDecision(str, Enum):
    """Authorization decision outcome from the auth gate cascade.

    Attributes:
        ALLOW: Tool invocation is permitted under current authorization scope.
        DENY: Tool invocation is rejected; the request must not proceed.
        SOFT_DENY: Tool invocation is permitted but flagged (e.g., emergency
            override active). A visible banner should be displayed.
    """

    ALLOW = "allow"
    """Tool invocation is fully authorized."""

    DENY = "deny"
    """Tool invocation is rejected."""

    SOFT_DENY = "soft_deny"
    """Tool invocation is permitted with a visible warning banner."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    def is_permitted(self) -> bool:
        """Return True if the decision allows the tool invocation."""
        return self in {EnumAuthDecision.ALLOW, EnumAuthDecision.SOFT_DENY}


__all__: list[str] = ["EnumAuthDecision"]
