# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TypedDict for `RoutingGate.audit()` results (OMN-7404).

Kept as a dedicated TypedDict (rather than ``dict[str, float | bool | None]``)
so each field carries its own precise type instead of a 3-way union value
type -- the latter counts against the repo's non-optional-union ratchet
(``onex-validate-unions``; only ``X | None`` optionals are exempted).
"""

from __future__ import annotations

from typing import TypedDict


class TypedDictRoutingAudit(TypedDict):
    """Shadow-mode routing audit result. Never used to select or alter routing.

    Attributes:
        confidence: Classifier's P(decision is good) in [0.0, 1.0], or
            ``None`` when no classifier is configured or it raised.
        would_flag: True when confidence is below the gate's threshold.
        audit_only: Always True -- documents that this result is advisory
            (logged/stored) and never feeds back into the routing decision.
    """

    confidence: float | None
    would_flag: bool
    audit_only: bool


__all__ = ["TypedDictRoutingAudit"]
