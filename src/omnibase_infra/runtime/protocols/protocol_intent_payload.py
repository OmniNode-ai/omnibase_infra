# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for typed intent payloads.

All intent payloads must have an ``intent_type`` field (typically a
Literal string) used by IntentExecutor for routing. This protocol
replaces hasattr-based duck typing with structural subtyping per PEP 544.

Related:
    - IntentExecutor: Uses this protocol for intent_type extraction
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
"""

from __future__ import annotations

from typing import Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class ProtocolIntentPayload(Protocol):
    """Protocol for typed intent payloads.

    All intent payloads must declare an ``intent_type`` field used by
    IntentExecutor for routing to the appropriate effect handler.
    """

    intent_type: str


__all__: list[str] = ["ProtocolIntentPayload"]
