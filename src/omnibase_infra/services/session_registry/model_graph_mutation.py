# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Graph mutation model for session graph projector.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ModelGraphMutation"]


@dataclass(frozen=True)
class ModelGraphMutation:
    """A single Cypher MERGE mutation with parameters.

    Attributes:
        cypher: Parameterized Cypher query string.
        params: Parameter dict for the query.
    """

    cypher: str
    params: dict[str, str] = field(default_factory=dict)
