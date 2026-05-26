# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Graph mutation model for session graph projector.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 9).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelGraphMutation"]


class ModelGraphMutation(BaseModel):
    """A single Cypher MERGE mutation with parameters.

    Attributes:
        cypher: Parameterized Cypher query string.
        params: Parameter dict for the query.
    """

    model_config = ConfigDict(frozen=True)

    cypher: str
    params: dict[str, str] = Field(default_factory=dict)
