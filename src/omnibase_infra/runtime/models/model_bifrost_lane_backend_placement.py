# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tier-ladder placement of a backend a lane overlay adds (OMN-19215).

The routing ladder ships in the omnimarket package and names only the backends
the base contract declares, so a backend a lane adds is reachable only by a
per-run pin. A placement names the tier and the rungs the added backend is a
fallback for; the renderer passes it through to the rendered contract and the
routing authority mirrors the backend into that tier after those rungs. The
tier and rung names are checked there, against the ladder it loads.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBifrostLaneBackendPlacement(BaseModel):
    """Where a lane-added backend sits in the routing tier ladder."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    tier: str = Field(min_length=1)
    fallback_for: tuple[str, ...] = Field(min_length=1)
    max_context_tokens: int = Field(gt=0)


__all__ = ["ModelBifrostLaneBackendPlacement"]
