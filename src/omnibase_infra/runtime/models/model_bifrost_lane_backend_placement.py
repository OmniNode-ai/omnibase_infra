# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tier-ladder placement of a backend a lane overlay adds (OMN-19215).

The routing ladder ships in the omnimarket package and names only the backends
the base contract declares, so a backend a lane adds is reachable only by a
per-run pin. A placement names the tier and the rungs the added backend is a
fallback for; the renderer passes it through to the rendered contract and the
routing authority mirrors the backend into that tier after those rungs. The
tier and rung names are checked there, against the ladder it loads.

``use_for`` and ``weight`` (OMN-19432) say which task classes the added backend
serves and how large a share of a spread group it takes; both render only when
they differ from the default.

``mode`` (AC4) says whether the added backend only catches its rungs' failures
(``fallback``, the default) or also shares their first-choice traffic
(``spread``). The renderer writes ``mode`` only when it is not the default, so a
fallback placement renders the same bytes a routing authority without the field
already accepts.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.enum_bifrost_lane_placement_mode import (
    EnumBifrostLanePlacementMode,
)


class ModelBifrostLaneBackendPlacement(BaseModel):
    """Where a lane-added backend sits in the routing tier ladder."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    tier: str = Field(min_length=1)
    fallback_for: tuple[str, ...] = Field(min_length=1)
    max_context_tokens: int = Field(gt=0)
    mode: EnumBifrostLanePlacementMode = EnumBifrostLanePlacementMode.FALLBACK
    #: OMN-19432. Task classes the backend is offered for, narrowing each rung it
    #: mirrors; None keeps the rung's whole list. The routing authority refuses a
    #: list that shares no class with a rung.
    use_for: tuple[str, ...] | None = Field(default=None, min_length=1)
    #: OMN-19432. Share of the rung's first-choice traffic in spread mode, against
    #: the rung's own 1.0, from measured capacity.
    weight: float = Field(default=1.0, gt=0)


__all__ = ["ModelBifrostLaneBackendPlacement"]
