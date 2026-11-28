# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the stability integration refresh did (OMN-26715)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelStabilityIntegrationRefreshResult(BaseModel):
    """Outcome, the heads built, the image and the rollback target.

    ``rollback_target`` holds one ``(service, image_id)`` pair per service, read
    before the first change. ``seconds_to_healthy`` is -1 when /health never
    read 200 inside the window.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Literal["refreshed", "rolled_back", "build_failed", "refused"]
    heads: tuple[tuple[str, str], ...] = ()
    deploy_tags: tuple[tuple[str, str], ...] = ()
    built_image_digest: str = ""
    rollback_target: tuple[tuple[str, str], ...] = ()
    seconds_to_healthy: float = -1.0
    detail: str = ""
