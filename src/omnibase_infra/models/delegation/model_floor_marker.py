# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The workspace floor marker an onboarding run stamps (OMN-22183)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

__all__ = ["FLOOR_MARKER_FILE", "ModelFloorMarker"]

FLOOR_MARKER_FILE = ".onex-workspace-floor.json"


class ModelFloorMarker(BaseModel):
    """The proven omnimarket commit ``.onex-workspace-floor.json`` names."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    omnimarket_commit: str = ""

    @classmethod
    def names_commit(cls, root: Path | None, commit: str) -> bool:
        """True when the marker under ``root`` names ``commit``.

        A missing, unreadable or malformed marker, or one with no commit,
        names nothing: the caller treats it as "no floor", never as a match.
        """
        if not root:
            return False
        try:
            raw = json.loads((Path(root) / FLOOR_MARKER_FILE).read_text("utf-8"))
            marker = cls.model_validate(raw)
        except (OSError, ValueError):
            return False
        return bool(marker.omnimarket_commit) and marker.omnimarket_commit == commit
