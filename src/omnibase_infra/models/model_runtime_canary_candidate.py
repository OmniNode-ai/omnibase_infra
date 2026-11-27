# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The four refs one runtime canary run judges (OMN-26463)."""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__: list[str] = ["ModelRuntimeCanaryCandidate"]

_SHA: Final = re.compile(r"^[0-9a-f]{7,40}$")
KEY_WIDTH: Final = 12
NO_OVERLAY: Final = "no-overlay"


class ModelRuntimeCanaryCandidate(BaseModel):
    """A dev candidate: omnibase_infra, omnimarket and omnibase_core refs plus an overlay revision.

    Every ref is a commit sha, never a branch name: a receipt for a moving name
    would prove nothing about the commit the dev roll later builds. The overlay
    revision is empty when the roll mounts no overlay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    infra_ref: str = Field(min_length=7)
    omnimarket_ref: str = Field(min_length=7)
    core_ref: str = Field(min_length=7)
    overlay_rev: str = ""

    @field_validator("infra_ref", "omnimarket_ref", "core_ref", "overlay_rev")
    @classmethod
    def _is_sha(cls, value: str) -> str:
        if value and not _SHA.match(value):
            raise ValueError(f"{value!r} is not a commit sha (7 to 40 lowercase hex)")
        return value

    @property
    def key(self) -> str:
        """The receipt's file stem: the first 12 hex of each ref, joined."""
        return "-".join(
            (ref[:KEY_WIDTH] if ref else NO_OVERLAY)
            for ref in (
                self.infra_ref,
                self.omnimarket_ref,
                self.core_ref,
                self.overlay_rev,
            )
        )
