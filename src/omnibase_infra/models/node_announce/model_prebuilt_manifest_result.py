# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of loading the prebuilt node manifest (OMN-26361)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.node_announce.model_prebuilt_node_manifest import (
    ModelPrebuiltNodeManifest,
)


class ModelPrebuiltManifestResult(BaseModel):
    """``prebuilt`` is set when the file was loaded and its stamp holds.

    A load that cannot be served leaves ``prebuilt`` unset and names why: ``reason``
    is ``absent``, ``unreadable`` or ``stale``, and ``differing_fields`` lists every
    stamp field that no longer matches the installed packages.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prebuilt: ModelPrebuiltNodeManifest | None = None
    reason: str = ""
    differing_fields: tuple[str, ...] = Field(default=())
