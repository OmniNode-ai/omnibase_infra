# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What boot step 0 hands the kernel: the node catalog and, from a prebuilt file, the manifest (OMN-26361)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest


class ModelBootNodeCatalog(BaseModel):
    """``manifest`` is set only when the catalog came from the prebuilt file.

    A caller that finds ``manifest`` unset discovers contracts from ``announced``
    exactly as before.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    announced: ModelPackageAnnounceResult
    manifest: ModelAutoWiringManifest | None = None
