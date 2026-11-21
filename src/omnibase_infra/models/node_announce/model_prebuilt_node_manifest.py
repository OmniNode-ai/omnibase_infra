# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one file the runtime image carries: node catalog, auto-wiring manifest and their stamp (OMN-26361)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)
from omnibase_infra.models.node_announce.model_prebuilt_manifest_stamp import (
    ModelPrebuiltManifestStamp,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest


class ModelPrebuiltNodeManifest(BaseModel):
    """Boot step 0's result and discovery's manifest, written once at image build."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    stamp: ModelPrebuiltManifestStamp
    announced: ModelPackageAnnounceResult
    manifest: ModelAutoWiringManifest
