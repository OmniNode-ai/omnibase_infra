# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One failed boot-time Bifrost endpoint probe (OMN-19455)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.enum_bifrost_endpoint_probe_failure_kind import (
    EnumBifrostEndpointProbeFailureKind,
)


class ModelBifrostEndpointProbeFailure(BaseModel):
    """What the probe found, and whether the backend was reachable at all."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EnumBifrostEndpointProbeFailureKind
    detail: str = Field(min_length=1)
