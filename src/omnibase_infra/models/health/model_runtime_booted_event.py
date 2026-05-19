# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime booted event model — OMN-9139.

Published on process start to declare what code is actually running.
Subscribed by ``node_runtime_source_attestor_effect`` which compares
``runtime_source_hash`` against the current ``main`` HEAD for each repo
and emits friction events when drift exceeds the configured threshold.

Topic: ``onex.evt.runtime.booted.v1``
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeBootedEvent(BaseModel):
    """Boot-time attestation published when the runtime process starts.

    All fields are required — the runtime must know its own source hash
    before it can serve requests. A missing or ``unknown`` hash is a
    deployment defect, not a graceful-degradation case.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    container_ref: str = Field(
        ...,
        description="Docker container name (from HOSTNAME env var or compose service name).",
    )
    runtime_source_hash: str = Field(
        ...,
        description=(
            "Git commit SHA baked into the image at build time via "
            "RUNTIME_SOURCE_HASH build-arg. Must not be 'unknown' or empty — "
            "those values indicate a build without attestation."
        ),
    )
    booted_at: datetime = Field(
        ...,
        description="UTC timestamp when the runtime process started.",
    )
    python_package_hashes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of installed ONEX package name to its __source_hash__ "
            "(from package/__about__.py). Populated for omnibase_core, "
            "omnibase_infra, omnibase_spi, and any plugin packages that "
            "expose __source_hash__. Empty dict is valid for packages that "
            "have not yet been updated to emit __about__.py."
        ),
    )
    compose_project: str = Field(
        default="unknown",
        description="Docker Compose project name (from COMPOSE_PROJECT env var).",
    )


__all__: list[str] = ["ModelRuntimeBootedEvent"]
