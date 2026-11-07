# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runner hook whose executed copy differs from the host copy."""

from pydantic import BaseModel, ConfigDict


class ModelRunnerHookMountFinding(BaseModel):
    """A container, the script it executes, and why that copy is not the host copy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    container: str
    script: str
    executed_path: str
    reason: str
    container_sha256: str | None = None
    host_sha256: str
