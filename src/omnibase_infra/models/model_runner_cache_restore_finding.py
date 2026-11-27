# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runner container whose cache mounts held entries the runner user does not own (OMN-26113)."""

from pydantic import BaseModel, ConfigDict


class ModelRunnerCacheRestoreFinding(BaseModel):
    """A container, the cache mount paths the check looked at, and the outcome of the restore."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    container: str
    paths: tuple[str, ...]
    outcome: str  # restored | failed
    detail: str = ""
