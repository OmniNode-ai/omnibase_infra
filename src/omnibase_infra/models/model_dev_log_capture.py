# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One failing container's captured log, kept before the recovery recreates it."""

from pydantic import BaseModel, ConfigDict, Field


class ModelDevLogCapture(BaseModel):
    """The kept tail of one container's log, with how complete it is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: str
    container: str
    bytes_total: int = Field(ge=0)
    bytes_kept: int = Field(ge=0)
    complete: bool
    redacted: bool
    sha256: str = Field(min_length=64, max_length=64)
    artifact_path: str
    error: str | None
    log: str
