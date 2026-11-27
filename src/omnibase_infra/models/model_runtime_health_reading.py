# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One curl reading of a runtime /health endpoint (OMN-18640)."""

from pydantic import BaseModel, ConfigDict


class ModelRuntimeHealthReading(BaseModel):
    """The exit code and output of the deploy agent's /health probe."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    returncode: int
    stdout: str
    stderr: str
