# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of a gated deploy-agent replacement."""

from pydantic import BaseModel, ConfigDict


class ModelDeployAgentReplacementResult(BaseModel):
    """Process output after admission; refusals raise before any replacement."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    returncode: int
    stdout: str = ""
    stderr: str = ""
    error_message: str | None = None
