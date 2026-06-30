# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input marker model for the emit daemon runtime topic contract."""

from pydantic import BaseModel, ConfigDict


class ModelEmitDaemonRuntimeInput(BaseModel):
    """Input marker for the emit daemon runtime topic contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__ = ["ModelEmitDaemonRuntimeInput"]
