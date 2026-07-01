# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output marker model for the emit daemon runtime topic contract."""

from pydantic import BaseModel, ConfigDict


class ModelEmitDaemonRuntimeOutput(BaseModel):
    """Output marker for the emit daemon runtime topic contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__ = ["ModelEmitDaemonRuntimeOutput"]
