# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner health snapshot effect — contract stub for topic provisioning."""

from pydantic import BaseModel


class ModelRunnerHealthSnapshotInput(BaseModel):
    """Stub input model for runner health snapshot emission."""


class ModelRunnerHealthSnapshotOutput(BaseModel):
    """Stub output model for runner health snapshot emission."""


__all__ = ["ModelRunnerHealthSnapshotInput", "ModelRunnerHealthSnapshotOutput"]
