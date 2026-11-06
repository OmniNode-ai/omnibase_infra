# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host paths for one existing hook environment's exact reconciliation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelHookVenvReconcileRequest(BaseModel):
    """Reconcile one hook project using the caller's resolved uv binary."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    operation: Literal["hook_venv.reconcile"] = "hook_venv.reconcile"
    workspace_root: Path
    project: Path
    uv_bin: Path
