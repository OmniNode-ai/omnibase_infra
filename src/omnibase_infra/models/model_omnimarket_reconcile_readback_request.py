# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Workspace whose installed package and clone must be read again."""

from pydantic import BaseModel, ConfigDict


class ModelOmnimarketReconcileReadbackRequest(BaseModel):
    """The canonical workspace root; empty when unavailable."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    workspace_root: str = ""
