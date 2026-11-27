# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Why the engine found no dispatcher for a message (OMN-14492)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumDlqFailureClass

__all__ = ["ModelNoDispatcherRefusal"]


class ModelNoDispatcherRefusal(BaseModel):
    """Why the engine found no dispatcher for a message (OMN-14492)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: EnumDlqFailureClass
    error_code: EnumCoreErrorCode
    error_message: str
    validation_detail: str | None = None
