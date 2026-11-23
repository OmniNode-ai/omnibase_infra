# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_failed_head_verdict_class import (
    EnumFailedHeadVerdictClass,
)


class ModelFailedHeadRefusal(BaseModel):
    """An idle build of a dev head the lane's last dev job already failed to boot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The omnibase_infra dev head the idle build would stage.
    head: str
    #: The newest terminal dev job, which failed on that head.
    failed_job: UUID
    verdict_class: EnumFailedHeadVerdictClass
