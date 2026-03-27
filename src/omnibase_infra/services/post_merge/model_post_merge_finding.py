# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic model for a single post-merge check finding.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.enum_finding_severity import (
    EnumFindingSeverity,
)


class ModelPostMergeFinding(BaseModel):
    """A single finding from a post-merge check stage.

    Related Tickets:
        - OMN-6727: post-merge consumer chain
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    stage: EnumCheckStage = Field(description="Which check stage produced this finding")
    severity: EnumFindingSeverity = Field(description="Finding severity")
    title: str = Field(description="Short summary of the finding")
    description: str = Field(description="Detailed description of the finding")
    file_path: str | None = Field(
        default=None,
        description="File path related to the finding, if applicable",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="Line number related to the finding, if applicable",
    )


__all__ = ["ModelPostMergeFinding"]
