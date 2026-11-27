# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verdict on a build request's git ref (OMN-26724)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_build_repository_ref_verdict import (
    EnumBuildRepositoryRefVerdict,
)


class ModelBuildRepositoryRefDecision(BaseModel):
    """The typed verdict and, when it refuses, the reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumBuildRepositoryRefVerdict
    reason: str = ""

    @property
    def refuses(self) -> bool:
        return self.verdict is EnumBuildRepositoryRefVerdict.MISSING
