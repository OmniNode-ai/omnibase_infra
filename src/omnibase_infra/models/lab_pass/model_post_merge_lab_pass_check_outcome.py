# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""One check of a post-merge lab pass, with its outcome and error class.

Ticket: OMN-20772
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.lab_pass.enum_post_merge_lab_pass_error_class import (
    EnumPostMergeLabPassErrorClass,
)


class ModelPostMergeLabPassCheckOutcome(BaseModel):
    """What one probe concluded and, when it did not pass, how it failed."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(min_length=1)
    outcome: Literal["pass", "fail", "indeterminate"]
    error_class: EnumPostMergeLabPassErrorClass
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def _error_class_matches_outcome(self) -> ModelPostMergeLabPassCheckOutcome:
        passing = self.outcome == "pass"
        has_class = self.error_class is not EnumPostMergeLabPassErrorClass.NONE
        if passing == has_class:
            msg = (
                f"check {self.name!r}: outcome={self.outcome!r} with "
                f"error_class={self.error_class.value!r}; a passing check has "
                "class 'none' and every other outcome names a failure class"
            )
            raise ValueError(msg)
        return self


__all__ = ["ModelPostMergeLabPassCheckOutcome"]
