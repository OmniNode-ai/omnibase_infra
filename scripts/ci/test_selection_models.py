# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic output contract for change-aware test selection."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class EnumFullSuiteReason(StrEnum):
    SHARED_MODULE = "shared_module"
    THRESHOLD_MODULES = "threshold_modules"
    TEST_INFRASTRUCTURE = "test_infrastructure"
    MAIN_BRANCH = "main_branch"
    MERGE_GROUP = "merge_group"
    SCHEDULED = "scheduled"
    FEATURE_FLAG_OFF = "feature_flag_off"
    # OMN-15245: a changed test path that cannot be narrowed below `tests/`
    # itself (a test module living directly in the tests/ root). Emitting
    # "tests/" as a smart selection would run the whole suite on the smart
    # step's split count; escalate to the real full suite instead.
    CHANGED_TEST_UNNARROWABLE = "changed_test_unnarrowable"


# A selectable pytest target: a directory under the root-collected `tests/`
# tree, OR a collocated `tests/` directory anywhere in the repo.
#
# OMN-15410 added the second alternative. The original `tests/`-only pattern
# encoded an assumption that stopped being true when pyproject `testpaths`
# grew to include four collocated roots (scripts/ci/tests/, scripts/tests/,
# scripts/runtime_build/tests/, and the agent_actions root): the selector could
# not emit them, so a narrowed run could never reach them and constructing the
# selection raised a pattern_mismatch ValidationError. The constraint stays
# tight — the final path component must still be `tests`, so the selector can
# never emit an arbitrary source directory to pytest.
TestPath = Annotated[
    str,
    StringConstraints(
        pattern=r"^tests(/[A-Za-z0-9_./-]+)?/$|^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*/tests/$"
    ),
]
ModuleName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", min_length=1),
]


class ModelTestSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # min_length=0 (not 1): a docs-only diff (OMN-14753) legitimately selects
    # no tests -- distinct from the conservative tests/unit/ fallback, which
    # always selects at least one path.
    selected_paths: list[TestPath] = Field(default_factory=list)
    split_count: int = Field(..., ge=1, le=15)
    is_full_suite: bool
    full_suite_reason: EnumFullSuiteReason | None = Field(default=None)
    matrix: list[int] = Field(...)

    @model_validator(mode="after")
    def validate_full_suite_reason(self) -> Self:
        if self.is_full_suite and self.full_suite_reason is None:
            raise ValueError("full_suite_reason required when is_full_suite=True")
        if not self.is_full_suite and self.full_suite_reason is not None:
            raise ValueError("full_suite_reason forbidden when is_full_suite=False")
        if len(self.matrix) != self.split_count:
            raise ValueError(
                f"matrix length {len(self.matrix)} must equal split_count {self.split_count}"
            )
        return self
