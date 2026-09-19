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
    # OMN-15245, narrowed by OMN-16745: a changed path directly in the tests/
    # root that pytest would NOT collect (it matches neither `python_files`
    # pattern) -- a shared helper module such as tests/infrastructure_config.py.
    # It has no containing directory below `tests/`, handing it to pytest
    # collects nothing (exit 5), and any suite in the tree may import it, so
    # the honest answer is the real full suite.
    #
    # A root-level module pytest DOES collect is no longer covered by this
    # reason: it is narrowable to itself, at file grain. See the ruling above
    # `CI_CONTRACT_TEST_ROOT` in detect_test_paths.py.
    CHANGED_TEST_UNNARROWABLE = "changed_test_unnarrowable"
    # OMN-18833: the scripts/-reference scan could not be completed (tests/
    # missing, a directory or file unreadable, a module that is not valid
    # UTF-8). The scan is the only thing that knows which tests exercise a
    # changed script outside the two tests/**/scripts/ prefixes, so a scan
    # that did not run leaves the selector unable to prove anything about a
    # scripts/ diff. Fail CLOSED: run the real full suite rather than emit a
    # narrowed selection derived from a partial walk.
    SCRIPT_REFERENCE_SCAN_FAILED = "script_reference_scan_failed"
    # OMN-18833: a module sitting directly in the tests/ root that pytest does
    # NOT collect references a changed script. Same shape as
    # CHANGED_TEST_UNNARROWABLE and the same answer -- it has no containing
    # directory below tests/, handing it to pytest collects nothing, and any
    # suite in the tree may import it -- but reached through the reference scan
    # rather than through the diff, so it gets its own reason instead of
    # borrowing one whose name would misreport why the suite escalated.
    SCRIPT_REFERENCE_UNNARROWABLE = "script_reference_unnarrowable"


# A selectable pytest target: a directory under the root-collected `tests/`
# tree, a collocated `tests/` directory anywhere in the repo, OR a single test
# MODULE anywhere under `tests/`.
#
# OMN-18833 extended the third alternative from the `tests/` root to any depth.
# The name must still match pytest's `python_files` patterns, so the selector
# can never hand pytest a module it would not collect. The reason OMN-16745
# gave for keeping it root-only -- "nested modules narrow to their own
# directory, which the selector already emits" -- is sound for a module the
# diff CHANGED and false for one the diff merely REFERENCES, which is the
# population the scripts/-reference scan produces. Measured over the last 20
# merged pull requests: that scan emitting parent DIRECTORIES costs +12.3%
# collectable modules, emitting the referencing MODULES costs +0.3%, because a
# single test in `tests/ci/` naming a script drags that whole 241-module
# directory in at directory grain. Both are correct; one is forty times
# cheaper, and a correct-but-expensive selector is the thing operators route
# around.
#
# OMN-16745 added the third alternative. A root-level module is the one
# changed-test shape with no containing directory below `tests/` itself, so
# before this the selector could not name it and escalated the whole diff to
# the full suite (`changed_test_unnarrowable`). File grain is strictly narrower
# than the directory it would otherwise emit and strictly covers the module, so
# this widens what the selector can PROVE, not what it may skip.
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
        pattern=(
            r"^tests(/[A-Za-z0-9_./-]+)?/$"
            r"|^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*/tests/$"
            r"|^tests/([A-Za-z0-9_.-]+/)*"
            r"(test_[A-Za-z0-9_-]*|[A-Za-z0-9_-]*_test)\.py$"
        )
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
