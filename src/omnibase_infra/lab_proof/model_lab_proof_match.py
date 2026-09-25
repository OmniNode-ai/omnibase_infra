# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Which pull requests a profile variant takes.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_match_predicate import (
    EnumLabProofMatchPredicate,
)


class ModelLabProofMatch(BaseModel):
    """Base branches, path globs, and an optional shared predicate.

    A pull request matches when its base branch is listed, at least one changed
    path matches an include glob and is not excluded, and the predicate (when
    set) holds. The predicate names the shared runtime-affecting classifier
    (OMN-19318); it is a name here, never a copy of the classifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_branches: tuple[str, ...] = Field(min_length=1)
    include_globs: tuple[str, ...] = Field(min_length=1)
    exclude_globs: tuple[str, ...] = ()
    predicate: EnumLabProofMatchPredicate | None = None


__all__ = ["ModelLabProofMatch"]
