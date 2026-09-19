# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed reason a PR was withheld from both merge tracks (OMN-18823).

The classifier previously carried its reason only as free prose, which nothing
downstream can branch on and no test can assert without matching a sentence.
A SKIP that means "a person is holding this" and a SKIP that means "it is a
draft" are different facts and must be distinguishable by type.
"""

from __future__ import annotations

from enum import StrEnum


class EnumClassifySkipReason(StrEnum):
    """Why a PR was not admitted to Track A or Track B."""

    NOT_SKIPPED = "not_skipped"
    """The PR was classified into a track; it was not withheld."""

    DRAFT = "draft"
    """The PR is a draft."""

    AUTO_MERGE_ENABLED = "auto_merge_enabled"
    """Auto-merge is already armed, so the sweep has nothing to do."""

    COLLABORATOR_EXCLUDED = "collaborator_excluded"
    """The PR is assigned to, or awaiting review from, a declared collaborator."""


__all__ = ["EnumClassifySkipReason"]
