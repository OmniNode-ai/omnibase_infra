# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether the build repository holds the commit a build request names (OMN-26724)."""

from enum import StrEnum


class EnumBuildRepositoryRefVerdict(StrEnum):
    """The verdict on a request's git ref against the build repository."""

    #: The clone already held the commit.
    HELD = "held"
    #: The clone lacked the commit and the one fetch brought it.
    HELD_AFTER_FETCH = "held_after_fetch"
    #: The clone lacked the commit and a fetch that succeeded did not bring it: refused.
    MISSING = "missing"
    #: The fetch did not answer, so absence is not proven; the build proceeds as requested.
    UNVERIFIABLE = "unverifiable"
