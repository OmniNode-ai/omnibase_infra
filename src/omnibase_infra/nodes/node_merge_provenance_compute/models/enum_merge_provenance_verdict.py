# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether a commit was validated by a successful merge group (OMN-19927)."""

from __future__ import annotations

from enum import StrEnum


class EnumMergeProvenanceVerdict(StrEnum):
    """VALIDATED keeps smart selection; the other two force the full suite."""

    VALIDATED = "VALIDATED"
    UNVALIDATED = "UNVALIDATED"
    UNDECIDABLE = "UNDECIDABLE"


__all__: list[str] = ["EnumMergeProvenanceVerdict"]
