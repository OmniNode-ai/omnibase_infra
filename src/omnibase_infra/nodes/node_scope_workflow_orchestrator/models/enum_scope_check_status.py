# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Status enum for scope-check workflow results."""

from enum import Enum, unique


@unique
class EnumScopeCheckStatus(str, Enum):
    """Status values for scope-check workflow completion."""

    COMPLETE = "complete"
    """Workflow completed successfully."""

    FAILED = "failed"
    """Workflow failed."""

    def __str__(self) -> str:
        return self.value
