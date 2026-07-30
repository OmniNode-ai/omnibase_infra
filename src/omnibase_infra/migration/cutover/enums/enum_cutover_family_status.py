# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable state of an independently stoppable cutover family."""

from enum import StrEnum


class EnumCutoverFamilyStatus(StrEnum):
    """Family-local state; one blocked family never blocks an unrelated family."""

    READY = "ready"
    BLOCKED = "blocked"
    CHECKPOINTED = "checkpointed"
    OBSERVING = "observing"
    COMPLETE = "complete"


__all__ = ["EnumCutoverFamilyStatus"]
