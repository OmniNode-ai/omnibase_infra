# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""How the query should match against learning records."""

from __future__ import annotations

from enum import StrEnum


class EnumLearningMatchType(StrEnum):
    """How the query should match against learning records."""

    ERROR_SIGNATURE = "error_signature"
    TASK_CONTEXT = "task_context"
    AUTO = "auto"
