# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one standalone compose recreate."""

from __future__ import annotations

from enum import StrEnum


class EnumStandaloneComposeRecreateOutcome(StrEnum):
    RECREATED = "recreated"
    REFUSED = "refused"
    FAILED = "failed"
    SCOPE_VIOLATION = "scope-violation"
