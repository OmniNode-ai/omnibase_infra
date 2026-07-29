# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed status values accepted from ownership evidence manifests."""

from enum import StrEnum, unique


@unique
class EnumOwnershipEvidenceStatus(StrEnum):
    """Finite evidence states; only successful terminal states satisfy the gate."""

    BLOCKED = "blocked"
    COMPLETE = "complete"
    FAIL = "fail"
    FAILED = "failed"
    PASS = "pass"
    PASSED = "passed"
    PENDING = "pending"
    VERIFIED = "verified"


__all__ = ["EnumOwnershipEvidenceStatus"]
