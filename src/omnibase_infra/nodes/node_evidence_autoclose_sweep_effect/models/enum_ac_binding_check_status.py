# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Status of the check a binding row names (OMN-18056).

Mirrors the four dod_verify per-check statuses this sweep can be handed, plus
the honest fifth: a status the verifier reported that this enum does not know.
Typed rather than a bare string because the whole point of a binding row is
that ONE of these values — ``VERIFIED`` — discharges a criterion and the rest
do not, and a comparison that decides that must not be a string literal.

Deliberately NOT imported from omnimarket's ``EnumEvidenceCheckStatus``: that
package is not a dependency of this repo and the sweep runs on a GitHub Actions
runner that has no omnimarket checkout. The values are copied and the copy is
declared, rather than a cross-repo import being invented for five strings.
"""

from __future__ import annotations

from enum import StrEnum


class EnumAcBindingCheckStatus(StrEnum):
    """What the declaring check's dod_verify run concluded."""

    #: Executed, passed, and its exit status could depend on the product.
    #: The ONLY value that discharges an acceptance criterion.
    VERIFIED = "verified"
    #: Executed and exited 0 in a way it could not have avoided (OMN-15391).
    #: A verdict, not a proof — it discharges nothing.
    NON_PROBATIVE = "non_probative"
    #: Never ran. Proves nothing in either direction.
    SKIPPED = "skipped"
    #: Ran and went red. Refused upstream by the flip predicate, and recorded
    #: here so a row is never silently absent.
    FAILED = "failed"
    #: A later item in the same contract supersedes this one.
    SUPERSEDED = "superseded"
    #: The verifier reported a status this enum does not know. Never inferred
    #: and never coerced to a neighbour: an unrecognised status is exactly the
    #: case where guessing would be a Done flip on evidence nobody read.
    UNKNOWN = "unknown"

    @classmethod
    def from_verdict(cls, raw: str) -> EnumAcBindingCheckStatus:
        """Parse a verdict's status string, falling back to UNKNOWN."""
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.UNKNOWN


__all__ = ["EnumAcBindingCheckStatus"]
