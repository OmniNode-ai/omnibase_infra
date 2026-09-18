# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Why the off-registry omnimarket drift comparison reached its verdict
(OMN-17255)."""

from __future__ import annotations

from enum import StrEnum

__all__ = ["EnumOffRegistryReason"]


class EnumOffRegistryReason(StrEnum):
    """Reason tokens for an off-registry verdict.

    Every value is a single token with no spaces: the line these appear on is
    machine-read.
    """

    PACKAGED_PINS_SATISFIED = "packaged_pins_satisfied"
    PACKAGED_PIN_UNSATISFIED = "packaged_pin_unsatisfied"
    #: No anchor distribution could be read at all -- neither omnimarket nor
    #: omnibase_infra. Reaching this from the real CLI would mean the running
    #: distribution cannot read its own metadata.
    NO_PACKAGED_PIN_ANCHOR = "no_packaged_pin_anchor"
    #: An anchor WAS found and declares no omni-internal requirement that
    #: applies to this environment. A different fact from an absent anchor, and
    #: told apart so a reader knows which half to go and fix.
    NO_APPLICABLE_PACKAGED_PINS = "no_applicable_packaged_pins"
