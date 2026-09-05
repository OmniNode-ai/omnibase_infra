# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The write mode a sweep run actually resolved to (OMN-17658)."""

from __future__ import annotations

from enum import StrEnum


class EnumEvidenceAutocloseMode(StrEnum):
    """How this run resolved the question "may I write to Linear?".

    ``dry_run`` on the result is a boolean and says only whether a mutation was
    possible. It cannot distinguish the four ways a run reaches "no", and those
    four are not interchangeable in an audit: a halted run did zero I/O, a
    disarmed run did every read and refused the writes, an unarmed scheduled
    run reached every decision under a contract that declines to arm it, and a
    dispatched preview is somebody rehearsing. Reporting the mode is what makes
    "the closer wrote nothing last night" answerable without reading the log.
    """

    # The schedule ran and the contract's `scheduled_apply` armed it.
    APPLY_SCHEDULED = "apply_scheduled"
    # An operator dispatched with the apply box ticked.
    APPLY_DISPATCHED = "apply_dispatched"
    # Every decision was reached; none was written. Either a dispatch with the
    # box unticked, or a scheduled run under `scheduled_apply=False`.
    DRY_RUN = "dry_run"
    # OMN-17658 auto-disarm: an unsafe closer flip was found, so this run
    # refused to write regardless of its arming. See
    # `ModelEvidenceAutocloseSweepResult.disarm_triggered_by`.
    DISARMED = "disarmed"
    # ONEX_AUTOCLOSE_DISABLED was set: zero I/O, no decisions at all.
    HALTED = "halted"


__all__ = ["EnumEvidenceAutocloseMode"]
