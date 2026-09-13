# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the runtime owes a state_io workflow whose process went away (OMN-18296)."""

from __future__ import annotations

from enum import StrEnum


class EnumRuntimeRestartPolicy(StrEnum):
    """What the runtime owes a workflow whose owning process went away.

    One member, because one behaviour is implemented. A ``resume`` member would
    read as a supported choice while nothing resumes a lost leg: the in-flight
    inference call is gone with the process, its consumer offset is committed,
    and no persisted step exists to resume FROM. Re-publishing a persisted
    outbox batch — the one thing that IS recoverable — already happens
    unconditionally on every sweep and is not a policy choice. Add a member when
    a second behaviour exists, never before.
    """

    TERMINALISE_FAILED = "terminalise_failed"
    """Past the bound, emit the contract's terminal FAILURE event and close the row."""


__all__: list[str] = ["EnumRuntimeRestartPolicy"]
