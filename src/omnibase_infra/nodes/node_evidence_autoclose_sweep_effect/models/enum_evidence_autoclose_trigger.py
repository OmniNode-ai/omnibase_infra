# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What launched this sweep run (OMN-17658)."""

from __future__ import annotations

from enum import StrEnum


class EnumEvidenceAutocloseTrigger(StrEnum):
    """The event class that started the run, as a typed request field.

    Before OMN-17658 this fact lived only in the workflow, as one
    ``github.event_name == 'schedule'`` disjunct inside the ``SWEEP_APPLY``
    expression, and it was that disjunct — not anything the node declared —
    that decided whether an unattended run wrote to Linear. So the arming
    authority for every write nobody was watching was a string comparison in a
    YAML expression, invisible to the contract, untyped, and unreachable from
    any test that did not parse the workflow file.

    Making it a request field moves the *fact* (what launched me) into the
    payload and leaves the *decision* (may a run of this class write) to
    ``ModelEvidenceAutocloseSweepRequest.scheduled_apply``, which the contract
    declares. Two fields rather than one because they answer different
    questions and only one of them is a policy.

    ``DISPATCH`` is the default deliberately: a caller that names nothing is
    not the schedule, so an un-named construction can never pick up the
    unattended arming authority by omission.
    """

    SCHEDULE = "schedule"
    DISPATCH = "dispatch"


__all__ = ["EnumEvidenceAutocloseTrigger"]
