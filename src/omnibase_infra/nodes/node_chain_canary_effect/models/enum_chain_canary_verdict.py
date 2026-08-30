# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Terminal verdict for one event-chain canary run (OMN-16773)."""

from __future__ import annotations

from enum import StrEnum


class EnumChainCanaryVerdict(StrEnum):
    """Exactly one verdict is recorded per canary run.

    Only ``GREEN`` and ``SKIPPED_DISABLED`` are non-failing. Everything
    else fails the workflow run, because a canary that reports a dead
    chain quietly is the failure mode this node exists to end.

    The failure values are deliberately NOT collapsed into a single "red".
    ``QUARANTINED`` and ``TERMINAL_MISSING`` are both true at once in the
    OMN-16767 incident, and they are different pages: the first names a
    handler refusing its own event, the second says only that nothing came
    back. Reporting the more specific one is the whole diagnostic value.

    A ``GREEN`` here is a PROBE verdict, not a chain proof. OMN-16025 is a
    five-link gate and this probe has legs for three of them, so read
    ``ModelChainCanaryResult.link_verdicts`` before quoting a colour at
    anybody: ``chain_proof_complete`` is the field that answers "is the
    chain proven", and it is False whenever any link is unproven
    (OMN-16931).
    """

    # A terminal event came back inside the budget, and the run's own
    # correlation id was NOT found in the quarantine sink (or that leg was
    # deliberately not configured — see EnumQuarantineCheckStatus).
    GREEN = "green"
    # The run's correlation id was found in the platform quarantine sink.
    # A handler received the event and refused/errored on it. This outranks
    # TERMINAL_MISSING when both hold, because it names the defect.
    QUARANTINED = "quarantined"
    # The declared terminal topics were read back for the probe's own
    # correlation id, for the whole remaining budget, and it was not there.
    # As of OMN-16931 this verdict is reported ONLY on that evidence — never
    # on an ingress response saying ok=false, and never on an ok=true that
    # carried no terminal field. Both of those are claims about the chain;
    # this verdict is a statement about the bus.
    TERMINAL_MISSING = "terminal_missing"
    # The ingress reported an error AND the terminal is on the bus for this
    # correlation id (OMN-16931, run 33251822642: a provider 429 on an
    # escalation rung, with delegate-skill-completed.v1 published 2s later).
    # RED, because something in the request path failed — but a DIFFERENT
    # page from TERMINAL_MISSING, which would send an operator hunting a
    # dead chain that is in fact alive.
    INGRESS_ERROR_TERMINAL_PRESENT = "ingress_error_terminal_present"
    # The terminal readback was configured but could not be executed. Fails
    # closed on the same terms as QUARANTINE_PROBE_FAILED.
    TERMINAL_READBACK_FAILED = "terminal_readback_failed"
    # No broker was configured for the terminal readback, so the run has no
    # evidence about the terminal at all. Deliberately NOT a fallback to the
    # ingress response: that fallback IS the OMN-16931 defect.
    TERMINAL_READBACK_NOT_CONFIGURED = "terminal_readback_not_configured"
    # The /skill ingress could not be reached at all (connection refused,
    # DNS failure, client-side timeout). The lane, not the chain, is the
    # first thing to look at — dev-lane-liveness.yml is the sibling signal.
    INGRESS_UNREACHABLE = "ingress_unreachable"
    # The quarantine leg was configured but could not be executed (broker
    # unreachable, topic missing, scan error). Fails closed: an unrunnable
    # check is never reported as a passing one.
    QUARANTINE_PROBE_FAILED = "quarantine_probe_failed"
    # ONEX_CHAIN_CANARY_DISABLED was set. Zero I/O was performed.
    SKIPPED_DISABLED = "skipped_disabled"


__all__ = ["EnumChainCanaryVerdict"]
