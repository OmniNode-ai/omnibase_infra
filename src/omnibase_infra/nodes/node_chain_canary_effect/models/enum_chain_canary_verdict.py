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
    """

    # A terminal event came back inside the budget, and the run's own
    # correlation id was NOT found in the quarantine sink (or that leg was
    # deliberately not configured — see EnumQuarantineCheckStatus).
    GREEN = "green"
    # The run's correlation id was found in the platform quarantine sink.
    # A handler received the event and refused/errored on it. This outranks
    # TERMINAL_MISSING when both hold, because it names the defect.
    QUARANTINED = "quarantined"
    # The ingress answered, but no terminal event arrived inside the
    # budget — either an explicit dispatch_timeout, or an ok=true response
    # carrying no terminal (the OMN-16027 fail-open shape: a cheerful
    # accept proves nothing about the chain behind it).
    TERMINAL_MISSING = "terminal_missing"
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
