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
    five-link gate and this probe has legs for four of them, so read
    ``ModelChainCanaryResult.link_verdicts`` before quoting a colour at
    anybody: ``chain_proof_complete`` is the field that answers "is the
    chain proven", and it is False whenever any link is unproven
    (OMN-16931).

    As of OMN-16963 a non-passing link 2 also fails the SCALAR verdict, not
    only ``chain_proof_complete``. Before that, an unconfigured or stranded
    projection still produced ``GREEN`` with ``success=True`` — the same
    over-reading this node exists to end, reproduced one level up at the
    summary rather than at the link.
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
    # -- link 2, the projection readback (OMN-16963) --------------------
    # These mirror the terminal readback's four non-passing outcomes rather
    # than collapsing into one, for the same reason the terminal ones are
    # not collapsed: they send an operator to different layers. They are
    # reported only when the terminal IS on the bus, because that is the
    # disagreement OMN-14843 measured — the topic layer healthy at the same
    # moment the FSM was stranded — and a missing terminal is the larger
    # fact when both hold.
    #
    # The projection carries a row for this correlation id and it is not a
    # terminal FSM state. The chain moved the event and the FSM did not
    # finish with it. This is the OMN-14843 shape.
    PROJECTION_STRANDED = "projection_stranded"
    # The projection was readable and carries NO row for this correlation
    # id. Distinct from STRANDED on purpose: a row that stopped mid-FSM is a
    # projection defect, an absent row may equally be a publish that never
    # happened, and they send you to different layers.
    PROJECTION_ROW_ABSENT = "projection_row_absent"
    # The projection readback was configured but could not be executed.
    # Fails closed on the same terms as TERMINAL_READBACK_FAILED.
    PROJECTION_READBACK_FAILED = "projection_readback_failed"
    # No DSN was configured for the projection readback, so the run has no
    # evidence about link 2 at all. Deliberately NOT green-with-a-caveat the
    # way an unconfigured quarantine leg is: quarantine is a supplementary
    # check, link 2 is one of the five OMN-16025 chain links, and a run that
    # cannot see it is the three-links-rendered-as-five defect this ticket
    # exists to end.
    PROJECTION_READBACK_NOT_CONFIGURED = "projection_readback_not_configured"
    # -- link 5, the ledger chain replay (OMN-16964) --------------------
    # Mirrors link 2's treatment for the same reason: the per-link status was
    # honest while the scalar verdict was not. Reported only when links 2 and
    # 4 have both passed, so the ledger is the last thing left to disprove.
    #
    # The assembled chain has a gap, so there is no complete chain to replay.
    LEDGER_CHAIN_INCOMPLETE = "ledger_chain_incomplete"
    # A COMPLETE chain was replayed and the replay was not green. Distinct
    # from CHAIN_INCOMPLETE: there the replay never had material to run on.
    # Named FAILED rather than NOT_GREEN to match the contract member that
    # landed in #3072 and EnumLedgerReplayStatus.REPLAY_FAILED one level
    # down — one vocabulary across the status enum, the verdict enum and the
    # contract, rather than three synonyms for the same fact.
    LEDGER_REPLAY_FAILED = "ledger_replay_failed"
    # The tier-2 verifier returned SKIP. It was pointed at this run and
    # checked nothing. This is the verdict OMN-16025's "SKIP != PASS" wording
    # exists to make expressible: unlike an unconfigured leg, this one ran.
    LEDGER_VERIFIER_SKIPPED = "ledger_verifier_skipped"
    # The chain could not be assembled, replayed or verified at all. Fails
    # closed on the same terms as the other unrunnable checks.
    LEDGER_REPLAY_UNREADABLE = "ledger_replay_unreadable"
    # No source was configured for the ledger replay, so the run has no
    # evidence about link 5. Red for the same reason link 2's is: link 5 is
    # one of the five OMN-16025 chain links, not a supplementary check.
    LEDGER_REPLAY_NOT_CONFIGURED = "ledger_replay_not_configured"
    # ONEX_CHAIN_CANARY_DISABLED was set. Zero I/O was performed.
    SKIPPED_DISABLED = "skipped_disabled"


__all__ = ["EnumChainCanaryVerdict"]
