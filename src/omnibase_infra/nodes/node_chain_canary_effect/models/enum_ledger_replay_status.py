# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Outcome of the ledger-chain assembly, replay and tier-2 verify (OMN-16964)."""

from __future__ import annotations

from enum import StrEnum


class EnumLedgerReplayStatus(StrEnum):
    """Did the probe's own chain assemble, replay, and verify honestly?

    This is the leg OMN-16964 adds. OMN-16025 link 5 reads *"Complete ledger
    chain + replay green through an HONEST tier-2 verifier (SKIP != PASS)"*,
    and the OMN-16025 verdict recorded it as flatly unexercised: the canary
    assembles no ledger chain, runs no replay, and invokes no verifier.

    ``VERIFIER_SKIPPED`` is the member the word "honest" in that gate text
    exists for. The failure this link catches is a verifier that reports a
    pass because it never ran the check — the same shape as OMN-16773's
    ``SKIPPED_NOT_CONFIGURED is not CLEAN`` and OMN-16931's verdict-from-claim
    defect. A SKIP counted as green re-creates precisely the defect the gate
    was written to prevent, so it is kept as its own non-passing member rather
    than folded into either PASS or ERROR.

    ``CHAIN_INCOMPLETE`` is likewise distinct from ``REPLAY_FAILED``: a gap in
    the assembled chain means the evidence was never there to replay, whereas
    a failed replay means complete evidence that did not reproduce. Both are
    non-passing, and they send you to different places.
    """

    # Complete chain, replayed, and a tier-2 verifier that actually ran and
    # passed. This is what discharges OMN-16025 link 5.
    VERIFIED = "verified"
    # The chain was assembled but a hop is missing, so there is nothing
    # complete to replay. No gap is ever tolerated silently.
    CHAIN_INCOMPLETE = "chain_incomplete"
    # A complete chain was replayed and the replay was not green.
    REPLAY_FAILED = "replay_failed"
    # The tier-2 verifier ran and returned SKIP. NOT a pass, and never
    # counted as one.
    VERIFIER_SKIPPED = "verifier_skipped"
    # The chain could not be read, the replay could not be driven, or the
    # verifier could not be invoked. Fails closed.
    ERROR = "error"
    # No ledger source configured for this run. No claim is made about the
    # chain, and therefore no green is available.
    SKIPPED_NOT_CONFIGURED = "skipped_not_configured"


__all__ = ["EnumLedgerReplayStatus"]
