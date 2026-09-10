# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The tier-2 verifier's own word about one hop (OMN-16964)."""

from __future__ import annotations

from enum import StrEnum


class EnumTierTwoVerdict(StrEnum):
    """PASS, FAIL, or the SKIP that OMN-16025 forbids counting as either.

    The gate text this discharges reads *"complete ledger chain + replay green
    through an HONEST tier-2 verifier (SKIP != PASS)"*. The failure mode it
    names is a verifier that reports a pass because it never ran the check.

    ``SKIP`` is therefore a first-class member rather than an absence: a
    verifier with no declared topology to check a hop against has established
    nothing, and "established nothing" is not "found nothing wrong". The values
    are the literal tokens ``node_chain_canary_effect`` compares against when
    it reads ``ledger_chain.verifier_verdict`` back, so they are wire
    identifiers and must not be renamed for cosmetic reasons.
    """

    # The declared topology reached this hop and the observation conformed.
    PASS = "pass"
    # The declared topology reached this hop and the observation contradicted it.
    FAIL = "fail"
    # There was no declaration reaching this hop, so no check ran. NOT a pass.
    SKIP = "skip"


__all__ = ["EnumTierTwoVerdict"]
