# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The five links of the OMN-16025 delegation-chain gate (OMN-16931)."""

from __future__ import annotations

from enum import StrEnum


class EnumChainLink(StrEnum):
    """One member per acceptance link of OMN-16025, in gate order.

    The canary used to report ONE scalar verdict. OMN-16025 is a FIVE-link
    gate, and this probe has legs for three of them — so a green scalar read
    as "the chain is proven" when it meant "the three things this probe looks
    at are fine". Run 33215999994 (2026-08-28, GREEN) was read exactly that
    way. Naming every link and reporting a status for each makes the unpaid
    links visible in the receipt instead of invisible in the verdict.

    Values are the gate's own wording, not paraphrase, so a receipt can be
    diffed against OMN-16025 without interpretation.
    """

    # "Intent submitted through the live gateway path."
    INGRESS_ACCEPTED = "link_1_intent_through_live_gateway"
    # "Routing decision PUBLISHED and PROJECTED (readback from projection,
    # not logs)." No leg exists — owed by OMN-16963.
    ROUTING_PROJECTED = "link_2_routing_published_and_projected"
    # "Delegated execution completes."
    DELEGATED_EXECUTION = "link_3_delegated_execution_completes"
    # "Emission OUTBOX-CONFIRMED via broker readback (not publish-return)."
    # OMN-16931 is the ticket that gave this link a real leg.
    TERMINAL_ON_BUS = "link_4_emission_outbox_confirmed_via_broker_readback"
    # "Complete ledger chain + replay green through an HONEST tier-2
    # verifier (SKIP != PASS)." No leg exists — owed by OMN-16964.
    LEDGER_REPLAY = "link_5_ledger_chain_replay_honest_tier2_verifier"


__all__ = ["EnumChainLink"]
