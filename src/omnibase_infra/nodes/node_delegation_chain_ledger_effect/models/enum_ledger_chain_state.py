# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Chain-level evidence labels shared by the writer and canary (OMN-17427)."""

from enum import StrEnum


class EnumLedgerChainState(StrEnum):
    """Wire tokens written to ledger_chain.chain_state and read by node_chain_canary_effect.

    COMPLETE means every declared hop was observed; replay and verification
    still determine whether it passed. INCOMPLETE means declared hops are
    missing. IN_PROCESS_TERMINAL_ONLY means only the declared terminal was
    observed, published by a contract-declared in-process producer with no
    parent. That path publishes no upstream hops: neither a pass nor a fault.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    IN_PROCESS_TERMINAL_ONLY = "in_process_terminal_only"


__all__ = ["EnumLedgerChainState"]
