# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stable id of each board check node_board_probe_effect runs.

The contract's ``board_checks`` block declares each id with its surface class.
A ``lab_hardware`` check's id is also a lab proof check name
(``EnumLabProofCheck``), so a lab proof profile can list it as mandatory.

Ticket: OMN-19930
"""

from __future__ import annotations

from enum import StrEnum


class EnumBoardCheckId(StrEnum):
    """The stable id of one board check."""

    FORWARDER_REFUSED_TOPIC = "forwarder_refused_topic"
    CONSUMER_FLOW = "consumer_flow"


__all__ = ["EnumBoardCheckId"]
