# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where ``onex delegate`` ended before anything was dispatched (OMN-19006).

``receipt.json`` is the one surface every caller is told to read. These are the
stages at which the command can end without a dispatch, and the receipt's
``refusal.stage`` names which one did.

.. versionadded:: OMN-19006
"""

from __future__ import annotations

from enum import Enum

__all__ = ["EnumDelegateRefusalStage"]


class EnumDelegateRefusalStage(str, Enum):
    """The stage at which a delegation ended without being dispatched."""

    ARGUMENT_PARSING = "argument_parsing"
    """The arguments never parsed: an unknown flag, a missing PROMPT, a value
    outside a flag's choices. Nothing of the request had been read."""

    REFUSED_BEFORE_DISPATCH = "refused_before_dispatch"
    """The arguments parsed and the request was refused by a check that runs
    before the broker is touched: a malformed flag value, an unreadable
    contract, a lane that does not resolve, a drifted install."""

    UNHANDLED_ERROR = "unhandled_error"
    """The command raised something no check names. The receipt carries the
    error's type and message so the run is not a bare traceback."""

    EXIT_WITHOUT_RECEIPT = "exit_without_receipt"
    """The command exited non-zero, or was interrupted, on a path that wrote no
    receipt of its own. A new early return lands here until it is given one."""
