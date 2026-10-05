# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How a delegation that left no terminal ended, as its receipt reports it (OMN-17427).

``onex delegate`` writes ``receipt.json`` for every way a run can end. Most
write a terminal; these four are the ones that end without one, and the
receipt's ``terminal_class`` says which.

.. versionadded:: OMN-17427
"""

from __future__ import annotations

from enum import Enum

__all__ = ["EnumDelegateIncompleteClass"]


class EnumDelegateIncompleteClass(str, Enum):
    """The class of a receipt written for a run with no delegation terminal."""

    IN_FLIGHT = "in_flight"
    """Filed before the terminal wait; what a caller's SIGKILL leaves behind.

    Not an outcome: a finished process replaces it, and one found after its
    process has gone means the run ended without a terminal.
    """

    TIMEOUT = "timeout"
    """The wait for the terminal ended at this CLI's own bound, or the runtime
    recorded ``timeout`` and no terminal payload."""

    INTERRUPTED = "interrupted"
    """The caller ended the process with a signal it can be answered on
    (``SIGTERM``, ``SIGHUP``, ``SIGINT``): its own timeout, or Ctrl-C."""

    FAILED = "failed"
    """The run ended and its receipt holds no terminal payload to write from."""
