# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""How a delegation's task class was decided (OMN-18305).

Before OMN-18305 the CLI chose a class from a hardcoded keyword table and
told the customer nothing: a prose standup was filed as ``test``, which
disarms the prose quality checks, and no surface said a class had been
chosen at all. Recording HOW the class was decided is what makes that
visible at the terminal and on the run artifacts.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["EnumTaskTypeResolution"]


class EnumTaskTypeResolution(StrEnum):
    """How this run's task class was decided, for the receipt and for stderr."""

    #: The caller passed ``--task-type``. Never second-guessed.
    EXPLICIT = "explicit"
    #: A declared contract predicate claimed the prompt.
    CONTRACT = "contract"
    #: No predicate claimed it; the declared fallback was used, and said so.
    FALLBACK = "fallback"
