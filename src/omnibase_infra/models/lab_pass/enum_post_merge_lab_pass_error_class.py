# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""How one post-merge lab-pass check failed.

Ticket: OMN-20772
"""

from __future__ import annotations

from enum import StrEnum


class EnumPostMergeLabPassErrorClass(StrEnum):
    """The class of a check's failure, carried beside its outcome.

    ``TRANSPORT`` means the probe never got an answer (a refused connection, a
    timeout, an unreachable broker): it says nothing about the thing probed.
    ``ASSERTION`` means the probe read an answer and the answer was wrong.
    ``INDETERMINATE`` means the check could not be established either way.
    ``NONE`` is the class of a passing check.
    """

    NONE = "none"
    ASSERTION = "assertion"
    TRANSPORT = "transport"
    INDETERMINATE = "indeterminate"


__all__ = ["EnumPostMergeLabPassErrorClass"]
