# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether the dev runtime may be recreated now (OMN-25912)."""

from enum import StrEnum


class EnumDevRecreateVerdict(StrEnum):
    """Whether the dev runtime may be recreated now."""

    RECREATE = "recreate"
    HOLD = "hold"
