# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The next workflow action the reducer requests (OMN-13247, plan §5.2)."""

from __future__ import annotations

from enum import Enum


class EnumCodingAgentIntentKind(str, Enum):
    """The next workflow action the reducer requests."""

    DISPATCH_VALIDATE = "dispatch_validate"
    DISPATCH_INVOKE = "dispatch_invoke"
    DISPATCH_CAPTURE = "dispatch_capture"
    EMIT_TERMINAL = "emit_terminal"
