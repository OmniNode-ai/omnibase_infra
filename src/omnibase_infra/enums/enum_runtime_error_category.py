# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Classification categories for runtime container errors (OMN-5649)."""

from __future__ import annotations

from enum import StrEnum


class EnumRuntimeErrorCategory(StrEnum):
    """Classification categories for runtime container errors.

    Used by RuntimeErrorEmitter in monitor_logs.py and downstream
    triage consumer (NodeRuntimeErrorTriageEffect).
    """

    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MISSING_TOPIC = "MISSING_TOPIC"
    CONNECTION = "CONNECTION"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    AUTHENTICATION = "AUTHENTICATION"
    UNKNOWN = "UNKNOWN"
