# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from enum import StrEnum


class EnumFailedHeadVerdictClass(StrEnum):
    """The verdict classes a failed dev roll records that a retry of the same head cannot change."""

    RUNTIME_DEGRADED = "runtime_degraded"
    HTTP_ERROR = "http_error"
    PROBE_UNREACHABLE = "probe_unreachable"
    BOOT_FAILED = "boot_failed"
    COMPOSE_UP_TIMED_OUT = "compose_up_timed_out"
