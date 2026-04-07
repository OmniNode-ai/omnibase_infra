# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structured reasons for handshake failure in auto-wiring (OMN-7657)."""

from __future__ import annotations

from enum import Enum


class HandshakeFailureReason(str, Enum):
    """Structured reasons for handshake failure.

    Used in quarantine records to classify why a contract failed its
    pre-wiring readiness check. Each reason maps to a distinct
    operational response (e.g., DB_OWNERSHIP requires DBA intervention,
    TCP_PROBE_FAILED may self-heal on retry).
    """

    TIMEOUT = "timeout"
    RESOLUTION_FAILED = "resolution_failed"
    DB_OWNERSHIP = "db_ownership"
    SCHEMA_FINGERPRINT = "schema_fingerprint"
    TCP_PROBE_FAILED = "tcp_probe_failed"
    HOOK_EXCEPTION = "hook_exception"
    HOOK_RETURNED_FAILURE = "hook_returned_failure"


__all__ = ["HandshakeFailureReason"]
