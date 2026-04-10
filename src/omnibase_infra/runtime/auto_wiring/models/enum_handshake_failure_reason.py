# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handshake failure reason enum (OMN-7657)."""

from __future__ import annotations

from enum import Enum


class HandshakeFailureReason(str, Enum):
    """Structured reasons for handshake failure."""

    TIMEOUT = "timeout"
    RESOLUTION_FAILED = "resolution_failed"
    DB_OWNERSHIP = "db_ownership"
    SCHEMA_FINGERPRINT = "schema_fingerprint"
    TCP_PROBE_FAILED = "tcp_probe_failed"
    HOOK_EXCEPTION = "hook_exception"
    HOOK_RETURNED_FAILURE = "hook_returned_failure"
