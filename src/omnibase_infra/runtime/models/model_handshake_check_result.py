# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handshake check result model.

This module provides the ModelHandshakeCheckResult for representing the outcome
of a single plugin handshake validation check during kernel bootstrap.

Related:
    - OMN-2089: Handshake Hardening - Bootstrap Attestation Gate
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelHandshakeCheckResult:
    """Result of a single handshake validation check.

    Attributes:
        check_name: Identifier for the check (e.g., "db_ownership", "schema_fingerprint").
        passed: Whether the check passed.
        message: Human-readable description of the check outcome.
    """

    check_name: str
    passed: bool
    message: str = ""


__all__ = [
    "ModelHandshakeCheckResult",
]
