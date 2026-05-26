# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handshake check result model.

The ModelHandshakeCheckResult for representing the outcome
of a single plugin handshake validation check during kernel bootstrap.

Related:
    - OMN-2089: Handshake Hardening - Bootstrap Attestation Gate
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ModelHandshakeCheckResult",
]


class ModelHandshakeCheckResult(BaseModel):
    """Result of a single handshake validation check.

    Attributes:
        check_name: Identifier for the check (e.g., "db_ownership", "schema_fingerprint").
        passed: Whether the check passed.
        message: Human-readable description of the check outcome.
    """

    model_config = ConfigDict(frozen=False)

    check_name: str  # pattern-ok: handshake check identifier, not an entity name
    passed: bool
    message: str = ""
