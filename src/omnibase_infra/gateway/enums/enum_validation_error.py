# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Envelope validation error codes for runtime gateway.

This module defines error codes returned by ServiceEnvelopeValidator when
envelope validation fails. Each error code identifies a specific failure
condition in the validation pipeline.

Exports:
    EnumValidationErrorCode: Error codes for envelope signature and realm validation.
"""

from __future__ import annotations

from enum import Enum


class EnumValidationErrorCode(str, Enum):
    """Validation error codes for envelope validation failures.

    These error codes identify specific failure conditions during envelope
    validation, enabling precise error handling and debugging.

    Attributes:
        REALM_MISMATCH: Envelope realm does not match expected gateway realm.
        UNKNOWN_SIGNER: Signer runtime_id not found in trusted public keys.
        HASH_MISMATCH: Recomputed payload hash differs from signature's payload_hash.
        INVALID_SIGNATURE: Ed25519 signature verification failed.
        MISSING_SIGNATURE: Envelope has no signature and reject_unsigned is enabled.
    """

    REALM_MISMATCH = "REALM_MISMATCH"
    UNKNOWN_SIGNER = "UNKNOWN_SIGNER"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    MISSING_SIGNATURE = "MISSING_SIGNATURE"


__all__ = ["EnumValidationErrorCode"]
