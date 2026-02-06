# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Gateway Utilities Module.

This module provides utility functions for gateway operations, including
key loading for envelope signing and validation.

Exports:
    load_private_key_from_pem: Load Ed25519 private key from PEM file
    load_public_key_from_pem: Load Ed25519 public key from PEM file
"""

from omnibase_infra.gateway.utils.util_key_loader import (
    load_private_key_from_pem,
    load_public_key_from_pem,
)

__all__: list[str] = [
    "load_private_key_from_pem",
    "load_public_key_from_pem",
]
