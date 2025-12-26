# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Operation Type Enum.

This module provides the discriminator enum for Consul operation payload types.
Each operation type corresponds to a specific payload model in the discriminated union.
"""

from __future__ import annotations

from enum import Enum


class EnumConsulOperationType(str, Enum):
    """Discriminator for Consul operation payload types.

    Each operation type corresponds to a specific payload model.
    """

    KV_GET_FOUND = "kv_get_found"
    KV_GET_NOT_FOUND = "kv_get_not_found"
    KV_GET_RECURSE = "kv_get_recurse"
    KV_PUT = "kv_put"
    REGISTER = "register"
    DEREGISTER = "deregister"


__all__: list[str] = ["EnumConsulOperationType"]
