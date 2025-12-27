# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Operation Type Enum.

This module provides the discriminator enum for Consul operation payload types.
Each operation type corresponds to a specific payload model in the discriminated union.

NOT_FOUND Response Pattern:
    The ``KV_GET_NOT_FOUND`` variant represents a successful Consul KV lookup
    where the requested key does not exist. This is distinct from error-based
    NOT_FOUND handling:

    - **Response variant (this enum)**: Used when key absence is a valid outcome.
      The handler returns a ``KV_GET_NOT_FOUND`` response, allowing callers to
      distinguish between "key exists with value" and "key does not exist"
      without exception handling.

    - **Error-based NOT_FOUND**: Used in ``SecretResolutionError`` when a
      required resource is missing (exceptional condition). See ``infra_errors.py``
      for error-based patterns.

    Example usage for callers:

    .. code-block:: python

        result = await consul_handler.kv_get("config/my-key")
        match result.operation_type:
            case EnumConsulOperationType.KV_GET_FOUND:
                return result.value
            case EnumConsulOperationType.KV_GET_NOT_FOUND:
                return default_config  # Key doesn't exist, use default

    This pattern allows callers to handle key absence as a normal control flow
    case rather than an exception, which is appropriate for optional configuration
    lookups or cache-style access patterns.
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
