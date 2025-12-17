# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Handler Payload Model.

This module provides the Pydantic model for Consul handler response payloads.

Note on payload typing:
    Consul operation payloads are typed as dict[str, object] because:
    1. Different operations return different payload structures (KV get vs register)
    2. Field names and types vary by operation
    3. The handler returns generic payloads that callers must interpret

    For strongly-typed domain models, callers should map these generic
    payloads to their specific Pydantic models after retrieval.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConsulHandlerPayload(BaseModel):
    """Payload containing Consul operation results.

    Attributes:
        data: Operation-specific result data as key->value dictionary.
            Field types vary by operation (str, int, bool, list, etc.).

    Example:
        >>> payload = ModelConsulHandlerPayload(
        ...     data={"found": True, "key": "mykey", "value": "myvalue"},
        ... )
        >>> print(payload.data["found"])
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    data: dict[str, object] = Field(
        description="Operation-specific result data as key->value dictionary",
    )


__all__: list[str] = ["ModelConsulHandlerPayload"]
