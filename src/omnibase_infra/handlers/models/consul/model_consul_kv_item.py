# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul KV Item Model.

This module provides the model for a single KV item from a recurse query.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelConsulKVItem(BaseModel):
    """Single KV item from recurse query.

    Attributes:
        key: The KV key path.
        value: The value stored at the key (decoded from bytes).
        flags: Optional user-defined flags associated with the key.
        modify_index: The Consul modify index for optimistic locking.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    key: str = Field(description="The KV key path")
    value: str | None = Field(description="The value stored at the key")
    flags: int | None = Field(default=None, description="User-defined flags")
    modify_index: int | None = Field(
        default=None, description="Consul modify index for CAS operations"
    )


__all__: list[str] = ["ModelConsulKVItem"]
