# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul KV Get Found Payload Model.

This module provides the payload model for consul.kv_get when key is found (single key mode).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelConsulKVGetFoundPayload(BaseModel):
    """Payload for consul.kv_get when key is found (single key mode).

    Attributes:
        operation_type: Discriminator literal "kv_get_found".
        found: Always True for this payload type.
        key: The KV key path that was queried.
        value: The value stored at the key (decoded from bytes).
        flags: Optional user-defined flags associated with the key.
        modify_index: The Consul modify index for optimistic locking.
        index: The Consul response index for blocking queries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", coerce_numbers_to_str=False)

    operation_type: Literal["kv_get_found"] = Field(
        default="kv_get_found", description="Discriminator for payload type"
    )
    found: Literal[True] = Field(
        default=True, description="Indicates the key was found"
    )
    key: str = Field(description="The KV key path that was queried")
    value: str | None = Field(description="The value stored at the key")
    flags: int | None = Field(default=None, description="User-defined flags")
    modify_index: int | None = Field(
        default=None, description="Consul modify index for CAS operations"
    )
    index: int = Field(description="Consul response index for blocking queries")


__all__: list[str] = ["ModelConsulKVGetFoundPayload"]
