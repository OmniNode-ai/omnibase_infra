# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Independently verified proof that a real application-path write occurred."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.cutover.models.model_connection_identity import (
    ModelConnectionIdentity,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelApplicationPathWriteProof(BaseModel):
    """Bind an application-path write claim to a live, server-verified readback.

    ``database_ref`` and ``principal`` are populated from ``current_database()``
    and ``current_user`` read back on the same connection the caller used to
    perform the write -- they are never accepted as free caller-supplied
    strings.  ``write_result_hash`` binds the canonicalized rows returned by
    the caller-declared, read-only ``verification_sql`` (hashed together with
    the query text itself via ``verification_query_hash`` so the query cannot
    be swapped after the fact).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    database_ref: str = Field(..., min_length=1, max_length=200)
    principal: str = Field(..., min_length=1, max_length=160)
    schema_ref: str = Field(..., min_length=1, max_length=160)
    target_sequence: int = Field(..., ge=1)
    verification_query_hash: str = Field(..., pattern=_SHA256_PATTERN)
    write_result_hash: str = Field(..., pattern=_SHA256_PATTERN)
    connection_identity: ModelConnectionIdentity


__all__ = ["ModelApplicationPathWriteProof"]
