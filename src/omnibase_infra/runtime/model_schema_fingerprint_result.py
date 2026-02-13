# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Schema fingerprint result model.

Holds the computed fingerprint, counts, and per-table hashes returned by
``compute_schema_fingerprint()``.

Related:
    - OMN-2087: Handshake hardening -- Schema fingerprint manifest + startup assertion
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSchemaFingerprintResult(BaseModel):
    """Immutable result of a schema fingerprint computation.

    Contains the overall fingerprint hash plus per-table breakdown for
    diff computation when a mismatch is detected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: str = Field(..., description="SHA256 hex digest of canonical schema")
    table_count: int = Field(..., description="Number of tables fingerprinted")
    column_count: int = Field(..., description="Total columns across all tables")
    constraint_count: int = Field(
        ..., description="Total constraints across all tables"
    )
    per_table_hashes: tuple[tuple[str, str], ...] = Field(
        ..., description="(table_name, hash) pairs for diff computation"
    )


__all__ = ["ModelSchemaFingerprintResult"]
