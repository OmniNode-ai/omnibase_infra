# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed read-only query contract for PostgreSQL transformation evidence."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_READ_PREFIX = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_MUTATING_TOKEN = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|GRANT|REVOKE|TRUNCATE|COPY|CALL|DO)\b",
    re.IGNORECASE,
)


class ModelPostgresEvidenceQuerySet(BaseModel):
    """One-column, read-only queries producing canonical evidence signatures.

    Source queries may explicitly transform legacy values into target semantics
    (for example by joining a checked-in legacy-value-to-UUID mapping).  Every
    dimension is required; no omitted query silently resolves to an empty proof.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    label: str = Field(..., min_length=1, max_length=160)
    keys_sql: str = Field(..., min_length=6)
    rows_sql: str = Field(..., min_length=6)
    foreign_keys_sql: str = Field(..., min_length=6)
    sequences_sql: str = Field(..., min_length=6)
    owners_sql: str = Field(..., min_length=6)
    grants_sql: str = Field(..., min_length=6)
    policies_sql: str = Field(..., min_length=6)
    views_functions_sql: str = Field(..., min_length=6)
    dependencies_sql: str = Field(..., min_length=6)
    collisions_sql: str = Field(..., min_length=6)

    @field_validator(
        "keys_sql",
        "rows_sql",
        "foreign_keys_sql",
        "sequences_sql",
        "owners_sql",
        "grants_sql",
        "policies_sql",
        "views_functions_sql",
        "dependencies_sql",
        "collisions_sql",
    )
    @classmethod
    def _read_only_single_statement(cls, value: str) -> str:
        if not _READ_PREFIX.match(value):
            raise ValueError("evidence query must start with SELECT or WITH")
        if ";" in value:
            raise ValueError("evidence query must be exactly one statement")
        if _MUTATING_TOKEN.search(value):
            raise ValueError("evidence query must be read-only")
        return value


__all__ = ["ModelPostgresEvidenceQuerySet"]
