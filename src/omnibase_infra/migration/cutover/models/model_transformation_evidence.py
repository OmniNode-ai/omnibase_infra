# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical source or target evidence for transformation receipts."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.migration.cutover.models.model_connection_identity import (
    ModelConnectionIdentity,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelTransformationEvidence(BaseModel):
    """Canonicalized evidence for one side of a family transformation.

    ``keys`` and ``transformed_row_hashes`` are already projected into target
    semantics. A legacy slug-to-UUID mapping or internal-column omission is
    therefore explicit in the hash-bound query contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    label: str = Field(..., min_length=1)
    evidence_contract_hash: str = Field(..., pattern=_SHA256_PATTERN)
    binding_ref: str = Field(..., min_length=1, max_length=200)
    connection_identity: ModelConnectionIdentity
    keys: tuple[str, ...]
    row_count: int = Field(..., ge=0)
    transformed_row_hashes: tuple[str, ...]
    foreign_keys: tuple[str, ...]
    sequences: tuple[str, ...]
    owners: tuple[str, ...]
    grants: tuple[str, ...]
    policies: tuple[str, ...]
    views_functions: tuple[str, ...]
    dependencies: tuple[str, ...]
    collision_keys: tuple[str, ...]

    @field_validator(
        "keys",
        "transformed_row_hashes",
        "foreign_keys",
        "sequences",
        "owners",
        "grants",
        "policies",
        "views_functions",
        "dependencies",
        "collision_keys",
    )
    @classmethod
    def _canonical_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)):
            raise ValueError("evidence tuples must be sorted deterministically")
        if len(value) != len(set(value)):
            raise ValueError("evidence tuples must not contain duplicates")
        return value

    @field_validator("transformed_row_hashes")
    @classmethod
    def _row_hashes_are_sha256(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(_SHA256_PATTERN, item) is None for item in value):
            raise ValueError("transformed row hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _counts_are_coherent(self) -> ModelTransformationEvidence:
        if self.row_count != len(self.keys):
            raise ValueError("row_count must equal the canonical key-set size")
        if self.row_count != len(self.transformed_row_hashes):
            raise ValueError("row_count must equal transformed row-hash count")
        return self


__all__ = ["ModelTransformationEvidence"]
