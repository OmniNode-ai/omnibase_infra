# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Result model for one headless secret-seeding run (OMN-16897).

This model is the run receipt. It is printed to stdout by ``onex skill``,
may be published as a terminal event, and is the thing an operator pastes
into a ticket — so it is the single most dangerous place a secret value
could leak to.

Every field here is a NAME, a COUNT, an ADDRESS, or a REDACTED message.
There is no value field, and ``extra="forbid"`` keeps one from being added
by accident at a call site. The contract-conformance test asserts this
model's field set contains no value-carrying name, so the property is
mechanically enforced rather than merely intended (CLAUDE.md Rule 5).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_secret_seed_effect.models.enum_secret_seed_verdict import (
    EnumSecretSeedVerdict,
)


class ModelSecretSeedResult(BaseModel):
    """Receipt for one seed run. Names, counts, addresses — never values."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Seed run correlation ID.")
    verdict: EnumSecretSeedVerdict = Field(
        ..., description="Terminal verdict for this run."
    )
    success: bool = Field(
        ...,
        description=(
            "True only for SEEDED and DRY_RUN. A run that seeded nothing is "
            "not a success."
        ),
    )
    detail: str = Field(
        default="",
        description=(
            "Human-readable explanation. Passed through error sanitisation "
            "AND explicit value redaction before it is ever set."
        ),
    )

    dry_run: bool = Field(
        ..., description="Whether this run was a plan-only run (zero writes)."
    )
    # Target addressing — echoed back so a receipt states which of the three
    # live instances it is about. None of these is a secret.
    infisical_host: str = Field(..., description="Target Infisical instance URL.")
    project_id: UUID = Field(..., description="Target project UUID.")
    environment_slug: str = Field(..., description="Target environment slug.")
    secret_path: str = Field(..., description="Target secret folder.")
    source_path: str = Field(
        ..., description="Source the values were read from (a path, not content)."
    )

    created_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names written that were absent from the store beforehand. In a "
            "dry run these are the names that WOULD be created."
        ),
    )
    updated_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names written that already existed (upsert). In a dry run these "
            "are the names that WOULD be updated."
        ),
    )
    failed_names: list[str] = Field(
        default_factory=list,
        description="Names whose write was rejected or errored.",
    )
    verified_names: list[str] = Field(
        default_factory=list,
        description="Names confirmed present by post-write NAME readback.",
    )
    unverified_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names whose write was accepted but which did not appear in the "
            "post-write name listing. Any entry here fails the run."
        ),
    )
    missing_from_source_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names requested via the keys allowlist that the source does not "
            "contain. Reported rather than skipped."
        ),
    )
    errors: list[str] = Field(
        default_factory=list,
        description=(
            "Per-name failure explanations, sanitised and value-redacted. "
            "Ordered to match failed_names."
        ),
    )

    @property
    def written_count(self) -> int:
        """Number of names actually written (zero on a dry run)."""
        if self.dry_run:
            return 0
        return len(self.created_names) + len(self.updated_names)


__all__ = ["ModelSecretSeedResult"]
