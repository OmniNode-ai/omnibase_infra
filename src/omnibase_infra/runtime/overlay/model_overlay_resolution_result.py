# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field


class ModelOverlayResolutionResult(BaseModel):
    """Pure resolution output — no side effects. Boot layer consumes this to mutate env."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolved_pairs: dict[str, str] = Field(
        default_factory=dict,
        description="Keys that should be injected (overlay provides, env does not have).",
    )
    skipped_existing_keys: frozenset[str] = Field(
        default_factory=frozenset,
        description="Keys the overlay provides but env already has — skip to preserve existing.",
    )
    unused_overlay_keys: frozenset[str] = Field(
        default_factory=frozenset,
        description="Keys in overlay that no contract requires — visibility into stale config.",
    )
    required_keys: frozenset[str] = Field(
        default_factory=frozenset,
        description="All keys contracts require (for audit/manifest).",
    )

    @property
    def resolved_pairs_hash(self) -> str:
        """Deterministic SHA-256 of resolved_pairs for replay verification."""
        canonical = json.dumps(
            dict(sorted(self.resolved_pairs.items())), sort_keys=True
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
