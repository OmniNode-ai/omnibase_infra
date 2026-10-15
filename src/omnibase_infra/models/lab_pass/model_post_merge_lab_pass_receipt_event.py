# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The bus event a post-merge lab-pass receipt publishes.

Ticket: OMN-20772
"""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.lab_pass.model_post_merge_lab_pass_check_outcome import (
    ModelPostMergeLabPassCheckOutcome,
)
from omnibase_infra.models.lab_pass.model_post_merge_lab_pass_generation import (
    ModelPostMergeLabPassGeneration,
)
from omnibase_infra.models.lab_pass.model_post_merge_lab_pass_identity import (
    ModelPostMergeLabPassIdentity,
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: The failing dimension a receipt with unreadable or invalid provenance adds
#: beside its failing checks.
BINDING_DIMENSION = "binding"


class ModelPostMergeLabPassReceiptEvent(BaseModel):
    """One event per post-merge lab-pass receipt, PASS or FAIL.

    ``event_id`` is derived from the receipt, so a republish of the same
    receipt carries the same id and a consumer can drop the duplicate.
    ``verdict`` is PASS only when the receipt passed AND the provenance binds
    it to the lane: a receipt whose provenance is unreadable publishes FAIL with
    ``binding_valid=False``, whatever its checks said.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    schema_version: str = Field(min_length=1)
    event_type: Literal["post-merge-lab-pass-receipt"]
    topic: str = Field(min_length=1)
    event_id: UUID
    receipt_key: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    sha: str
    generation: ModelPostMergeLabPassGeneration
    identity: ModelPostMergeLabPassIdentity
    verdict: Literal["PASS", "FAIL"]
    receipt_result: Literal["PASS", "FAIL"]
    binding_valid: bool
    binding_reason: str
    checks: tuple[ModelPostMergeLabPassCheckOutcome, ...] = Field(min_length=1)
    failing_dimensions: tuple[str, ...]
    started_at: str = Field(min_length=1)
    finished_at: str = Field(min_length=1)
    agent_command_id: UUID | None

    @model_validator(mode="after")
    def _verdict_is_consistent(self) -> ModelPostMergeLabPassReceiptEvent:
        if not _SHA_RE.fullmatch(self.sha):
            msg = f"sha={self.sha!r} is not a full 40-character lowercase sha"
            raise ValueError(msg)
        failing_checks = {c.name for c in self.checks if c.outcome != "pass"}
        expected = set(failing_checks)
        if not self.binding_valid:
            expected.add(BINDING_DIMENSION)
        if set(self.failing_dimensions) != expected or len(
            set(self.failing_dimensions)
        ) != len(self.failing_dimensions):
            msg = (
                f"failing_dimensions={list(self.failing_dimensions)} does not "
                f"match the failing checks and binding validity {sorted(expected)}"
            )
            raise ValueError(msg)
        should_pass = (
            self.receipt_result == "PASS" and self.binding_valid and not failing_checks
        )
        if (self.verdict == "PASS") != should_pass:
            msg = (
                f"verdict={self.verdict!r} contradicts receipt_result="
                f"{self.receipt_result!r}, binding_valid={self.binding_valid} "
                f"and failing dimensions {sorted(expected)}"
            )
            raise ValueError(msg)
        if self.binding_valid:
            complete = self.identity.complete and all(
                (
                    self.generation.container,
                    self.generation.container_ref,
                    self.generation.image,
                    self.generation.revision,
                )
            )
            if not complete:
                msg = (
                    "binding_valid=True requires a complete generation and "
                    "identity; an unreadable provenance is binding_valid=False"
                )
                raise ValueError(msg)
        elif not self.binding_reason:
            msg = "binding_valid=False requires a binding_reason"
            raise ValueError(msg)
        return self


__all__ = ["BINDING_DIMENSION", "ModelPostMergeLabPassReceiptEvent"]
