# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Outcome of one application-database ACL apply attempt."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.validation.models.model_acl_apply_consent import (
    ModelAclApplyConsent,
)


class ModelAclApplyReport(BaseModel):
    """Apply outcome, safe to print in full: it carries no credential value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mutated: bool
    probes_run: int
    probes_passed: int
    snapshot_path: str
    consent: ModelAclApplyConsent
    probe_descriptions: tuple[str, ...] = ()


__all__ = ["ModelAclApplyReport"]
