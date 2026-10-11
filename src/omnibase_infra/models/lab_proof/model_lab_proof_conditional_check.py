# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""A check a lab proof profile variant demands only of some diffs.

Ticket: OMN-19566
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck


class ModelLabProofConditionalCheck(BaseModel):
    """A check a variant demands only of a PR that changes the surface it proves.

    Some checks prove one surface: ``forwarder_refused_topic`` reads the gateway
    forwarder's inbound-topic refusals, and ``changed_path_live`` reads the
    runtime logs for the nodes a PR changes. A PR that changes neither (an
    omnimarket pin advance, a core lock bump, a Dockerfile pin) gives the check
    nothing to prove, and demanding it anyway made every such receipt
    MISSING_MANDATORY_CHECK (OMN-19566). The row says, as reviewed data, which
    changed paths make the check mandatory; the pin resolves it from the PR's
    own diff, and an unread or empty diff makes every conditional check
    mandatory (fail closed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: EnumLabProofCheck
    when_changed: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)


__all__ = ["ModelLabProofConditionalCheck"]
