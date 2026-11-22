# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_post_failure_verdict import EnumPostFailureVerdict


class ModelPostFailureDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumPostFailureVerdict
    detail: str
    #: The heads that differ from the failed job's source, on a CONVERGE.
    heads: dict[str, str] = {}
    #: What a CONVERGE marks attempted: ``<repository>@<head>`` per head in ``heads``.
    attempt_keys: tuple[str, ...] = ()
