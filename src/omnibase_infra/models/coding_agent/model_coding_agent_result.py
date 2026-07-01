# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed result of a coding-agent invocation (OMN-13247, plan §5.4).

Provenance is explicit: acceptance trusts ONLY the system-derived fields. The
handler never trusts "the agent said it edited files" — ``files_changed`` and
``diff`` come from git, captured AFTER the subprocess exits. ``output`` and
``usage`` are agent-reported and advisory only.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.coding_agent.enum_agent_status import EnumAgentStatus
from omnibase_infra.models.coding_agent.enum_cli_backend_status import (
    EnumCliBackendStatus,
)


class ModelCodingAgentResult(BaseModel):
    """System-derived authoritative result, plus agent-reported advisory fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow run correlation id.")

    # --- system-derived (authoritative; acceptance evidence) -----------------
    status: EnumAgentStatus = Field(
        ..., description="Terminal status (COMPLETED | FAILED | REJECTED)."
    )
    exit_code: int | None = Field(
        default=None,
        description="Subprocess exit code; None if no subprocess ran (REJECTED).",
    )
    files_changed: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Paths from `git diff --name-only`, captured after exit.",
    )
    diff: str = Field(
        default="",
        description="`git diff` output captured after the subprocess exits.",
    )
    diff_hash: str = Field(
        default="",
        description="Stable hash of `diff` for projection / dedupe; empty if no diff.",
    )
    starting_head_sha: str | None = Field(
        default=None,
        description="git HEAD sha recorded before invocation; None if not a repo.",
    )
    error_class: EnumCliBackendStatus = Field(
        default=EnumCliBackendStatus.SUCCESS,
        description="Structured failure class (UNAVAILABLE/TIMEOUT/...).",
    )
    timed_out: bool = Field(
        default=False, description="True if the subprocess was killed on timeout."
    )
    duration_ms: float = Field(
        default=0.0, description="Wall-clock subprocess duration in ms."
    )

    # --- agent-reported (advisory only; NEVER acceptance evidence) -----------
    output: str = Field(
        default="",
        description="Agent stdout. Advisory only — not acceptance evidence.",
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Agent-reported token usage if parseable. Advisory only.",
    )
