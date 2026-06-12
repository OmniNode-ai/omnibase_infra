# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed receipt ``result`` payload for runs without a handler result (OMN-13094).

When ``onex node``/``onex run`` executes in ``--output receipt`` mode, the
receipt's ``result`` field carries the node's full handler result whenever one
exists. Runs that produce no handler result (event-only orchestrations) or
that fail carry this summary model instead, so the receipt's ``result`` is
always a typed model and errors are never hidden: on failure the FULL capture
log and error text travel inline in the receipt (parent invariant — see
``docs/plans/2026-06-12-skill-output-suppression-plan.md``, Phase 2 item 1).

.. versionadded:: OMN-13094
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

__all__ = ["ModelReceiptRuntimeSummary"]


class ModelReceiptRuntimeSummary(BaseModel):
    """Receipt ``result`` payload when no handler result exists or the run failed.

    Carries the full ``workflow_result.json`` content plus — on non-success —
    the FULL capture log and error text inline. Errors are never hidden
    behind an artifact ref.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    workflow_result: str = Field(
        ...,
        description=(
            "Terminal workflow result value ('completed', 'partial', "
            "'failed', 'timeout') or 'error' when the runtime raised before "
            "producing a result."
        ),
    )
    exit_code: int = Field(
        ...,
        description="CLI exit code corresponding to the workflow result.",
    )
    workflow: str = Field(
        ...,
        description="Path of the contract that was executed.",
    )
    terminal_payload: JsonValue = Field(
        default=None,
        description="Terminal event payload from workflow_result.json, if any.",
    )
    handler_result: JsonValue = Field(
        default=None,
        description="Handler result from workflow_result.json, if any.",
    )
    error: str = Field(
        default="",
        description=(
            "Full error text (including traceback) when the runtime raised. "
            "Empty when the run produced a terminal workflow result."
        ),
    )
    capture_log: str = Field(
        default="",
        description=(
            "FULL runtime capture log text, inlined on non-success outcomes "
            "so errors are never hidden behind an artifact ref. Empty on "
            "success (the log remains retrievable via the receipt's "
            "artifact_refs)."
        ),
    )
