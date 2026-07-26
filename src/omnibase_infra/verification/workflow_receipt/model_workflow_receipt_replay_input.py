# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``ModelWorkflowReceiptReplayInput`` (OMN-15095)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.verification.workflow_receipt.enum_workflow_receipt_terminal_status import (
    EnumWorkflowReceiptTerminalStatus,
)


class ModelWorkflowReceiptReplayInput(BaseModel):
    """The minimal subset of ``omninode_infra``'s ``ModelWorkflowReceipt``
    (OMN-15094, ``docker/onex-api/models/model_workflow_receipt.py``) this
    verifier needs to replay-then-diff. Constructed from a receipt JSON file
    rendered by ``workflow_receipt_renderer.render_workflow_receipt`` --
    ``omninode_infra`` produces the artifact, this repo (which already owns
    ``node_kafka_replay_compute``) verifies it. The two services are joined
    by the durable ``workflow_receipt.json`` artifact, not a shared Python
    import -- ``omnibase_infra`` cannot be added as a pip dependency of
    ``docker/onex-api`` without breaking its pinned fastapi/redis/uvicorn
    versions (verified 2026-07-25: omnibase-infra 0.36.1 requires
    ``fastapi<0.137``/``redis<8.0``/``uvicorn<0.50``, all violated by
    onex-api's live pins), so this is the correct seam, not a workaround.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: UUID
    correlation_id: UUID
    status: EnumWorkflowReceiptTerminalStatus
    terminal_model_used: str
    terminal_total_tokens: int
    terminal_latency_ms: int
    projection_row_hash: str
    terminal_event_hash: str


__all__ = ["ModelWorkflowReceiptReplayInput"]
