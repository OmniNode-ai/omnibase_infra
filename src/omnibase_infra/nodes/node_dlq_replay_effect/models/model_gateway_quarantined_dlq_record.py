# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recognize the gateway's own durable quarantine record (OMN-20318).

The gateway forensic record is already the gateway's own quarantine of a
delivery failure, with no original message to replay. Re-quarantining it is
the same self-amplification OMN-20318 / PR 4435 removed for run summaries.
Carry its DLQ coordinate to the handler so the offset can advance while the
record stays on the DLQ topic where the gateway wrote it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayQuarantinedDlqRecord(BaseModel):
    """A gateway forensic record already durably quarantined on its DLQ."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dlq_topic: str = Field(..., min_length=1, description="DLQ topic read from.")
    dlq_partition: int = Field(..., ge=0, description="Partition of the record.")
    dlq_offset: int = Field(..., ge=0, description="Offset of the record.")
    failure_class: str = Field(..., description="Gateway quarantine classification.")
    original_topic: str = Field(
        ..., min_length=1, description="Topic the gateway could not deliver."
    )

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        dlq_topic: str,
        dlq_partition: int,
        dlq_offset: int,
    ) -> ModelGatewayQuarantinedDlqRecord | None:
        """Recognize only the gateway forensic shape, preserving other paths."""
        if not isinstance(payload, dict) or "original_message" in payload:
            return None
        direction = payload.get("direction")
        if not isinstance(direction, str) or direction not in {"outbound", "inbound"}:
            return None
        original_topic = payload.get("original_topic")
        if not isinstance(original_topic, str) or not original_topic:
            return None
        failure_class = payload.get("failure_class")
        if (
            not isinstance(failure_class, str)
            or not failure_class.startswith("gateway_")
            or not failure_class.endswith("_record")
        ):
            return None
        return cls(
            dlq_topic=dlq_topic,
            dlq_partition=dlq_partition,
            dlq_offset=dlq_offset,
            failure_class=failure_class,
            original_topic=original_topic,
        )


__all__ = ["ModelGatewayQuarantinedDlqRecord"]
