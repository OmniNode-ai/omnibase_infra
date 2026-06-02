# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Parsed DLQ message model for the contract-native DLQ replay node (OMN-12619).

Relocated from ``scripts/dlq_replay.py`` so the node owns the parsing surface.
The script imports this model back from the node (thin-CLI shim).
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelDlqMessage(BaseModel):
    """A DLQ message parsed from a Kafka payload, with replay metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_topic: str = Field(
        ..., description="Topic the message originally failed on."
    )
    original_key: str | None = Field(default=None, description="Original message key.")
    original_value: str = Field(
        default="", description="Original message value (UTF-8)."
    )
    original_offset: str | None = Field(
        default=None, description="Original Kafka offset (string for JSON consistency)."
    )
    original_partition: int | None = Field(
        default=None, description="Original Kafka partition."
    )
    failure_reason: str = Field(
        default="", description="Human-readable failure reason."
    )
    failure_timestamp: str = Field(
        default="", description="ISO-8601 timestamp of the original failure."
    )
    correlation_id: UUID = Field(
        ..., description="Correlation ID of the original message."
    )
    retry_count: int = Field(default=0, ge=0, description="Prior retry attempts.")
    error_type: str = Field(default="Unknown", description="Classified error type.")
    dlq_offset: int = Field(
        ..., ge=0, description="Offset of this message in the DLQ topic."
    )
    dlq_partition: int = Field(
        ..., ge=0, description="Partition of this message in the DLQ topic."
    )
    raw_payload: dict[str, object] = Field(
        default_factory=dict, description="The full decoded DLQ payload."
    )

    @classmethod
    def from_kafka_message(
        cls,
        payload: dict[str, object],
        dlq_offset: int,
        dlq_partition: int,
    ) -> ModelDlqMessage:
        """Parse a DLQ message from a decoded Kafka payload.

        Raises:
            ValueError: If retry_count is present but not a valid integer.
        """
        original_message = payload.get("original_message", {})
        if not isinstance(original_message, dict):
            original_message = {}

        correlation_id_str = payload.get("correlation_id", "")
        try:
            correlation_id = UUID(str(correlation_id_str))
        except (ValueError, AttributeError):
            correlation_id = uuid4()

        retry_count = cls._parse_retry_count(payload.get("retry_count", 0))

        offset_value = original_message.get("offset")
        return cls(
            original_topic=str(payload.get("original_topic", "unknown")),
            original_key=original_message.get("key"),
            original_value=str(original_message.get("value", "")),
            original_offset=str(offset_value) if offset_value is not None else None,
            original_partition=original_message.get("partition"),
            failure_reason=str(payload.get("failure_reason", "")),
            failure_timestamp=str(payload.get("failure_timestamp", "")),
            correlation_id=correlation_id,
            retry_count=retry_count,
            error_type=str(payload.get("error_type", "Unknown")),
            dlq_offset=dlq_offset,
            dlq_partition=dlq_partition,
            raw_payload=payload,
        )

    @staticmethod
    def _parse_retry_count(value: object) -> int:
        """Parse retry_count with explicit validation (no silent coercion)."""
        if isinstance(value, bool):
            raise ValueError(
                f"Invalid retry_count type: expected int or str, got {type(value).__name__}"
            )
        if isinstance(value, int):
            return value
        if value is None:
            return 0
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return 0
            try:
                return int(stripped)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid retry_count value: '{value}' is not a valid integer"
                ) from exc
        raise ValueError(
            f"Invalid retry_count type: expected int or str, got {type(value).__name__}"
        )


__all__ = ["ModelDlqMessage"]
