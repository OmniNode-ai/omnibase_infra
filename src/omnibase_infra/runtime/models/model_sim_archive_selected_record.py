# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Redacted broker-record claim used only by the OMN-19726 sim selector."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

_SHA256 = r"^[0-9a-f]{64}$"
CanonicalSha256 = Annotated[str, Field(pattern=_SHA256, strict=True)]


class ModelSimArchiveSelectedRecord(BaseModel):
    """A redacted claimed source record, not an authenticated raw record.

    ``broker_headers_sha256`` is over the exact ordered raw Kafka header
    sequence, never over normalized ledger headers.  ``broker_key_sha256`` is
    explicitly null only when the original Kafka key was null.  A trusted
    receipt/raw-record comparator must verify every broker digest and timestamp
    before a record can be given enqueue authority.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_topic: str = Field(min_length=1, max_length=249, strict=True)
    source_partition: int = Field(ge=0, strict=True)
    source_offset: int = Field(ge=0, strict=True)
    envelope_sha256: CanonicalSha256
    parent_envelope_sha256: CanonicalSha256 | None = None
    broker_key_sha256: CanonicalSha256 | None
    broker_value_sha256: CanonicalSha256
    broker_headers_sha256: CanonicalSha256
    broker_timestamp_ms: int = Field(ge=0, strict=True)
    ledger_value_sha256: CanonicalSha256
    ledger_headers_normalized: Literal[True]


__all__ = ["CanonicalSha256", "ModelSimArchiveSelectedRecord"]
