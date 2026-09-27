# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed raw terminal evidence observed at the dispatch boundary."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ModelDelegationTerminalEvidence:
    """Exact terminal-envelope evidence for one tenant-scoped dispatch.

    ``raw_envelope`` is the original UTF-8 JSON event representation read from
    the broker. A typed view derived later must retain this value, its encoding,
    topic, and cursor as the provenance source.
    """

    correlation_id: UUID
    tenant_id: str
    topic: str
    raw_envelope: bytes
    encoding: str
    partition: int | None
    offset: str | None
