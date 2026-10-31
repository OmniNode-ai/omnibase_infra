# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""What one consumer-group offset handover did (OMN-21753).

Returned by ``HandlerConsumerGroupOffsetHandover.handle``
so the caller can log, and a test can assert, exactly which partitions were
seeded into the new group and which it already owned.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelOffsetHandoverResult(BaseModel):
    """One topic's handover from a retired group to its successor group.

    Attributes:
        topic: Canonical topic the handover covered.
        from_group: The retired group read from.
        to_group: The effective group written to (after any per-topic,
            instance or namespace scoping the transport applies).
        committed: Partition to the offset committed into ``to_group``.
        kept: Partitions ``to_group`` had already committed, left untouched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., min_length=1)
    from_group: str = Field(..., min_length=1)
    to_group: str = Field(..., min_length=1)
    committed: dict[int, int] = Field(default_factory=dict)
    kept: tuple[int, ...] = Field(default=())
