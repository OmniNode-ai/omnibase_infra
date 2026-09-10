# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-run commit ledger for the DLQ replay drain (OMN-17896, OMN-18119).

One drain builds one of these. It answers a single question per record: may this
offset be committed? The two halves are inseparable, which is why they are one
model rather than two arguments passed side by side -- ``completed`` without
``blocked`` would commit past a record that is durable nowhere, and that is the
silent drop OMN-17896 exists to prevent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDlqCommitLedger(BaseModel):
    """The offsets one drain may commit, and the partitions it may not.

    Keys are ``(dlq_topic, partition)``. OMN-18119 made the topic half load
    bearing: a run visits one consumer per declared subscribe topic, so a
    ledger spans more than one topic and an offset must be committed against
    the topic its record was actually read from.

    Attributes:
        completed: Per partition, the NEXT offset to read -- one past the last
            record whose handling reached a durable terminal outcome.
        blocked: Partitions stopped by a record that is durable nowhere. A
            partition is blocked by the FIRST such record and stays blocked for
            the rest of the run, so no later success can commit over it.
    """

    model_config = ConfigDict(extra="forbid")

    completed: dict[tuple[str, int], int] = Field(
        default_factory=dict,
        description="(topic, partition) -> next offset to read.",
    )
    blocked: set[tuple[str, int]] = Field(
        default_factory=set,
        description="(topic, partition) pairs frozen by a non-durable record.",
    )

    def mark_completed(self, key: tuple[str, int], next_offset: int) -> None:
        """Advance a partition's committable offset, unless it is blocked."""
        if key in self.blocked:
            return
        self.completed[key] = next_offset

    def block(self, key: tuple[str, int]) -> None:
        """Freeze a partition for the rest of the run."""
        self.blocked.add(key)

    @property
    def has_committable_offsets(self) -> bool:
        return bool(self.completed)


__all__ = ["ModelDlqCommitLedger"]
