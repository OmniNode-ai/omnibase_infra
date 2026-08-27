# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ``ClusterMetadata`` surface the DLQ offset reader uses (OMN-16769).

Declared structurally instead of typing the snapshot as ``Any`` so the
metadata surface stays checkable and the repo's Any-type gate has nothing to
flag. aiokafka ships no stubs for this class.
"""

from __future__ import annotations

from typing import Protocol


class ProtocolClusterMetadata(Protocol):
    """The two ``ClusterMetadata`` methods the DLQ offset reader calls."""

    def topics(self) -> set[str]:
        """Every topic name in this metadata snapshot."""
        ...

    def partitions_for_topic(self, topic: str) -> set[int] | None:
        """Partition ids for one topic, or None when unresolvable."""
        ...


__all__ = ["ProtocolClusterMetadata"]
