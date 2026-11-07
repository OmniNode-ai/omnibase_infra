# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Port to the per-node serving count read (OMN-22132)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omnibase_infra.handlers.placement.model_placement_serving_report import (
        ModelPlacementServingReport,
    )


class ProtocolPlacementServingReader(Protocol):
    """Reads how many live runtimes serve each expected node."""

    async def __call__(
        self, *, expected_nodes: Sequence[str], min_serving: int
    ) -> ModelPlacementServingReport:
        """Count, per expected node, the live runtimes whose membership lists it."""
        ...


__all__ = ["ProtocolPlacementServingReader"]
