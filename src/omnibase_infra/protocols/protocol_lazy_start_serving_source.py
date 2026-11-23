# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the lazy-start store reads from the front door: its serving status (OMN-26316)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.models.lazy_start.model_lazy_start_serving_status import (
    ModelLazyStartServingStatus,
)

__all__: list[str] = ["ProtocolLazyStartServingSource"]


@runtime_checkable
class ProtocolLazyStartServingSource(Protocol):
    """The front door's serving set and failed wakes, read as one snapshot."""

    def serving_status(self) -> ModelLazyStartServingStatus:
        """The contracts woken so far and, per failed wake, the reason."""
        ...
