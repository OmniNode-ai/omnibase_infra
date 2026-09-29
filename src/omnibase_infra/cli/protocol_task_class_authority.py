# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The members of the task-class contract ``onex delegate`` reads (OMN-19407).

An interface, not a vocabulary: it holds no class name, phrase, fallback or
number. The object behind it is whatever the ``onex.contracts`` registry entry
``task_class_authority`` loads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from omnibase_infra.cli.protocol_execution_budget import ProtocolExecutionBudget
from omnibase_infra.cli.protocol_selection_fallback import ProtocolSelectionFallback
from omnibase_infra.cli.protocol_task_type_resolution import (
    ProtocolTaskTypeResolution,
)

__all__ = ["ProtocolTaskClassAuthority"]


@runtime_checkable
class ProtocolTaskClassAuthority(Protocol):
    """The members of the task-class contract ``onex delegate`` reads."""

    @property
    def public_task_classes(self) -> frozenset[str]: ...

    @property
    def internal_task_classes(self) -> frozenset[str]: ...

    @property
    def unroutable_task_classes(self) -> Mapping[str, object]: ...

    @property
    def selection_fallback(self) -> ProtocolSelectionFallback | None: ...

    def resolve_task_type(
        self, prompt: str, *, explicit: str | None
    ) -> ProtocolTaskTypeResolution: ...

    def execution_budget(self, task_class: str) -> ProtocolExecutionBudget: ...
