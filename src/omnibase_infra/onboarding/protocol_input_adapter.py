# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Protocol for interactive onboarding input collection.

Implementations may collect input via CLI, Claude Code AskUserQuestion,
or a fake adapter for testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep


@runtime_checkable
class ProtocolInputAdapter(Protocol):
    async def collect_choice(self, step: ModelInteractiveStep) -> str: ...

    async def collect_multi_choice(self, step: ModelInteractiveStep) -> list[str]: ...

    async def collect_text(self, step: ModelInteractiveStep) -> str: ...

    async def notify_action(self, step: ModelInteractiveStep) -> None: ...


__all__ = ["ProtocolInputAdapter"]
