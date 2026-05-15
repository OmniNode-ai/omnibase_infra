# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Protocol for interactive onboarding input collection.

Implementations may collect input via CLI, Claude Code AskUserQuestion,
or a fake adapter for testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.onboarding.models_interactive import ModelInteractiveStep


@runtime_checkable
class ProtocolInputAdapter(Protocol):
    """Adapter protocol for collecting user input during interactive onboarding.

    Implementations:
    - AdapterCliInput: collects via stdin (CLI)
    - AdapterFakeInput: returns predetermined responses (tests)

    Action steps use notify_action which is notification-only — the adapter
    never executes side effects. Write actions are the outer handler's concern.
    """

    async def collect_choice(self, step: ModelInteractiveStep) -> str:
        """Collect a single choice from the given options."""
        ...

    async def collect_multi_choice(self, step: ModelInteractiveStep) -> list[str]:
        """Collect one or more choices from the given options."""
        ...

    async def collect_text(self, step: ModelInteractiveStep) -> str:
        """Collect free-form text input."""
        ...

    async def notify_action(self, step: ModelInteractiveStep) -> None:
        """Notify the user that an action step is being executed.

        Notification only — never executes the action itself.
        """
        ...


__all__ = ["ProtocolInputAdapter"]
