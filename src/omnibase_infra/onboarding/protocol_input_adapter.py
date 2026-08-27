# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Protocol for interactive onboarding input collection.

Implementations may collect input via CLI, Claude Code AskUserQuestion,
or a fake adapter for testing.

``collect_secret`` is a separate method rather than a flag on ``collect_text``
(OMN-16038): the two have different terminal behavior — one echoes, one must
not — and a boolean argument makes the non-echoing path something a caller can
forget to ask for. A distinct method means an adapter that has not implemented
masked collection fails to satisfy the protocol instead of silently echoing a
client secret.
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

    async def collect_secret(self, step: ModelInteractiveStep) -> str: ...

    async def notify_action(self, step: ModelInteractiveStep) -> None: ...


__all__ = ["ProtocolInputAdapter"]
