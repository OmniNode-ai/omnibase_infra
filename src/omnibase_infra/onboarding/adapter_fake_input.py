# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fake input adapter — returns predetermined responses for tests."""

from __future__ import annotations

from collections.abc import Mapping

from omnibase_infra.onboarding.models_interactive import ModelInteractiveStep


class AdapterFakeInput:
    """Drives interactive onboarding with predetermined responses.

    responses: maps step.id → str (for choice/text) or list[str] (for multi_choice).
    action steps via notify_action are recorded in notified_steps without side effects.
    """

    def __init__(self, responses: Mapping[str, object]) -> None:
        self._responses = responses
        self.notified_steps: list[str] = []

    async def collect_choice(self, step: ModelInteractiveStep) -> str:
        response = self._responses[step.id]
        if not isinstance(response, str):
            raise TypeError(
                f"Step '{step.id}': expected str response, got {type(response)}"
            )
        return response

    async def collect_multi_choice(self, step: ModelInteractiveStep) -> list[str]:
        response = self._responses[step.id]
        if isinstance(response, str):
            return [t.strip() for t in response.split(",") if t.strip()]
        if not isinstance(response, list):
            raise TypeError(
                f"Step '{step.id}': expected list response, got {type(response)}"
            )
        non_str = [item for item in response if not isinstance(item, str)]
        if non_str:
            raise TypeError(
                f"Step '{step.id}': list response contains non-str elements: "
                f"{[type(x).__name__ for x in non_str]}"
            )
        return [t.strip() for t in response if t.strip()]

    async def collect_text(self, step: ModelInteractiveStep) -> str:
        response = self._responses.get(step.id, "")
        if not isinstance(response, str):
            raise TypeError(
                f"Step '{step.id}': expected str response, got {type(response)}"
            )
        return response.strip()

    async def notify_action(self, step: ModelInteractiveStep) -> None:
        self.notified_steps.append(step.id)


__all__ = ["AdapterFakeInput"]
