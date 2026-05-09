# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fake input adapter for testing.

Takes a dict[step_id, response] at construction and returns predetermined
responses without any I/O. Used by executor and handler unit tests.
"""

from __future__ import annotations

from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep


class AdapterFakeInput:
    def __init__(self, responses: dict[str, str | list[str]]) -> None:
        self._responses = responses

    async def collect_choice(self, step: ModelInteractiveStep) -> str:
        value = self._responses[step.id]
        if not isinstance(value, str):
            raise TypeError(
                f"Expected str response for step {step.id!r}, got {type(value)}"
            )
        return value

    async def collect_multi_choice(self, step: ModelInteractiveStep) -> list[str]:
        value = self._responses[step.id]
        if isinstance(value, list):
            return value
        return [v.strip() for v in str(value).split(",") if v.strip()]

    async def collect_text(self, step: ModelInteractiveStep) -> str:
        value = self._responses[step.id]
        if not isinstance(value, str):
            raise TypeError(
                f"Expected str response for step {step.id!r}, got {type(value)}"
            )
        return value

    async def notify_action(self, step: ModelInteractiveStep) -> None:  # stub-ok
        pass


__all__ = ["AdapterFakeInput"]
