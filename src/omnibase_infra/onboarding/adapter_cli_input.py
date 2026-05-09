# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI input adapter for interactive onboarding.

Reads from stdin via input(). Validates choices against step options and
retries on invalid input. Supports comma-separated multi-choice input.
"""

from __future__ import annotations

import sys

from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep


class AdapterCliInput:
    async def collect_choice(self, step: ModelInteractiveStep) -> str:
        while True:
            raw = input(f"{step.prompt} [{'/'.join(step.options)}]: ").strip()
            if not step.options or raw in step.options:
                return raw
            sys.stderr.write(f"Invalid choice {raw!r}. Valid options: {step.options}\n")

    async def collect_multi_choice(self, step: ModelInteractiveStep) -> list[str]:
        while True:
            raw = input(
                f"{step.prompt} (comma-separated) [{', '.join(step.options)}]: "
            )
            selected = [v.strip() for v in raw.split(",") if v.strip()]
            invalid = [v for v in selected if step.options and v not in step.options]
            if not invalid:
                return selected
            sys.stderr.write(
                f"Invalid selections {invalid}. Valid options: {step.options}\n"
            )

    async def collect_text(self, step: ModelInteractiveStep) -> str:
        return input(f"{step.prompt}: ").strip()

    async def notify_action(self, step: ModelInteractiveStep) -> None:
        sys.stdout.write(f"[action] {step.prompt}\n")


__all__ = ["AdapterCliInput"]
