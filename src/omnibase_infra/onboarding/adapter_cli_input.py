# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI input adapter for interactive onboarding.

Reads from stdin via input(). Validates choices against step options and
retries on invalid input. Supports comma-separated multi-choice input.

Secret-marked steps go through :func:`getpass.getpass` instead (OMN-16038).
``getpass`` reads from the controlling terminal with echo disabled, so the
client secret never lands in the terminal scrollback, in a screen recording, or
in the shell's own history of the session.
"""

from __future__ import annotations

import sys
from getpass import getpass

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

    async def collect_secret(self, step: ModelInteractiveStep) -> str:
        """Read a secret with terminal echo disabled.

        Re-prompts on blank input for a required step rather than accepting an
        empty string: an empty secret would be written to the credentials
        artifact and only fail much later, at the first gateway call.
        """
        while True:
            value = getpass(f"{step.prompt} (input hidden): ").strip()
            if value or not step.required:
                return value
            # Only the fact of the retry is printed — never the input.
            sys.stderr.write("A value is required. Nothing was echoed; try again.\n")

    async def notify_action(self, step: ModelInteractiveStep) -> None:
        sys.stdout.write(f"[action] {step.prompt}\n")


__all__ = ["AdapterCliInput"]
