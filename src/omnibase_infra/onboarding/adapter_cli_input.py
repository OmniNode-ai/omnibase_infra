# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CLI input adapter — collects user input via stdin."""

from __future__ import annotations

import asyncio

import click

from omnibase_infra.onboarding.models_interactive import ModelInteractiveStep


class AdapterCliInput:
    """Collects interactive onboarding input via stdin.

    Validates choice inputs against declared options (rejects unknown values).
    Multi-choice accepts comma-separated input; trims whitespace per token.
    Required steps re-prompt on empty/invalid input; optional steps return "" on empty.
    """

    async def collect_choice(self, step: ModelInteractiveStep) -> str:
        options = step.options
        if not options:
            if step.required:
                raise ValueError(f"Step '{step.id}' is required but has no options")
            return ""
        prompt_text = self._format_choice_prompt(step)
        while True:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, input, prompt_text
            )
            value = raw.strip()
            if not value:
                if not step.required:
                    return ""
                click.echo(f"  Choice required. Valid options: {', '.join(options)}")
                continue
            if value not in options:
                click.echo(
                    f"  Invalid choice '{value}'. Valid options: {', '.join(options)}"
                )
                continue
            return value

    async def collect_multi_choice(self, step: ModelInteractiveStep) -> list[str]:
        options = step.options
        if not options:
            if step.required:
                raise ValueError(f"Step '{step.id}' is required but has no options")
            return []
        prompt_text = self._format_multi_choice_prompt(step)
        while True:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, input, prompt_text
            )
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            if not tokens:
                if not step.required:
                    return []
                click.echo(
                    f"  At least one selection required. Valid options: {', '.join(options)}"
                )
                continue
            invalid = [t for t in tokens if t not in options]
            if invalid:
                click.echo(
                    f"  Unknown options: {', '.join(invalid)}. Valid options: {', '.join(options)}"
                )
                continue
            return tokens

    async def collect_text(self, step: ModelInteractiveStep) -> str:
        prompt_text = f"{step.prompt}: "
        while True:
            raw = await asyncio.get_event_loop().run_in_executor(
                None, input, prompt_text
            )
            value = raw.strip()
            if not value and step.required:
                click.echo("  This field is required.")
                continue
            return value

    async def notify_action(self, step: ModelInteractiveStep) -> None:
        click.echo(f"  > {step.prompt}")

    def _format_choice_prompt(self, step: ModelInteractiveStep) -> str:
        options_str = "/".join(step.options)
        return f"{step.prompt} [{options_str}]: "

    def _format_multi_choice_prompt(self, step: ModelInteractiveStep) -> str:
        options_str = ", ".join(step.options)
        return f"{step.prompt} (comma-separated) [{options_str}]: "


__all__ = ["AdapterCliInput"]
