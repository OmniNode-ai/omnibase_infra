# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""onex env CLI subcommand group.

Provides:
  onex env render-settings --machine-id <id>

Outputs canonical settings.json JSON derived from the machine registry.
No paths are hardcoded — all derived from ModelMachineEntry computed properties.

Related:
    - OMN-7528: Task 2 — onex env render-settings command
    - OMN-7526: Environment suite epic
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

_MACHINES_YAML_DEFAULT = Path(__file__).parents[4] / "config" / "machines.yaml"

_MODEL = os.environ.get("OMNI_MODEL", "opus[1m]")
_OMNICLAUDE_MODE = os.environ.get("OMNICLAUDE_MODE", "agent")


def _infra_env() -> dict[str, str]:
    """Infrastructure addresses, read at render time.

    The infrastructure host is deployment-specific, so there is no shipped
    default: an unset ``OMNI_INFRA_HOST`` is refused (OMN-20931).
    """
    host = os.environ.get("OMNI_INFRA_HOST", "").strip()
    if not host:
        raise click.ClickException(
            "OMNI_INFRA_HOST is not set. onex ships no infrastructure host: "
            "set OMNI_INFRA_HOST (for example in ~/.omnibase/.env) to the host "
            "that runs your Postgres, Redpanda and Valkey, then run render-settings again."
        )
    kafka = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", f"{host}:19092")
    return {
        "OMNI_INFRA_HOST": host,
        "POSTGRES_HOST": os.environ.get("POSTGRES_HOST", host),
        "POSTGRES_PORT": os.environ.get("POSTGRES_PORT", "5436"),
        "VALKEY_HOST": os.environ.get("VALKEY_HOST", host),
        "VALKEY_PORT": os.environ.get("VALKEY_PORT", "6379"),
        "KAFKA_BOOTSTRAP_SERVERS": kafka,
    }


@click.group("env")
def env_group() -> None:  # stub-ok: click group
    """Environment configuration commands."""


@env_group.command("render-settings")
@click.option("--machine-id", required=True, help="Machine ID from machines.yaml.")
@click.option(
    "--registry",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to machines.yaml (defaults to config/machines.yaml in repo root).",
)
def render_settings(machine_id: str, registry: Path | None) -> None:
    """Render canonical settings.json for a registered machine."""
    from omnibase_infra.models.environment.model_machine_registry import (
        ModelMachineRegistry,
    )

    registry_path = registry or _MACHINES_YAML_DEFAULT
    reg = ModelMachineRegistry.from_yaml(registry_path)

    try:
        machine = reg.get_machine(machine_id)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    settings = render_settings_json(machine)
    click.echo(json.dumps(settings, indent=2))


def render_settings_json(machine: object) -> dict[str, object]:
    """Build canonical settings.json dict from a ModelMachineEntry.

    Machine-specific values (paths) come from registry entry computed properties.
    The infrastructure host comes from OMNI_INFRA_HOST; the render refuses when
    it is unset.
    """
    from omnibase_infra.models.environment.model_machine_registry import (
        ModelMachineEntry,
    )

    assert isinstance(machine, ModelMachineEntry)

    infra = _infra_env()

    env: dict[str, str] = {
        "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
        "OMNI_HOME": machine.omni_home,
        "ONEX_STATE_DIR": machine.onex_state_dir,
        "OMNICLAUDE_MODE": _OMNICLAUDE_MODE,
        "OMNI_INFRA_HOST": infra["OMNI_INFRA_HOST"],
        "MAX_THINKING_TOKENS": "10000",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "POSTGRES_HOST": infra["POSTGRES_HOST"],
        "POSTGRES_PORT": infra["POSTGRES_PORT"],
        "VALKEY_HOST": infra["VALKEY_HOST"],
        "VALKEY_PORT": infra["VALKEY_PORT"],
        "KAFKA_BOOTSTRAP_SERVERS": infra["KAFKA_BOOTSTRAP_SERVERS"],
    }

    return {
        "$schema": "https://raw.githubusercontent.com/anthropics/claude-code/main/.claude/settings.schema.json",
        "env": env,
        "includeCoAuthoredBy": True,
        "model": _MODEL,
        "statusLine": {
            "command": machine.statusline_path,
        },
        "enabledPlugins": ["omninode-tools"],
        "voiceEnabled": False,
        "skipDangerousModePermissionPrompt": False,
        "autoUpdates": True,
    }
