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
from pathlib import Path

import click

_MACHINES_YAML_DEFAULT = Path(__file__).parents[4] / "config" / "machines.yaml"

# Fleet-wide constants — same across all machines
_INFRA_HOST = "192.168.86.201"
_KAFKA_BOOTSTRAP = f"{_INFRA_HOST}:19092"
_POSTGRES_HOST = _INFRA_HOST
_POSTGRES_PORT = "5436"
_VALKEY_HOST = _INFRA_HOST
_VALKEY_PORT = "6379"
_MODEL = "opus[1m]"
_OMNICLAUDE_MODE = "agent"


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
    Fleet-wide constants (infra host, Kafka, Postgres) are the same for every machine.
    """
    from omnibase_infra.models.environment.model_machine_registry import (
        ModelMachineEntry,
    )

    assert isinstance(machine, ModelMachineEntry)

    env: dict[str, str] = {
        "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
        "OMNI_HOME": machine.omni_home,
        "ONEX_STATE_DIR": machine.onex_state_dir,
        "OMNICLAUDE_MODE": _OMNICLAUDE_MODE,
        "OMNI_INFRA_HOST": _INFRA_HOST,
        "MAX_THINKING_TOKENS": "10000",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "POSTGRES_HOST": _POSTGRES_HOST,
        "POSTGRES_PORT": _POSTGRES_PORT,
        "VALKEY_HOST": _VALKEY_HOST,
        "VALKEY_PORT": _VALKEY_PORT,
        "KAFKA_BOOTSTRAP_SERVERS": _KAFKA_BOOTSTRAP,
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
