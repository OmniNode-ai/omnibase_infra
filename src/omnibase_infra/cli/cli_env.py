# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Environment management CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from omnibase_infra.models.environment.model_machine_registry import (
    ModelMachineEntry,
    ModelMachineRegistry,
)

# ---------------------------------------------------------------------------
# Fleet-wide constants (same for all machines)
# ---------------------------------------------------------------------------
_SCHEMA_URL = "https://json.schemastore.org/claude-code-settings.json"
_OMNI_INFRA_HOST = "192.168.86.201"
_KAFKA_BOOTSTRAP_SERVERS = "192.168.86.201:19092"
_POSTGRES_HOST = "192.168.86.201"
_POSTGRES_PORT = "5436"
_VALKEY_HOST = "192.168.86.201"
_VALKEY_PORT = "16379"
_MODEL = "opus[1m]"


def render_settings_json(machine: ModelMachineEntry) -> dict[str, Any]:
    """Produce the canonical settings.json shape for a machine.

    Machine-specific values (paths) come from the registry entry.
    Common values (infra host, Kafka, Postgres, etc.) are fleet-wide constants.
    """
    return {
        "$schema": _SCHEMA_URL,
        "env": {
            "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
            "OMNI_HOME": machine.omni_home,
            "ONEX_STATE_DIR": machine.onex_state_dir,
            "OMNICLAUDE_MODE": "full",
            "OMNI_INFRA_HOST": _OMNI_INFRA_HOST,
            "MAX_THINKING_TOKENS": "31999",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "POSTGRES_HOST": _POSTGRES_HOST,
            "POSTGRES_PORT": _POSTGRES_PORT,
            "VALKEY_HOST": _VALKEY_HOST,
            "VALKEY_PORT": _VALKEY_PORT,
            "KAFKA_BOOTSTRAP_SERVERS": _KAFKA_BOOTSTRAP_SERVERS,
        },
        "includeCoAuthoredBy": False,
        "model": _MODEL,
        "statusLine": {
            "type": "command",
            "command": machine.statusline_path,
            "padding": 0,
        },
        "enabledPlugins": {
            "code-review@claude-plugins-official": True,
            "onex@omninode-tools": True,
        },
        "voiceEnabled": True,
        "skipDangerousModePermissionPrompt": True,
        "autoUpdates": True,
    }


# ---------------------------------------------------------------------------
# CLI command group
# ---------------------------------------------------------------------------


@click.group("env")
def env_group() -> None:  # stub-ok: click group
    """Environment management commands."""


@env_group.command("render-settings")
@click.option("--machine-id", required=True, help="Machine ID from the registry")
@click.option(
    "--registry",
    default="config/machines.yaml",
    help="Path to machines.yaml registry file",
)
def render_settings(machine_id: str, registry: str) -> None:
    """Render the canonical settings.json for a machine."""
    reg = ModelMachineRegistry.from_yaml(Path(registry))
    machine = reg.get_machine(machine_id)
    settings = render_settings_json(machine)
    click.echo(json.dumps(settings, indent=2))
