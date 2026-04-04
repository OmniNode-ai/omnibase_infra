# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for onex env render-settings command."""

from __future__ import annotations

import json

from omnibase_infra.cli.cli_env import render_settings_json
from omnibase_infra.models.environment.model_machine_registry import (
    EnumMachineRole,
    ModelMachineEntry,
)

# Canonical output schema. render_settings_json must produce exactly this shape.
EXPECTED_ENV_KEYS = {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
    "OMNI_HOME",
    "ONEX_STATE_DIR",
    "OMNICLAUDE_MODE",
    "OMNI_INFRA_HOST",
    "MAX_THINKING_TOKENS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "DISABLE_TELEMETRY",
    "DISABLE_ERROR_REPORTING",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "VALKEY_HOST",
    "VALKEY_PORT",
    "KAFKA_BOOTSTRAP_SERVERS",
}
EXPECTED_TOP_KEYS = {
    "$schema",
    "env",
    "includeCoAuthoredBy",
    "model",
    "statusLine",
    "enabledPlugins",
    "voiceEnabled",
    "skipDangerousModePermissionPrompt",
    "autoUpdates",
}


def test_render_settings_for_dev_machine():
    machine = ModelMachineEntry(
        machine_id="test-dev",
        hostname="test.local",
        ip="1.1.1.1",
        role=EnumMachineRole.DEV,
        omni_home="/Users/test/Code/omni_home",
        ssh_user="test",
    )
    result = render_settings_json(machine)
    # Shape check
    assert set(result.keys()) == EXPECTED_TOP_KEYS
    assert set(result["env"].keys()) == EXPECTED_ENV_KEYS
    # Path check
    assert result["env"]["OMNI_HOME"] == "/Users/test/Code/omni_home"
    assert result["env"]["ONEX_STATE_DIR"] == "/Users/test/Code/omni_home/.onex_state"
    assert result["statusLine"]["command"].startswith("/Users/test/Code/omni_home/")
    # No empty hooks block
    assert "hooks" not in result
    # Common values (not machine-specific)
    assert result["env"]["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
    assert result["env"]["OMNI_INFRA_HOST"] == "192.168.86.201"
    assert result["model"] == "opus[1m]"
    # Unset values are omitted, not null
    for v in result["env"].values():
        assert v is not None


def test_render_settings_for_infra_machine():
    machine = ModelMachineEntry(
        machine_id="infra",
        hostname="srv",
        ip="2.2.2.2",
        role=EnumMachineRole.INFRA,
        omni_home="/data/omninode/omni_home",
        ssh_user="jonah",
        home_dir="/home/jonah",
    )
    result = render_settings_json(machine)
    assert result["env"]["OMNI_HOME"] == "/data/omninode/omni_home"
    assert result["env"]["ONEX_STATE_DIR"] == "/data/omninode/omni_home/.onex_state"
    assert result["statusLine"]["command"].startswith("/data/omninode/")


def test_rendered_json_is_valid():
    """Rendered output must be valid JSON when serialized."""
    machine = ModelMachineEntry(
        machine_id="t",
        hostname="h",
        ip="1.1.1.1",
        role=EnumMachineRole.DEV,
        omni_home="/x",
        ssh_user="u",
    )
    result = render_settings_json(machine)
    serialized = json.dumps(result)
    reparsed = json.loads(serialized)
    assert reparsed == result
