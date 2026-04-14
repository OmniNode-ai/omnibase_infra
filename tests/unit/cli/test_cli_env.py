# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for onex env render-settings CLI command. [OMN-7528]"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from omnibase_infra.cli.cli_env import render_settings_json
from omnibase_infra.cli.commands import cli
from omnibase_infra.models.environment.model_machine_registry import (
    EnumMachineRole,
    ModelMachineEntry,
)

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


def _make_entry(**kwargs) -> ModelMachineEntry:
    defaults = {
        "machine_id": "test-dev",
        "hostname": "test.local",
        "role": EnumMachineRole.DEV_WORKSTATION,
        "omni_home": "/Users/test/Code/omni_home",
        "resolved_home_dir": "/Users/test",
    }
    defaults.update(kwargs)
    return ModelMachineEntry(**defaults)


class TestRenderSettingsJson:
    def test_top_keys_exact(self) -> None:
        result = render_settings_json(_make_entry())
        assert set(result.keys()) == EXPECTED_TOP_KEYS

    def test_env_keys_exact(self) -> None:
        result = render_settings_json(_make_entry())
        assert set(result["env"].keys()) == EXPECTED_ENV_KEYS

    def test_omni_home_derived(self) -> None:
        result = render_settings_json(
            _make_entry(omni_home="/Users/test/Code/omni_home")
        )
        assert result["env"]["OMNI_HOME"] == "/Users/test/Code/omni_home"

    def test_onex_state_dir_derived(self) -> None:
        result = render_settings_json(
            _make_entry(omni_home="/Users/test/Code/omni_home")
        )
        assert (
            result["env"]["ONEX_STATE_DIR"] == "/Users/test/Code/omni_home/.onex_state"
        )

    def test_statusline_path_starts_with_omni_home(self) -> None:
        result = render_settings_json(
            _make_entry(omni_home="/Users/test/Code/omni_home")
        )
        assert result["statusLine"]["command"].startswith("/Users/test/Code/omni_home/")

    def test_no_hooks_block(self) -> None:
        result = render_settings_json(_make_entry())
        assert "hooks" not in result

    def test_fleet_constants(self) -> None:
        result = render_settings_json(_make_entry())
        assert result["env"]["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
        assert result["env"]["OMNI_INFRA_HOST"] == "192.168.86.201"
        assert result["model"] == "opus[1m]"

    def test_no_null_env_values(self) -> None:
        result = render_settings_json(_make_entry())
        for v in result["env"].values():
            assert v is not None

    def test_infra_machine_paths(self) -> None:
        m = _make_entry(
            machine_id="infra-server",
            hostname="infra-server",
            role=EnumMachineRole.INFRA_SERVER,
            omni_home="/home/jonah/Code/omni_home",
            resolved_home_dir="/home/jonah",
        )
        result = render_settings_json(m)
        assert result["env"]["OMNI_HOME"] == "/home/jonah/Code/omni_home"
        assert (
            result["env"]["ONEX_STATE_DIR"] == "/home/jonah/Code/omni_home/.onex_state"
        )
        assert result["statusLine"]["command"].startswith("/home/jonah/Code/omni_home/")

    def test_is_valid_json(self) -> None:
        result = render_settings_json(_make_entry())
        serialized = json.dumps(result)
        reparsed = json.loads(serialized)
        assert reparsed == result


class TestRenderSettingsCli:
    def _make_registry_yaml(
        self, tmp_path: Path, machine_id: str, omni_home: str, role: str
    ) -> Path:
        data = {
            "machines": [
                {
                    "machine_id": machine_id,
                    "hostname": f"{machine_id}.local",
                    "role": role,
                    "omni_home": omni_home,
                    "resolved_home_dir": str(Path(omni_home).parents[2]),
                }
            ]
        }
        p = tmp_path / "machines.yaml"
        p.write_text(yaml.dump(data))
        return p

    def test_cli_infra_server(self, tmp_path: Path) -> None:
        reg = self._make_registry_yaml(
            tmp_path, "infra-server", "/home/jonah/Code/omni_home", "infra_server"
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "env",
                "render-settings",
                "--machine-id",
                "infra-server",
                "--registry",
                str(reg),
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["env"]["OMNI_HOME"] == "/home/jonah/Code/omni_home"

    def test_cli_m2_ultra(self, tmp_path: Path) -> None:
        reg = self._make_registry_yaml(
            tmp_path, "m2-ultra", "/Users/jonah/Code/omni_home", "dev_workstation"
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "env",
                "render-settings",
                "--machine-id",
                "m2-ultra",
                "--registry",
                str(reg),
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["env"]["OMNI_HOME"] == "/Users/jonah/Code/omni_home"

    def test_cli_unknown_machine_id_fails(self, tmp_path: Path) -> None:
        reg = self._make_registry_yaml(
            tmp_path, "infra-server", "/home/jonah/Code/omni_home", "infra_server"
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "env",
                "render-settings",
                "--machine-id",
                "unknown",
                "--registry",
                str(reg),
            ],
        )
        assert result.exit_code != 0

    def test_cli_output_shape(self, tmp_path: Path) -> None:
        reg = self._make_registry_yaml(
            tmp_path, "test", "/Users/test/Code/omni_home", "dev_workstation"
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["env", "render-settings", "--machine-id", "test", "--registry", str(reg)],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert set(parsed.keys()) == EXPECTED_TOP_KEYS
        assert set(parsed["env"].keys()) == EXPECTED_ENV_KEYS
