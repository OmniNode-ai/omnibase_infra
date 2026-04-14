# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for onex env render-settings CLI command. [OMN-7528]"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.cli.commands import cli


@pytest.mark.integration
class TestEnvRenderSettingsCLI:
    """Integration tests for the env render-settings subcommand."""

    def _make_registry_yaml(self, tmp_path: Path) -> Path:
        data = {
            "machines": [
                {
                    "machine_id": "test-server",
                    "hostname": "test-server.local",
                    "role": "dev_workstation",
                    "omni_home": str(tmp_path / "omni_home"),
                    "resolved_home_dir": str(tmp_path),
                }
            ]
        }
        registry = tmp_path / "machines.yaml"
        registry.write_text(yaml.dump(data))
        return registry

    def test_render_settings_outputs_valid_json(self, tmp_path: Path) -> None:
        """render-settings command emits parseable JSON with required top-level keys."""
        registry = self._make_registry_yaml(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "env",
                "render-settings",
                "--machine-id",
                "test-server",
                "--registry",
                str(registry),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "env" in parsed
        assert "OMNI_HOME" in parsed["env"]
        assert "ONEX_STATE_DIR" in parsed["env"]

    def test_render_settings_omni_home_from_registry(self, tmp_path: Path) -> None:
        """OMNI_HOME in output matches the registry entry's omni_home field."""
        expected_home = str(tmp_path / "omni_home")
        registry = self._make_registry_yaml(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "env",
                "render-settings",
                "--machine-id",
                "test-server",
                "--registry",
                str(registry),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["env"]["OMNI_HOME"] == expected_home

    def test_render_settings_unknown_machine_id_exits_nonzero(
        self, tmp_path: Path
    ) -> None:
        """render-settings exits non-zero when machine-id is not in registry."""
        registry = self._make_registry_yaml(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "env",
                "render-settings",
                "--machine-id",
                "nonexistent",
                "--registry",
                str(registry),
            ],
        )
        assert result.exit_code != 0
