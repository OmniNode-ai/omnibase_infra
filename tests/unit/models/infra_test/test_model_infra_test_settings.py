# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The infra-test CLI reads its endpoints and waits from a declared file (OMN-21623)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from omnibase_infra.cli.infra_test import env as env_module
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.infra_test import (
    ModelInfraTestSettings,
    load_infra_test_settings,
)

_SETTINGS_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "cli"
    / "infra_test"
    / "infra_test_settings.yaml"
)


def _declared_block() -> dict[str, object]:
    return yaml.safe_load(_SETTINGS_FILE.read_text(encoding="utf-8"))["settings"]


@pytest.mark.unit
class TestInfraTestSettingsContract:
    def test_loaded_settings_equal_the_declared_block(self) -> None:
        loaded = load_infra_test_settings()
        assert loaded.model_dump() == _declared_block()

    @pytest.mark.parametrize("key", sorted(_declared_block()))
    def test_block_missing_a_value_fails_to_load(
        self, key: str, tmp_path: Path
    ) -> None:
        block = _declared_block()
        del block[key]
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump({"settings": block}), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            ModelInfraTestSettings.from_file(path)

    def test_file_without_a_settings_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.yaml"
        path.write_text("other: 1\n", encoding="utf-8")
        with pytest.raises(ProtocolConfigurationError):
            ModelInfraTestSettings.from_file(path)

    def test_non_positive_wait_is_refused(self, tmp_path: Path) -> None:
        block = _declared_block()
        block["health_poll_interval_seconds"] = 0
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump({"settings": block}), encoding="utf-8")
        with pytest.raises(ValueError, match="health_poll_interval_seconds"):
            ModelInfraTestSettings.from_file(path)


@pytest.mark.unit
class TestCliUsesDeclaredSettings:
    def test_compose_call_carries_the_declared_timeout(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        with patch.object(env_module.subprocess, "run", return_value=completed) as run:
            env_module._run_compose("compose.yml", "proj", ["ps"])
        assert (
            run.call_args.kwargs["timeout"]
            == load_infra_test_settings().compose_command_timeout_seconds
        )

    def test_health_wait_polls_at_the_declared_interval(self) -> None:
        settings = load_infra_test_settings()
        failing = subprocess.CompletedProcess(args=[], returncode=1, stdout="")
        sleeper = MagicMock()
        with (
            patch.object(env_module.subprocess, "run", return_value=failing),
            patch.object(env_module.time, "sleep", sleeper),
            patch.object(env_module.time, "monotonic", side_effect=[0.0, 0.0, 1e9]),
        ):
            assert env_module._wait_for_healthy("compose.yml", "proj") is False
        sleeper.assert_called_once_with(settings.health_poll_interval_seconds)
