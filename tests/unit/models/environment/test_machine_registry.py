# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelMachineRegistry and ModelMachineEntry. [OMN-7527]"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.models.environment.model_machine_registry import (
    EnumMachineRole,
    ModelMachineEntry,
    ModelMachineRegistry,
)


def _make_entry(**kwargs) -> ModelMachineEntry:
    defaults = {
        "machine_id": "test",
        "hostname": "test.local",
        "role": EnumMachineRole.DEV_WORKSTATION,
        "omni_home": "/Users/test/Code/omni_home",
        "resolved_home_dir": "/Users/test",
    }
    defaults.update(kwargs)
    return ModelMachineEntry(**defaults)


class TestModelMachineEntry:
    def test_computed_onex_state_dir(self) -> None:
        m = _make_entry(omni_home="/Users/test/Code/omni_home")
        assert m.onex_state_dir == "/Users/test/Code/omni_home/.onex_state"

    def test_computed_statusline_path(self) -> None:
        m = _make_entry(omni_home="/Users/test/Code/omni_home")
        assert m.statusline_path == (
            "/Users/test/Code/omni_home/omniclaude/plugins/onex/hooks/scripts/statusline.sh"
        )

    def test_computed_plugin_path(self) -> None:
        m = _make_entry(omni_home="/Users/test/Code/omni_home")
        assert m.plugin_path == "/Users/test/Code/omni_home/omniclaude/plugins/onex"

    def test_computed_claude_settings_path(self) -> None:
        m = _make_entry(resolved_home_dir="/Users/test")
        assert m.claude_settings_path == "/Users/test/.claude/settings.json"

    def test_relative_omni_home_rejected(self) -> None:
        with pytest.raises(ValidationError, match="omni_home must be absolute"):
            _make_entry(omni_home="relative/path")

    def test_relative_resolved_home_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="resolved_home_dir must be absolute"):
            _make_entry(resolved_home_dir="relative/home")

    def test_infra_machine_paths(self) -> None:
        m = _make_entry(
            machine_id="infra",
            hostname="infra-server",
            role=EnumMachineRole.INFRA_SERVER,
            omni_home="/home/jonah/Code/omni_home",
            resolved_home_dir="/home/jonah",
        )
        assert m.onex_state_dir == "/home/jonah/Code/omni_home/.onex_state"
        assert m.statusline_path.startswith("/home/jonah/Code/omni_home/")
        assert m.claude_settings_path == "/home/jonah/.claude/settings.json"


class TestModelMachineRegistry:
    def test_get_machine_found(self) -> None:
        reg = ModelMachineRegistry(machines=[_make_entry(machine_id="x")])
        m = reg.get_machine("x")
        assert m.machine_id == "x"

    def test_get_machine_not_found(self) -> None:
        reg = ModelMachineRegistry(machines=[_make_entry(machine_id="x")])
        with pytest.raises(KeyError):
            reg.get_machine("missing")

    def test_get_machines_by_role(self) -> None:
        reg = ModelMachineRegistry(
            machines=[
                _make_entry(machine_id="a", role=EnumMachineRole.DEV_WORKSTATION),
                _make_entry(machine_id="b", role=EnumMachineRole.INFRA_SERVER),
                _make_entry(machine_id="c", role=EnumMachineRole.DEV_WORKSTATION),
            ]
        )
        devs = reg.get_machines_by_role(EnumMachineRole.DEV_WORKSTATION)
        assert len(devs) == 2
        assert all(m.role == EnumMachineRole.DEV_WORKSTATION for m in devs)

    def test_duplicate_machine_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate machine_id"):
            ModelMachineRegistry(
                machines=[
                    _make_entry(machine_id="dup", hostname="h1"),
                    _make_entry(machine_id="dup", hostname="h2"),
                ]
            )

    def test_from_yaml(self, tmp_path: Path) -> None:
        data = {
            "machines": [
                {
                    "machine_id": "infra-server",
                    "hostname": "infra-server",
                    "role": "infra_server",
                    "omni_home": "/home/jonah/Code/omni_home",
                    "resolved_home_dir": "/home/jonah",
                }
            ]
        }
        yaml_path = tmp_path / "machines.yaml"
        yaml_path.write_text(yaml.dump(data))
        reg = ModelMachineRegistry.from_yaml(yaml_path)
        assert reg.get_machine("infra-server").role == EnumMachineRole.INFRA_SERVER

    def test_resolve_local_machine_no_match(self) -> None:
        reg = ModelMachineRegistry(
            machines=[_make_entry(machine_id="x", hostname="definitely-not-this-host")]
        )
        assert reg.resolve_local_machine() is None

    def test_seed_file_loads(self) -> None:
        # Walk up from tests/unit/models/environment/ to repo root
        repo_root = Path(__file__).parents[4]
        seed = repo_root / "config" / "machines.yaml"
        reg = ModelMachineRegistry.from_yaml(seed)
        assert len(reg.machines) >= 2
        assert reg.get_machine("infra-server") is not None
        assert reg.get_machine("m2-ultra") is not None
