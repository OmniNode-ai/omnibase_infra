# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for machine registry model."""

import socket

import pytest
from pydantic import ValidationError

from omnibase_infra.models.environment.model_machine_registry import (
    EnumMachineRole,
    ModelMachineEntry,
    ModelMachineRegistry,
)


def test_machine_entry_dev_role():
    entry = ModelMachineEntry(
        machine_id="m2-ultra",
        hostname="Stickybeatz-Studio.local",
        ip="192.168.86.101",
        role=EnumMachineRole.DEV,
        omni_home="/Users/jonah/Code/omni_home",
        ssh_user="jonah",
    )
    assert entry.onex_state_dir == "/Users/jonah/Code/omni_home/.onex_state"
    assert entry.claude_settings_path == "/Users/jonah/.claude/settings.json"


def test_machine_entry_infra_role():
    entry = ModelMachineEntry(
        machine_id="infra-server",
        hostname="omninode-infra",
        ip="192.168.86.201",
        role=EnumMachineRole.INFRA,
        omni_home="/data/omninode/omni_home",
        ssh_user="jonah",
        home_dir="/home/jonah",
    )
    assert entry.onex_state_dir == "/data/omninode/omni_home/.onex_state"
    assert (
        entry.statusline_path
        == "/data/omninode/omni_home/omniclaude/plugins/onex/hooks/scripts/statusline.sh"
    )
    assert entry.claude_settings_path == "/home/jonah/.claude/settings.json"


def test_relative_omni_home_rejected():
    with pytest.raises(ValidationError, match="omni_home must be absolute"):
        ModelMachineEntry(
            machine_id="bad",
            hostname="h",
            ip="1.1.1.1",
            role=EnumMachineRole.DEV,
            omni_home="relative/path",
            ssh_user="u",
        )


def test_duplicate_machine_id_rejected():
    with pytest.raises(ValidationError, match="Duplicate machine_id"):
        ModelMachineRegistry(
            machines=[
                ModelMachineEntry(
                    machine_id="dup",
                    hostname="h1",
                    ip="1.1.1.1",
                    role=EnumMachineRole.DEV,
                    omni_home="/a",
                    ssh_user="u",
                ),
                ModelMachineEntry(
                    machine_id="dup",
                    hostname="h2",
                    ip="2.2.2.2",
                    role=EnumMachineRole.DEV,
                    omni_home="/b",
                    ssh_user="u",
                ),
            ]
        )


def test_registry_loads_from_yaml(tmp_path):
    yaml_content = """
machines:
  - machine_id: test-dev
    hostname: test.local
    ip: 192.168.86.100
    role: dev
    omni_home: /Users/test/Code/omni_home
    ssh_user: test
"""
    f = tmp_path / "machines.yaml"
    f.write_text(yaml_content)
    registry = ModelMachineRegistry.from_yaml(f)
    assert len(registry.machines) == 1
    assert registry.get_machine("test-dev").ip == "192.168.86.100"


def test_registry_get_by_role():
    registry = ModelMachineRegistry(
        machines=[
            ModelMachineEntry(
                machine_id="dev1",
                hostname="h1",
                ip="1.1.1.1",
                role=EnumMachineRole.DEV,
                omni_home="/a",
                ssh_user="u",
            ),
            ModelMachineEntry(
                machine_id="infra1",
                hostname="h2",
                ip="2.2.2.2",
                role=EnumMachineRole.INFRA,
                omni_home="/b",
                ssh_user="u",
            ),
        ]
    )
    infra = registry.get_machines_by_role(EnumMachineRole.INFRA)
    assert len(infra) == 1
    assert infra[0].machine_id == "infra1"


def test_resolve_local_machine():
    """Local machine resolved by matching hostname at runtime."""
    local_hostname = socket.gethostname().split(".")[0]
    registry = ModelMachineRegistry(
        machines=[
            ModelMachineEntry(
                machine_id="this-one",
                hostname=local_hostname,
                ip="127.0.0.1",
                role=EnumMachineRole.DEV,
                omni_home="/x",
                ssh_user="u",
            ),
            ModelMachineEntry(
                machine_id="other",
                hostname="remote.host",
                ip="2.2.2.2",
                role=EnumMachineRole.DEV,
                omni_home="/y",
                ssh_user="u",
            ),
        ]
    )
    local = registry.resolve_local_machine()
    assert local is not None
    assert local.machine_id == "this-one"
