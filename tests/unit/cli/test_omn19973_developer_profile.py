# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.cli_profile import profile_group
from omnibase_infra.cli.store_developer_profile import StoreDeveloperProfile

pytestmark = pytest.mark.unit


@pytest.fixture
def tmp_path_home(tmp_path):
    return tmp_path


@pytest.fixture
def onex_home(tmp_path_home, monkeypatch):
    home = tmp_path_home / ".onex"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path_home)
    return home


def _write_config(onex_home: Path, data: dict) -> None:
    config_path = onex_home / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f)


def test_no_config_file_lane_binding_is_none(onex_home):
    store = StoreDeveloperProfile(onex_home=onex_home)
    assert store.lane_binding() is None


def test_bind_then_read_back(onex_home):
    store = StoreDeveloperProfile(onex_home=onex_home)
    store.bind_lane("dev")
    assert store.lane_binding() == "dev"


def test_bind_preserves_existing_blocks_and_keys(onex_home):
    _write_config(
        onex_home,
        {
            "lanes": {"prod": {"sasl_username": "prod-user"}},
            "gateway": {"host": "gw.local"},
            "developer": {"other_key": "value", "another": 42},
        },
    )
    store = StoreDeveloperProfile(onex_home=onex_home)
    store.bind_lane("dev")

    with open(onex_home / "config.yaml") as f:
        data = yaml.safe_load(f)

    assert data["lanes"] == {"prod": {"sasl_username": "prod-user"}}
    assert data["gateway"] == {"host": "gw.local"}
    assert data["developer"]["other_key"] == "value"
    assert data["developer"]["another"] == 42
    assert data["developer"]["lane_binding"] == "dev"


def test_unbind_removes_key_and_emptied_block(onex_home):
    _write_config(onex_home, {"developer": {"lane_binding": "dev"}})
    store = StoreDeveloperProfile(onex_home=onex_home)

    assert store.unbind_lane() is True
    with open(onex_home / "config.yaml") as f:
        data = yaml.safe_load(f)
    assert "developer" not in data

    assert store.unbind_lane() is False


def test_developer_list_raises_configuration_parse_error(onex_home):
    _write_config(onex_home, {"developer": [1, 2]})
    store = StoreDeveloperProfile(onex_home=onex_home)
    with pytest.raises(ModelOnexError) as excinfo:
        store.lane_binding()
    assert excinfo.value.error_code == EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR


def test_bind_refuses_non_mapping_developer_block(onex_home):
    _write_config(onex_home, {"developer": "not-a-mapping"})
    store = StoreDeveloperProfile(onex_home=onex_home)
    with pytest.raises(ModelOnexError) as excinfo:
        store.bind_lane("dev")
    assert excinfo.value.error_code == EnumCoreErrorCode.CONFIGURATION_PARSE_ERROR
    with open(onex_home / "config.yaml") as f:
        assert yaml.safe_load(f) == {"developer": "not-a-mapping"}


def test_developer_lane_binding_whitespace_raises_invalid_configuration(onex_home):
    _write_config(onex_home, {"developer": {"lane_binding": "  "}})
    store = StoreDeveloperProfile(onex_home=onex_home)
    with pytest.raises(ModelOnexError) as excinfo:
        store.lane_binding()
    assert excinfo.value.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION


def test_bind_lane_whitespace_raises_invalid_configuration(onex_home):
    store = StoreDeveloperProfile(onex_home=onex_home)
    with pytest.raises(ModelOnexError) as excinfo:
        store.bind_lane("  ")
    assert excinfo.value.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION


def test_cli_bind_lane_no_stored_identity_shows_warning(onex_home):
    runner = CliRunner()
    result = runner.invoke(profile_group, ["bind-lane", "dev"])
    assert result.exit_code == 0
    assert "WARNING" in result.stderr
    store = StoreDeveloperProfile(onex_home=onex_home)
    assert store.lane_binding() == "dev"


def test_cli_bind_lane_with_existing_lanes_no_warning(onex_home):
    _write_config(
        onex_home,
        {
            "lanes": {
                "dev": {"sasl_username": "u", "sasl_password_ref": "dev-lane-sasl"}
            }
        },
    )
    runner = CliRunner()
    result = runner.invoke(profile_group, ["bind-lane", "dev"])
    assert result.exit_code == 0
    assert "WARNING" not in result.stderr
    store = StoreDeveloperProfile(onex_home=onex_home)
    assert store.lane_binding() == "dev"


def test_cli_show_before_and_after_bind(onex_home):
    runner = CliRunner()

    result_before = runner.invoke(profile_group, ["show"])
    assert result_before.exit_code == 0
    assert "lane_binding: (none)" in result_before.output

    runner.invoke(profile_group, ["bind-lane", "dev"])

    result_after = runner.invoke(profile_group, ["show"])
    assert result_after.exit_code == 0
    assert "lane_binding: dev" in result_after.output


def test_cli_unbind_lane_messages(onex_home):
    runner = CliRunner()

    runner.invoke(profile_group, ["bind-lane", "dev"])
    result_unbind = runner.invoke(profile_group, ["unbind-lane"])
    assert result_unbind.exit_code == 0
    assert "Unbound lane 'dev'." in result_unbind.output

    result_no_bind = runner.invoke(profile_group, ["unbind-lane"])
    assert result_no_bind.exit_code == 0
    assert "No lane was bound." in result_no_bind.output


def test_profile_group_is_a_shipped_onex_command() -> None:
    """``onex profile`` reaches users only through the onex.cli entry point."""
    import tomllib

    pyproject = Path(__file__).parents[3] / "pyproject.toml"
    entry_points = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "entry-points"
    ]["onex.cli"]
    assert entry_points.get("profile") == "omnibase_infra.cli.cli_profile:profile_group"
    assert set(profile_group.commands) == {"bind-lane", "unbind-lane", "show"}
