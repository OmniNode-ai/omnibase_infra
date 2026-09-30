# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the developer lane binding profile CLI (OMN-19973).

The unit modules beside this one drive the store and the transport resolver
directly. This one goes through the real ``click`` command group -- the real
option parsing, the real ``~/.onex/config.yaml`` round trip, and the real
``resolve_embedded_runtime_config`` -- because the gate the CI ticket covers is
the command line itself: a developer types ``onex profile bind-lane dev`` and
then the transport authority must answer ``kafka`` on that lane, not the
shipped in-memory default.

Home is redirected to a ``tmp_path`` and ``ONEX_CONTRACTS_DIR`` is removed, so
every byte that lands on disk and every provenance string that comes back is
from this test's own fixtures. The stand-in ``task_class_authority`` registry
from ``tests/integration/cli/conftest.py`` applies automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml
from click.testing import CliRunner, Result

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.cli.cli_profile import profile_group
from omnibase_infra.cli.store_developer_profile import StoreDeveloperProfile
from omnibase_infra.runtime.service_kernel import resolve_embedded_runtime_config

pytestmark = pytest.mark.integration


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``~/.onex`` rooted in a throwaway home, and no bootstrap pointer."""
    onex_home = tmp_path / ".onex"
    onex_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    return tmp_path


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(profile_group, args, catch_exceptions=False)


def _write_config(onex_home: Path, data: dict[str, object]) -> None:
    config_path = onex_home / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _read_config(onex_home: Path) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        yaml.safe_load((onex_home / "config.yaml").read_text(encoding="utf-8")),
    )


def test_bind_show_unbind_round_trip_through_the_real_command_group(
    tmp_home: Path,
) -> None:
    """bind -> show -> unbind, each through the real command group."""
    onex_home = tmp_home / ".onex"

    bound = _invoke(["bind-lane", "dev"])
    assert bound.exit_code == 0
    assert "Bound lane 'dev'" in bound.output

    shown = _invoke(["show"])
    assert shown.exit_code == 0
    assert "lane_binding: dev" in shown.output

    document = _read_config(onex_home)
    developer = cast("dict[str, object]", document["developer"])
    assert developer["lane_binding"] == "dev"

    unbound = _invoke(["unbind-lane"])
    assert unbound.exit_code == 0
    assert "Unbound lane 'dev'." in unbound.output

    cleared = _invoke(["show"])
    assert cleared.exit_code == 0
    assert "lane_binding: (none)" in cleared.output


def test_a_bound_lane_reaches_the_transport_authority(tmp_home: Path) -> None:
    """The binding the CLI wrote is the value the transport resolver takes."""
    onex_home = tmp_home / ".onex"

    bound = _invoke(["bind-lane", "dev"])
    assert bound.exit_code == 0

    binding = StoreDeveloperProfile(onex_home=onex_home).lane_binding()
    assert binding == "dev"

    config, source = resolve_embedded_runtime_config(developer_lane_binding=binding)
    assert config.event_bus.type is EnumEventBusType.KAFKA
    assert config.event_bus.lane == "dev"
    assert "developer lane binding 'dev'" in source

    unbound = _invoke(["unbind-lane"])
    assert unbound.exit_code == 0

    assert StoreDeveloperProfile(onex_home=onex_home).lane_binding() is None
    config_unbound, _ = resolve_embedded_runtime_config(developer_lane_binding=None)
    assert config_unbound.event_bus.lane != "dev"


def test_bind_lane_preserves_unrelated_config_blocks(tmp_home: Path) -> None:
    """Binding a lane must not touch another block of the shared config."""
    onex_home = tmp_home / ".onex"
    _write_config(
        onex_home,
        {"lanes": {"prod": {"sasl_username": "u"}}, "gateway": {"host": "gw.local"}},
    )

    bound = _invoke(["bind-lane", "dev"])
    assert bound.exit_code == 0

    document = _read_config(onex_home)
    assert document["lanes"] == {"prod": {"sasl_username": "u"}}
    assert document["gateway"] == {"host": "gw.local"}
    developer = cast("dict[str, object]", document["developer"])
    assert developer["lane_binding"] == "dev"
