# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex env render-settings`` ships no infrastructure host (OMN-20931).

The infra host used to default to the lab's LAN address, so a fork or a
customer with no ``OMNI_INFRA_HOST`` got settings that pointed every client at
our lab. The host is now a required setting: unset, the render refuses and
names the variable; set, the rendered settings carry exactly that host.
"""

from __future__ import annotations

from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from omnibase_infra.cli.cli_env import render_settings_json
from omnibase_infra.cli.commands import cli
from omnibase_infra.models.environment.model_machine_registry import ModelMachineEntry
from tests.unit.cli.test_cli_env import _make_entry

# RFC 5737 documentation address: never a real host.
_DOC_HOST = "192.0.2.10"
_DERIVED = (
    "OMNI_INFRA_HOST",
    "POSTGRES_HOST",
    "VALKEY_HOST",
    "KAFKA_BOOTSTRAP_SERVERS",
)


@pytest.fixture
def no_infra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _DERIVED:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.unit
@pytest.mark.usefixtures("no_infra_env")
def test_render_refuses_without_an_infra_host() -> None:
    with pytest.raises(click.ClickException, match="OMNI_INFRA_HOST"):
        render_settings_json(_make_entry())


@pytest.mark.unit
@pytest.mark.usefixtures("no_infra_env")
def test_render_carries_the_configured_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNI_INFRA_HOST", _DOC_HOST)
    env = render_settings_json(_make_entry())["env"]
    assert isinstance(env, dict)
    assert env["OMNI_INFRA_HOST"] == _DOC_HOST
    assert env["POSTGRES_HOST"] == _DOC_HOST
    assert env["VALKEY_HOST"] == _DOC_HOST
    assert env["KAFKA_BOOTSTRAP_SERVERS"] == f"{_DOC_HOST}:19092"


@pytest.mark.unit
@pytest.mark.usefixtures("no_infra_env")
def test_cli_exits_nonzero_and_names_the_variable(tmp_path: Path) -> None:
    registry = tmp_path / "machines.yaml"
    entry = _make_entry().model_dump(
        mode="json", include=set(ModelMachineEntry.model_fields)
    )
    registry.write_text(yaml.safe_dump({"machines": [entry]}), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "env",
            "render-settings",
            "--machine-id",
            "test-dev",
            "--registry",
            str(registry),
        ],
    )
    assert result.exit_code != 0
    assert "OMNI_INFRA_HOST" in result.output
