# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19543 -- a deploy-agent instance is data.

AC1 (executor half): each non-default instance's dev-lane composition is read
from its ``lane:`` block in ``config/deploy_lane_routing.yaml``; a fixture table
naming ``dev-999`` builds a composition with no Python edit, and dev-202's
composition is exactly the one OMN-19522 wrote as a literal.

AC4 (repo half): a LaunchAgent, rendered from the one template, carries the
systemd unit's environment name for name, a health port in its block bound to
loopback, and its own clone.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from deploy_agent.events import DEV_LANE_GATEWAY_SERVICES, EnumRuntimeLane
from deploy_agent.executor import (
    _LANE_CONFIGS,
    COMPOSE_FILE,
    DEFAULT_DEV_INSTANCE,
    DEV_INSTANCE_LANE_CONFIGS,
    EnumInstancePhase,
    ModelLaneConfig,
    build_dev_instance_lane_configs,
    lane_with_phases_off,
    select_dev_instance,
)
from deploy_agent.instance_lanes import parse_instance_lanes
from deploy_agent.routing import (
    AGENT_CLONE_ROOT,
    RoutingTableError,
    lane_flags_for_instance,
    parse_routing_table,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_LAUNCHD = _DEPLOY_DIR / "launchd"
_RENDER = _LAUNCHD / "render-deploy-agent-plist.sh"
_DEV_202_UNIT = _DEPLOY_DIR / "deploy-agent-dev-202.service"

#: The port block the LaunchAgent health port must sit in (OMN-19543 AC3).
_LAUNCHD_BLOCK = range(42000, 43000)
_LAUNCHD_AGENT_PORT = 42098
_LAUNCHD_INSTANCE = "dev-999"
_LAUNCHD_HOME = (
    "/Users/labuser"  # local-path-ok: the fake user home the renderer is given
)


@pytest.fixture(autouse=True)
def _reset_instance() -> Any:
    yield
    select_dev_instance(DEFAULT_DEV_INSTANCE)


# --------------------------------------------------------------------------- #
# AC1 -- the composition is data                                               #
# --------------------------------------------------------------------------- #
_FIXTURE_TABLE = """
default_instance: dev-201
instances:
  dev-201:
    hostnames: [omninode-pc]
    consumer_group: onex-deploy-agent
  dev-999:
    hostnames: [somewhere]
    consumer_group: onex-deploy-agent-dev-999
    lane:
      compose_overlay: docker/docker-compose.dev-999.yml
      compose_project: omnibase-infra-dev-999
      postgres_container: omnibase-infra-dev-999-postgres
      runtime_container: omninode-dev-999-runtime
      health_ports: {main: 44085, effects: 44086}
      disabled_phases: [lab-overlay]
      disabled_services: [onex-api]
      receipt_lane: compose-dev-999
      proves: [omnimarket]
routes: []
"""


class TestInstanceCompositionIsData:
    def test_an_unknown_instance_needs_no_python_edit(self) -> None:
        configs = build_dev_instance_lane_configs(
            parse_instance_lanes(_FIXTURE_TABLE), repo_dir="/r"
        )
        cfg = configs["dev-999"]
        assert cfg.lane is EnumRuntimeLane.DEV
        assert cfg.compose_project == cfg.build_project == "omnibase-infra-dev-999"
        assert cfg.compose_files[-1] == "/r/docker/docker-compose.dev-999.yml"
        assert [Path(f).name for f in cfg.compose_files] == [
            "docker-compose.infra.yml",
            "docker-compose.dev-lane.yml",
            "docker-compose.dev-999.yml",
        ]
        assert cfg.runtime_health_targets == (
            ("omninode-runtime", 44085),
            ("runtime-effects", 44086),
        )
        assert cfg.main_runtime_container == "omninode-dev-999-runtime"
        assert cfg.disabled_phases == frozenset({EnumInstancePhase.LAB_OVERLAY})
        # gateway-deploy is NOT disabled here, so its services are not either.
        assert cfg.disabled_services == frozenset({"onex-api"})
        # The default declares no ``lane:`` block; its phases are the table's
        # ``flags:`` (OMN-21629), applied over the historical composition.
        assert configs[DEFAULT_DEV_INSTANCE] == lane_with_phases_off(
            _LANE_CONFIGS[EnumRuntimeLane.DEV],
            lane_flags_for_instance(
                AGENT_CLONE_ROOT, DEFAULT_DEV_INSTANCE
            ).disabled_phases,
        )

    def test_dev_202_is_exactly_the_omn19522_literal(self) -> None:
        """The composition OMN-19522 wrote in executor.py, restated here, is what
        the table now yields: moving it to data changed nothing on .202."""
        literal = ModelLaneConfig(
            lane=EnumRuntimeLane.DEV,
            compose_files=(
                COMPOSE_FILE,
                _LANE_CONFIGS[EnumRuntimeLane.DEV].compose_files[1],
                COMPOSE_FILE.replace(
                    "docker-compose.infra.yml", "docker-compose.dev-202.yml"
                ),
            ),
            compose_project="omnibase-infra-dev-202",
            build_project="omnibase-infra-dev-202",
            postgres_container="omnibase-infra-dev-202-postgres",
            runtime_health_targets=(
                ("omninode-runtime", 61085),
                ("runtime-effects", 61086),
            ),
            runtime_container="omninode-dev-202-runtime",
            disabled_phases=frozenset(EnumInstancePhase),
            disabled_services=frozenset(
                {
                    "onex-api",
                    "cloud-migration-files",
                    "cloud-migration",
                    "keycloak",
                    "infisical",
                    *DEV_LANE_GATEWAY_SERVICES,
                }
            ),
        )
        assert DEV_INSTANCE_LANE_CONFIGS["dev-202"] == literal

    def test_the_default_instance_may_not_declare_a_lane_block(self) -> None:
        text = _FIXTURE_TABLE.replace(
            "    consumer_group: onex-deploy-agent\n",
            "    consumer_group: onex-deploy-agent\n    lane: {}\n",
            1,
        )
        with pytest.raises(RoutingTableError, match="default instance"):
            parse_instance_lanes(text)

    def test_a_non_default_instance_without_a_block_refuses(self) -> None:
        text = _FIXTURE_TABLE.split("    lane:\n", maxsplit=1)[0] + "routes: []\n"
        with pytest.raises(RoutingTableError, match="no lane: block"):
            parse_instance_lanes(text)

    @pytest.mark.parametrize(
        ("old", "new", "match"),
        [
            ("      proves: [omnimarket]\n", "", "proves"),
            (
                "      proves: [omnimarket]\n",
                "      proves: [omnimarket]\n      extra_key: 1\n",
                "extra_key",
            ),
            ("{main: 44085, effects: 44086}", "{main: 44085}", "effects"),
        ],
    )
    def test_a_malformed_block_refuses(self, old: str, new: str, match: str) -> None:
        with pytest.raises(RoutingTableError, match=match):
            parse_instance_lanes(_FIXTURE_TABLE.replace(old, new, 1))

    def test_an_unknown_phase_refuses(self) -> None:
        with pytest.raises(RoutingTableError, match="reboot-host"):
            parse_instance_lanes(
                _FIXTURE_TABLE.replace("[lab-overlay]", "[lab-overlay, reboot-host]")
            )

    def test_two_instances_may_not_share_a_receipt_lane(self) -> None:
        block = _FIXTURE_TABLE.split("  dev-999:\n")[1].split("routes: []")[0]
        text = _FIXTURE_TABLE.replace(
            "routes: []", "  dev-998:\n" + block + "routes: []"
        )
        with pytest.raises(RoutingTableError, match="same receipt_lane"):
            parse_instance_lanes(text)

    def test_the_router_still_parses_a_table_carrying_lane_blocks(self) -> None:
        """An agent from before OMN-19543 reads this table at a command's ref:
        its parser ignores the lane key, so routing is unchanged."""
        table = parse_routing_table(_FIXTURE_TABLE)
        assert set(table.instances) == {"dev-201", "dev-999"}


# --------------------------------------------------------------------------- #
# AC4 -- the LaunchAgent                                                       #
# --------------------------------------------------------------------------- #
_ENVIRONMENT_RE = re.compile(r'^Environment="?([A-Za-z_][A-Za-z0-9_]*)=')


def _unit_env_names(unit: Path) -> set[str]:
    return {
        m.group(1)
        for line in unit.read_text().splitlines()
        if (m := _ENVIRONMENT_RE.match(line))
    }


def _render(
    *args: str, home: str = "/Users/labuser"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_RENDER), *args],
        capture_output=True,
        text=True,
        check=False,
        env={"HOME": home, "PATH": "/usr/bin:/bin"},
    )


def _rendered_instance() -> dict[str, Any]:
    result = _render(
        _LAUNCHD_INSTANCE,
        str(_LAUNCHD_AGENT_PORT),
        f"{_LAUNCHD_HOME}/onex-lanes/{_LAUNCHD_INSTANCE}",
    )
    assert result.returncode == 0, result.stderr
    assert "@" not in result.stdout.split("-->", 1)[1], "an unrendered @TOKEN@"
    return plistlib.loads(result.stdout.encode())


class TestLaunchAgent:
    def test_the_plist_names_its_instance_and_its_own_clone(self) -> None:
        plist = _rendered_instance()
        env = plist["EnvironmentVariables"]
        root = f"{_LAUNCHD_HOME}/onex-lanes/{_LAUNCHD_INSTANCE}"
        assert plist["Label"] == f"ai.omninode.deploy-agent-{_LAUNCHD_INSTANCE}"
        assert env["DEPLOY_AGENT_INSTANCE"] == _LAUNCHD_INSTANCE
        assert env["DEPLOY_AGENT_DIR"] == f"{root}/omnibase_infra/scripts/deploy-agent"
        assert env["DEPLOY_AGENT_PYTHON"].startswith(f"{root}/omnibase_infra/")
        assert env["DEPLOY_AGENT_REPO_DIR"] == f"{root}/omni_home/omnibase_infra"
        assert env["OMNI_HOME"] == f"{root}/omni_home"
        assert (
            env["DEPLOY_AGENT_ENV_FILE"]
            == f"{_LAUNCHD_HOME}/.omnibase/{_LAUNCHD_INSTANCE}.env"
        )
        assert env["DEPLOY_AGENT_STATE_DIR"].endswith(f"/jobs-{_LAUNCHD_INSTANCE}")
        assert plist["ProgramArguments"][-1].endswith(
            "deploy/launchd/deploy-agent-launchd.sh"
        )
        assert plist["KeepAlive"] == {"SuccessfulExit": False}

    def test_the_plist_carries_the_dev_202_unit_environment_name_for_name(
        self,
    ) -> None:
        env = _rendered_instance()["EnvironmentVariables"]
        unit = _unit_env_names(_DEV_202_UNIT)
        assert sorted(unit - set(env)) == []
        # What the plist adds, each for a stated reason in the template. PATH
        # is not among them: the dev-202 unit declares its own (OMN-19522), so
        # the first assertion above already requires the plist to carry it.
        # Its value is per platform (launchd has no %h), so it is not compared.
        assert set(env) - unit == {
            "HOME",
            "OMNI_HOME",
            "DEPLOY_AGENT_REPO_DIR",
            "DEPLOY_AGENT_BIND_HOST",
        }
        for name in (
            "DEPLOY_AGENT_TRACKING_REF",
            "DEPLOY_AGENT_ALLOWED_LANES",
            "KAFKA_BOOTSTRAP_SERVERS",
            "KAFKA_ENVIRONMENT",
            "KAFKA_SECURITY_PROTOCOL",
            "KAFKA_SASL_MECHANISM",
            "KAFKA_SASL_ENV_PREFIX",
        ):
            dev_202 = next(
                line.split(f"{name}=", 1)[1].rstrip('"')
                for line in _DEV_202_UNIT.read_text().splitlines()
                if line.startswith(f"Environment={name}=")
            )
            assert env[name] == dev_202, name

    def test_the_plist_protects_every_name_it_sets(self) -> None:
        env = _rendered_instance()["EnvironmentVariables"]
        assert sorted(set(env) - set(env["DEPLOY_AGENT_ENV_PROTECTED"].split())) == []

    def test_the_health_port_is_in_the_block_on_loopback(self) -> None:
        env = _rendered_instance()["EnvironmentVariables"]
        assert int(env["DEPLOY_AGENT_PORT"]) in _LAUNCHD_BLOCK
        assert env["DEPLOY_AGENT_BIND_HOST"] == "127.0.0.1"

    @pytest.mark.parametrize(
        "args",
        [
            (_LAUNCHD_INSTANCE, "42098"),
            ("dev_999", "42098", "/x"),
            (_LAUNCHD_INSTANCE, "61098", "/x"),
            (_LAUNCHD_INSTANCE, "80", "/x"),
            (_LAUNCHD_INSTANCE, "42098", "relative/root"),
        ],
    )
    def test_the_renderer_refuses_bad_arguments(self, args: tuple[str, ...]) -> None:
        result = _render(*args)
        assert result.returncode == 2
        assert result.stdout == ""


@pytest.mark.parametrize(
    ("env", "expected"),
    [({}, "0.0.0.0"), ({"DEPLOY_AGENT_BIND_HOST": "127.0.0.1"}, "127.0.0.1")],  # noqa: S104
)
def test_the_health_bind_defaults_to_every_interface(
    env: dict[str, str], expected: str
) -> None:
    """The .201 and .202 units set no bind host and keep what they had; the
    LaunchAgent's 127.0.0.1 is honoured. Read in a fresh interpreter, because
    the module reads it once at import."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import deploy_agent.agent as a; print(a.HEALTH_BIND_HOST)",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        env={
            **{k: v for k, v in os.environ.items() if k != "DEPLOY_AGENT_BIND_HOST"},
            **env,
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected
