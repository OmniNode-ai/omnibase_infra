# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19543 -- a deploy-agent instance is data, and dev-200 is one.

AC1 (executor half): each non-default instance's dev-lane composition is read
from its ``lane:`` block in ``config/deploy_lane_routing.yaml``; a fixture table
naming ``dev-999`` builds a composition with no Python edit, and dev-202's
composition is exactly the one OMN-19522 wrote as a literal.

AC3/AC4 (repo half): the dev-200 instance on the .200 host, routed nothing; its
composition agrees with docker/docker-compose.dev-200.yml; its LaunchAgent,
rendered from the one template, carries the dev-202 unit's environment name for
name, a health port in its block bound to loopback, and its own clone.
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
import yaml
from deploy_agent.events import DEV_LANE_GATEWAY_SERVICES, EnumRuntimeLane
from deploy_agent.executor import (
    _LANE_CONFIGS,
    COMPOSE_FILE,
    DEFAULT_DEV_INSTANCE,
    DEV_INSTANCE_LANE_CONFIGS,
    EnumInstancePhase,
    ModelLaneConfig,
    build_dev_instance_lane_configs,
    lane_config_for,
    select_dev_instance,
)
from deploy_agent.instance_lanes import parse_instance_lanes
from deploy_agent.routing import (
    RoutingTableError,
    load_routing_table,
    parse_routing_table,
    resolve_instance,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TABLE = _REPO_ROOT / "config" / "deploy_lane_routing.yaml"
_DOCKER = _REPO_ROOT / "docker"
_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_LAUNCHD = _DEPLOY_DIR / "launchd"
_RENDER = _LAUNCHD / "render-deploy-agent-plist.sh"
_DEV_202_UNIT = _DEPLOY_DIR / "deploy-agent-dev-202.service"
_DEV_200_OVERLAY = _DOCKER / "docker-compose.dev-200.yml"

#: dev-200's port block (docker-compose.dev-200.yml header, OMN-19543 AC3).
_DEV_200_BLOCK = range(42000, 43000)
_DEV_200_AGENT_PORT = 42098


class _ComposeLoader(yaml.SafeLoader):
    """Reads the compose ``!override`` / ``!reset`` tags as plain values."""


def _passthrough(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)  # type: ignore[arg-type]


_ComposeLoader.add_constructor("!override", _passthrough)
_ComposeLoader.add_constructor("!reset", _passthrough)


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
        assert configs[DEFAULT_DEV_INSTANCE] is _LANE_CONFIGS[EnumRuntimeLane.DEV]

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
        specs = parse_instance_lanes(
            _FIXTURE_TABLE.replace("[lab-overlay]", "[lab-overlay, reboot-host]")
        )
        with pytest.raises(RoutingTableError, match="reboot-host"):
            build_dev_instance_lane_configs(specs)

    def test_two_instances_may_not_share_a_receipt_lane(self) -> None:
        text = _TABLE.read_text().replace(
            "receipt_lane: compose-dev-200", "receipt_lane: compose-dev-202"
        )
        with pytest.raises(RoutingTableError, match="same receipt_lane"):
            parse_instance_lanes(text)

    def test_the_router_still_parses_a_table_carrying_lane_blocks(self) -> None:
        """An agent from before OMN-19543 reads this table at a command's ref:
        its parser ignores the lane key, so routing is unchanged."""
        table = parse_routing_table(_FIXTURE_TABLE)
        assert set(table.instances) == {"dev-201", "dev-999"}


# --------------------------------------------------------------------------- #
# dev-200 -- declared, composed, routed nothing                                #
# --------------------------------------------------------------------------- #
class TestDev200Instance:
    def test_dev_200_is_declared_with_its_own_group_and_host(self) -> None:
        table = load_routing_table(_REPO_ROOT)
        instance = table.instances["dev-200"]
        assert instance.consumer_group == "onex-deploy-agent-dev-200"
        # socket.gethostname() on .200 reads Stickybeatz-Studio.local
        # (lane-manifest verified: line 2026-09-23); the router compares the
        # lowercased first label.
        assert (
            resolve_instance(table, env={}, hostname="Stickybeatz-Studio.local")
            is instance
        )
        assert resolve_instance(table, env={"DEPLOY_AGENT_INSTANCE": "dev-200"}) is (
            instance
        )

    def test_dev_200_is_routed_nothing(self) -> None:
        """No route names dev-200, so every command takes the default, and the
        dev-200 agent skips it (a route to a stopped instance would stall every
        omnimarket rebuild)."""
        table = load_routing_table(_REPO_ROOT)
        assert all(r.instance != "dev-200" for r in table.routes)
        for requester in ("gha/omnimarket/123", "gha/omnibase_infra/9", "operator"):
            assert table.route(EnumRuntimeLane.DEV, requester) == "dev-201"

    def test_dev_200_proves_omnimarket_only(self) -> None:
        spec = parse_instance_lanes(_TABLE.read_text())["dev-200"]
        assert spec.receipt_lane == "compose-dev-200"
        assert spec.proves == ("omnimarket",)

    def test_dev_200_composition(self) -> None:
        select_dev_instance("dev-200")
        cfg = lane_config_for(EnumRuntimeLane.DEV)
        assert cfg.compose_project == cfg.build_project == "omnibase-infra-dev-200"
        assert Path(cfg.compose_files[-1]).name == "docker-compose.dev-200.yml"
        assert [p for _, p in cfg.runtime_health_targets] == [42085, 42086]
        assert cfg.disabled_phases == frozenset(EnumInstancePhase)

    def test_dev_200_composition_agrees_with_its_overlay(self) -> None:
        cfg = DEV_INSTANCE_LANE_CONFIGS["dev-200"]
        overlay = yaml.load(_DEV_200_OVERLAY.read_text(), Loader=_ComposeLoader)  # noqa: S506
        services: dict[str, Any] = overlay["services"]
        assert overlay["name"] == cfg.compose_project
        assert services["postgres"]["container_name"] == cfg.postgres_container
        assert (
            services["omninode-runtime"]["container_name"] == cfg.main_runtime_container
        )
        for service, port in cfg.runtime_health_targets:
            published = [str(p).split(":")[-2] for p in services[service]["ports"]]
            assert str(port) in published, (service, port, published)
        for service in cfg.disabled_services - set(DEV_LANE_GATEWAY_SERVICES):
            assert services[service]["profiles"] == ["dev-200-disabled"], service


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


def _rendered_dev_200() -> dict[str, Any]:
    result = _render(
        "dev-200", str(_DEV_200_AGENT_PORT), "/Users/labuser/onex-lanes/dev-200"
    )
    assert result.returncode == 0, result.stderr
    assert "@" not in result.stdout.split("-->", 1)[1], "an unrendered @TOKEN@"
    return plistlib.loads(result.stdout.encode())


class TestLaunchAgent:
    def test_the_plist_names_its_instance_and_its_own_clone(self) -> None:
        plist = _rendered_dev_200()
        env = plist["EnvironmentVariables"]
        root = "/Users/labuser/onex-lanes/dev-200"
        assert plist["Label"] == "ai.omninode.deploy-agent-dev-200"
        assert env["DEPLOY_AGENT_INSTANCE"] == "dev-200"
        assert env["DEPLOY_AGENT_DIR"] == f"{root}/omnibase_infra/scripts/deploy-agent"
        assert env["DEPLOY_AGENT_PYTHON"].startswith(f"{root}/omnibase_infra/")
        assert env["DEPLOY_AGENT_REPO_DIR"] == f"{root}/omni_home/omnibase_infra"
        assert env["OMNI_HOME"] == f"{root}/omni_home"
        assert env["DEPLOY_AGENT_ENV_FILE"] == "/Users/labuser/.omnibase/dev-200.env"
        assert env["DEPLOY_AGENT_STATE_DIR"].endswith("/jobs-dev-200")
        assert plist["ProgramArguments"][-1].endswith(
            "deploy/launchd/deploy-agent-launchd.sh"
        )
        assert plist["KeepAlive"] == {"SuccessfulExit": False}

    def test_the_plist_carries_the_dev_202_unit_environment_name_for_name(
        self,
    ) -> None:
        env = _rendered_dev_200()["EnvironmentVariables"]
        unit = _unit_env_names(_DEV_202_UNIT)
        assert sorted(unit - set(env)) == []
        # What the plist adds, each for a stated reason in the template.
        assert set(env) - unit == {
            "HOME",
            "PATH",
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
        env = _rendered_dev_200()["EnvironmentVariables"]
        assert sorted(set(env) - set(env["DEPLOY_AGENT_ENV_PROTECTED"].split())) == []

    def test_the_health_port_is_in_the_block_on_loopback_and_unpublished(
        self,
    ) -> None:
        env = _rendered_dev_200()["EnvironmentVariables"]
        assert int(env["DEPLOY_AGENT_PORT"]) in _DEV_200_BLOCK
        assert env["DEPLOY_AGENT_BIND_HOST"] == "127.0.0.1"
        assert f":{_DEV_200_AGENT_PORT}:" not in _DEV_200_OVERLAY.read_text()

    @pytest.mark.parametrize(
        "args",
        [
            ("dev-200", "42098"),
            ("dev_200", "42098", "/x"),
            ("dev-200", "61098", "/x"),
            ("dev-200", "80", "/x"),
            ("dev-200", "42098", "relative/root"),
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
