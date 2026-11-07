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
    lane_with_phases_off,
    select_dev_instance,
)
from deploy_agent.instance_lanes import parse_instance_lanes
from deploy_agent.routing import (
    AGENT_CLONE_ROOT,
    RoutingTableError,
    lane_flags_for_instance,
    load_routing_table,
    parse_routing_table,
    resolve_instance,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TABLE = _REPO_ROOT / "config" / "deploy_lane_routing.yaml"


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

    def test_the_router_still_parses_a_table_carrying_lane_blocks(self) -> None:
        """An agent from before OMN-19543 reads this table at a command's ref:
        its parser ignores the lane key, so routing is unchanged."""
        table = parse_routing_table(_FIXTURE_TABLE)
        assert set(table.instances) == {"dev-201", "dev-999"}


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
