# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23011: the estimator's tailnet gate source reaches the lab container, and port 3080 binds one address.

AC1 -- the dev-lane compose and the report-host compose pass ESTIMATOR_GATE_OPEN_SOURCES to the page
container from a declared lane variable, and the deploy agent accepts the variable in a lane env fragment.

AC2 -- the dev lane publishes port 3080 on the host's tailnet address, named by the lane variable
COST_ESTIMATOR_BIND_ADDR (the host's own env file carries the address), and on no other address: unset it
binds loopback, never every interface.

The dev-lane file needs dozens of unrelated required values to render whole, so the page service and the
top-level declaration it merges are rendered alone with ``docker compose config`` (a non-mutating render);
the service text and the declaration are exactly the file's own.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_DEV_LANE = _REPO / "docker" / "docker-compose.dev-lane.yml"
_REPORT_HOST = _REPO / "docker" / "docker-compose.cost-estimator-web.yml"
_INSTANCE_LANES = (
    _REPO / "scripts" / "deploy-agent" / "deploy_agent" / "instance_lanes.py"
)

_SERVICE = "cost-estimator-web"
_VAR = "ESTIMATOR_GATE_OPEN_SOURCES"
_BIND_VAR = "COST_ESTIMATOR_BIND_ADDR"
_DECLARATION = "x-dev-lane-estimator-gate-env"
# RFC 5737 and RFC 3849 documentation ranges: a stand-in for the lane's own value.
_LANE_RANGES = "198.51.100.0/24,2001:db8::/32"
_ALL_INTERFACES = ".".join(("0",) * 4)
# RFC 5737 documentation addresses: never a real host.
_DOC_ADDR = "192.0.2.1"
_COMPOSE_TAG = re.compile(r"!(?:override|reset|merge)(?=\s)")


def _document(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(_COMPOSE_TAG.sub("", path.read_text(encoding="utf-8")))
    assert isinstance(document, dict)
    return document


def _dev_lane_service_text() -> str:
    """The page service's own block in the dev-lane file, as written."""
    text = _DEV_LANE.read_text(encoding="utf-8")
    start = text.index(f"\n  {_SERVICE}:\n")
    end = text.index("\nvolumes:\n", start)
    return text[start:end]


def _render(compose: Path, env_values: dict[str, str]) -> dict[str, Any]:
    # --env-file os.devnull stops Compose reading a stray .env; the caller's shell can never supply the
    # lane variables under test.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in (_VAR, _BIND_VAR) and not k.startswith("COST_ESTIMATOR_")
    }
    env.update(env_values)
    done = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            os.devnull,
            "-f",
            str(compose),
            "config",
            "--format",
            "json",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)["services"][_SERVICE]


def _dev_lane_page(tmp_path: Path, env_values: dict[str, str]) -> dict[str, Any]:
    """Render the dev lane's page service with its own declaration, alone."""
    document = _document(_DEV_LANE)
    page = dict(document["services"][_SERVICE])
    # Only what the page needs to render by itself: the network and the shared logging anchor are the
    # whole file's, not the page's gate or binding.
    for key in ("networks", "logging", "profiles"):
        page.pop(key, None)
    mini = {"services": {_SERVICE: page}}
    path = tmp_path / "dev-lane-page.yml"
    path.write_text(yaml.safe_dump(mini), encoding="utf-8")
    return _render(path, env_values)


# -- AC1: the passthrough ---------------------------------------------------


def test_ac1_dev_lane_declares_the_lane_variable_once() -> None:
    declared = _document(_DEV_LANE)[_DECLARATION]
    assert declared == {_VAR: f"${{{_VAR}:-}}"}


def test_ac1_dev_lane_page_takes_the_variable_from_the_declaration_not_a_literal() -> (
    None
):
    service_text = _dev_lane_service_text()
    assert "*dev_lane_estimator_gate_env" in service_text
    environment = _document(_DEV_LANE)["services"][_SERVICE]["environment"]
    assert environment[_VAR] == f"${{{_VAR}:-}}"


def test_ac1_dev_lane_page_receives_the_lane_value_and_unset_opens_nothing(
    tmp_path: Path,
) -> None:
    assert _dev_lane_page(tmp_path, {})["environment"][_VAR] == ""
    environment = _dev_lane_page(tmp_path, {_VAR: _LANE_RANGES})["environment"]
    assert environment[_VAR] == _LANE_RANGES


def test_ac1_report_host_page_passes_the_variable_through_and_unset_opens_nothing() -> (
    None
):
    declared = _document(_REPORT_HOST)["services"][_SERVICE]["environment"]
    assert declared[_VAR] == f"${{{_VAR}:-}}"
    env = {
        "COST_ESTIMATOR_WEB_IMAGE": "example/cost-estimator-web:report-host",
        _BIND_VAR: _DOC_ADDR,
        "COST_ESTIMATOR_PORT": "3080",
        "COST_ESTIMATOR_API_BASE_URL": "http://192.0.2.2:8090",
        "COST_ESTIMATOR_AUTH_URL": "https://report.example.invalid",
    }
    assert _render(_REPORT_HOST, env)["environment"][_VAR] == ""
    passed = _render(_REPORT_HOST, {**env, _VAR: _LANE_RANGES})
    assert passed["environment"][_VAR] == _LANE_RANGES


def _schema_keys() -> set[str]:
    tree = ast.parse(_INSTANCE_LANES.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "LANE_ENV_FRAGMENT_SCHEMA_KEYS"
            and node.value is not None
        ):
            return {
                n.value
                for n in ast.walk(node.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
    raise AssertionError("LANE_ENV_FRAGMENT_SCHEMA_KEYS not assigned")


def test_ac1_deploy_agent_accepts_the_variable_in_a_lane_env_fragment() -> None:
    assert _VAR in _schema_keys()


# -- AC2: the binding -------------------------------------------------------


def test_ac2_dev_lane_publishes_3080_on_the_declared_bind_address_only() -> None:
    ports = _document(_DEV_LANE)["services"][_SERVICE]["ports"]
    assert ports == [f"${{{_BIND_VAR}:-127.0.0.1}}:3080:3000"]
    assert _ALL_INTERFACES not in _dev_lane_service_text()
    assert '"3080:3000"' not in _dev_lane_service_text()


def test_ac2_rendered_binding_is_the_lane_variables_address(tmp_path: Path) -> None:
    published = _dev_lane_page(tmp_path, {_BIND_VAR: _DOC_ADDR})["ports"]
    assert [(p["host_ip"], p["published"], p["target"]) for p in published] == [
        (_DOC_ADDR, "3080", 3000)
    ]


def test_ac2_unset_variable_binds_loopback_never_every_interface(
    tmp_path: Path,
) -> None:
    published = _dev_lane_page(tmp_path, {})["ports"]
    assert [(p["host_ip"], p["published"]) for p in published] == [
        ("127.0.0.1", "3080")
    ]
    assert all(p.get("host_ip") != _ALL_INTERFACES for p in published)


# -- the host's address is the host's own, never in this repository ---------


def test_the_repository_names_no_tailnet_host_address() -> None:
    """The address is the host's own and this repository is public: only the range (a /10 block) appears."""
    host_address = re.compile(
        r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b(?!/)"
    )
    for path in (_DEV_LANE, _REPORT_HOST):
        assert host_address.findall(path.read_text(encoding="utf-8")) == [], path.name
