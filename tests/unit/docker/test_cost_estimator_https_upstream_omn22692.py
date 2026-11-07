# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22692: the dev lane keeps the loopback upstream of the tailnet https front alive.

The https listener node (node_tailnet_https_serve_effect) fronts a loopback upstream only; the dev
estimator's overlay names loopback port 3080. OMN-23011 moved the published 3080 off every interface onto
the host's tailnet address, so nothing listened on loopback 3080 any more and the https name answered 502.

AC1 -- the dev-lane page service publishes 3080 on loopback as well as on the declared bind address, and
on loopback alone when the bind address is unset (the same binding twice is one).

AC2 -- no published address is every interface, so OMN-23011's tailnet-only rule stands.

The page service is rendered alone with ``docker compose config`` (a non-mutating render), as the
OMN-23011 binding tests do.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_DEV_LANE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.dev-lane.yml"
)
_SERVICE = "cost-estimator-web"
_BIND_VAR = "COST_ESTIMATOR_BIND_ADDR"
_LOOPBACK = "127.0.0.1"
_ALL_INTERFACES = ".".join(("0",) * 4)
# RFC 5737 documentation address: never a real host.
_DOC_ADDR = "192.0.2.1"
_COMPOSE_TAG = re.compile(r"!(?:override|reset|merge)(?=\s)")


def _render_ports(
    tmp_path: Path, env_values: dict[str, str]
) -> list[tuple[str, str, int]]:
    document: dict[str, Any] = yaml.safe_load(
        _COMPOSE_TAG.sub("", _DEV_LANE.read_text(encoding="utf-8"))
    )
    page = dict(document["services"][_SERVICE])
    for key in ("networks", "logging", "profiles"):
        page.pop(key, None)
    path = tmp_path / "page.yml"
    path.write_text(yaml.safe_dump({"services": {_SERVICE: page}}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("COST_ESTIMATOR_")}
    env.update(env_values)
    done = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            os.devnull,
            "-f",
            str(path),
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
    published = json.loads(done.stdout)["services"][_SERVICE]["ports"]
    return [(p["host_ip"], p["published"], p["target"]) for p in published]


def test_ac1_loopback_3080_is_published_beside_the_tailnet_address(
    tmp_path: Path,
) -> None:
    ports = _render_ports(tmp_path, {_BIND_VAR: _DOC_ADDR})
    assert (_LOOPBACK, "3080", 3000) in ports
    assert (_DOC_ADDR, "3080", 3000) in ports


def test_ac1_unset_bind_address_publishes_loopback_once(tmp_path: Path) -> None:
    assert _render_ports(tmp_path, {}) == [(_LOOPBACK, "3080", 3000)]


def test_ac2_no_published_address_is_every_interface(tmp_path: Path) -> None:
    for env_values in ({}, {_BIND_VAR: _DOC_ADDR}):
        ports = _render_ports(tmp_path, env_values)
        assert all(host_ip != _ALL_INTERFACES for host_ip, _, _ in ports)
        assert {host_ip for host_ip, _, _ in ports} <= {_LOOPBACK, _DOC_ADDR}
