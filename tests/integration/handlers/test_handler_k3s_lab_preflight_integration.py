# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: the k3s lab preflight against this host's real interfaces (OMN-24956)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_k3s_lab_preflight import (
    HandlerK3sLabPreflight,
    live_interface_for,
)
from omnibase_infra.models.model_k3s_lab_preflight_request import (
    ModelK3sLabPreflightRequest,
)

pytestmark = pytest.mark.integration

CONFIG = Path(__file__).resolve().parents[3] / "deploy/lab/k3s/k3s-server-config.yaml"


def _live_pair() -> tuple[str, str]:
    """A (address, interface) pair of this host's live IPv4 interfaces."""
    try:
        table = subprocess.run(
            ["ip", "-br", "-4", "addr"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("ip -br addr is unavailable on this host")
    for line in table.splitlines():
        columns = line.split()
        if len(columns) >= 3 and columns[1] in {"UP", "UNKNOWN"}:
            address = columns[2].split("/", 1)[0]
            return address, live_interface_for(address, table)
    pytest.skip("no live IPv4 interface on this host")


def _render(node_ip: str, iface: str) -> str:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["node-ip"] = node_ip
    config["advertise-address"] = node_ip
    config["flannel-iface"] = iface
    return yaml.safe_dump(config, sort_keys=False)


def test_k3s_node_ip_live_host_interface_passes() -> None:
    address, iface = _live_pair()
    result = HandlerK3sLabPreflight().handle(
        ModelK3sLabPreflightRequest(config_text=_render(address, iface))
    )
    assert result.verdict == "pass"
    assert result.live_interface == iface


def test_k3s_node_ip_absent_from_live_host_refuses() -> None:
    _, iface = _live_pair()
    result = HandlerK3sLabPreflight().handle(
        ModelK3sLabPreflightRequest(config_text=_render("192.0.2.254", iface))
    )
    assert result.verdict == "refuse"
    assert result.reason == "node-ip-not-live"
