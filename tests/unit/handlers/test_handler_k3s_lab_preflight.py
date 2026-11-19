# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the lab k3s server preflight refusal (OMN-24956)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_k3s_lab_preflight import (
    HandlerK3sLabPreflight,
    check_k3s_server_config,
)
from omnibase_infra.models.model_k3s_lab_preflight_request import (
    ModelK3sLabPreflightRequest,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "deploy/lab/k3s/k3s-server-config.yaml"

#: Documentation addresses only (RFC 5737); never a real LAN address.
NODE_IP = "192.0.2.10"
DOWN_IP = "198.51.100.7"
NODE_IFACE = "eno4"
DOWN_IFACE = "eno1"

TABLE = (
    "lo               UNKNOWN        127.0.0.1/8 ::1/128\n"
    "eno1             DOWN           198.51.100.7/24\n"
    "eno4             UP             192.0.2.10/24 fe80::1/64\n"
    "docker0          UP             203.0.113.1/24\n"
)


def committed_config_text() -> str:
    """The committed template's YAML text, placeholders unrendered."""
    return CONFIG.read_text(encoding="utf-8")


def rendered_config_text(node_ip: str = NODE_IP, iface: str = NODE_IFACE) -> str:
    """The committed config with every placeholder filled by ``node_ip``/``iface``."""
    config: dict[str, Any] = yaml.safe_load(committed_config_text())
    config["node-ip"] = node_ip
    config["advertise-address"] = node_ip
    config["flannel-iface"] = iface
    return yaml.safe_dump(config, sort_keys=False)


def config_without(key: str) -> str:
    """The committed config with ``key`` removed entirely."""
    config: dict[str, Any] = yaml.safe_load(committed_config_text())
    del config[key]
    return yaml.safe_dump(config, sort_keys=False)


def test_k3s_committed_config_sets_node_ip() -> None:
    """The committed template sets both keys; the unrendered check passes it."""
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=committed_config_text(),
            interface_table=TABLE,
            require_rendered=False,
        )
    )
    assert result.verdict == "pass"
    assert result.node_ip == "__NODE_IP__"
    assert result.flannel_iface == "__INTERFACE__"


def test_k3s_config_without_node_ip_refuses() -> None:
    """A config whose node-ip key is gone is refused, rendered check or not."""
    text = config_without("node-ip")
    for require_rendered in (False, True):
        result = check_k3s_server_config(
            ModelK3sLabPreflightRequest(
                config_text=text,
                interface_table=TABLE,
                require_rendered=require_rendered,
            )
        )
        assert result.verdict == "refuse"
        assert result.reason == "node-ip-unset"


def test_k3s_config_with_blank_node_ip_refuses() -> None:
    """A blank node-ip is refused the same as a missing one."""
    config: dict[str, Any] = yaml.safe_load(committed_config_text())
    config["node-ip"] = ""
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=yaml.safe_dump(config, sort_keys=False),
            interface_table=TABLE,
            require_rendered=True,
        )
    )
    assert result.verdict == "refuse"
    assert result.reason == "node-ip-unset"


def test_k3s_unrendered_node_ip_refuses() -> None:
    """The committed template still holding its placeholders is refused rendered."""
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=committed_config_text(),
            interface_table=TABLE,
            require_rendered=True,
        )
    )
    assert result.verdict == "refuse"
    assert result.reason == "node-ip-unrendered"


def test_k3s_rendered_node_ip_on_live_interface_passes() -> None:
    """A rendered config whose node-ip eno4 holds passes with eno4 as the iface."""
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=rendered_config_text(node_ip=NODE_IP, iface=NODE_IFACE),
            interface_table=TABLE,
            require_rendered=True,
        )
    )
    assert result.verdict == "pass"
    assert result.node_ip == NODE_IP
    assert result.live_interface == NODE_IFACE


def test_k3s_node_ip_on_down_interface_refuses() -> None:
    """An address held only by a DOWN interface is the retired cluster's failure."""
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=rendered_config_text(node_ip=DOWN_IP, iface=DOWN_IFACE),
            interface_table=TABLE,
            require_rendered=True,
        )
    )
    assert result.verdict == "refuse"
    assert result.reason == "node-ip-not-live"
    assert DOWN_IP in result.detail


def test_k3s_node_ip_absent_refuses() -> None:
    """An address no interface holds at all is refused, naming the address."""
    absent_ip = "192.0.2.99"
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=rendered_config_text(node_ip=absent_ip, iface=NODE_IFACE),
            interface_table=TABLE,
            require_rendered=True,
        )
    )
    assert result.verdict == "refuse"
    assert result.reason == "node-ip-not-live"
    assert absent_ip in result.detail


def test_k3s_node_ip_flannel_mismatch_refuses() -> None:
    """A live node-ip held by an interface other than the pinned one is refused."""
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=rendered_config_text(node_ip=NODE_IP, iface=DOWN_IFACE),
            interface_table=TABLE,
            require_rendered=True,
        )
    )
    assert result.verdict == "refuse"
    assert result.reason == "flannel-iface-mismatch"
    assert result.live_interface == NODE_IFACE


def test_k3s_node_ip_reads_live_table_when_none_given() -> None:
    """An empty interface_table makes the check read ``ip -br addr`` live."""
    result = check_k3s_server_config(
        ModelK3sLabPreflightRequest(
            config_text=rendered_config_text(node_ip=NODE_IP, iface=NODE_IFACE),
            interface_table="",
            require_rendered=True,
        ),
        read_interfaces=lambda: TABLE,
    )
    assert result.verdict == "pass"
    assert result.live_interface == NODE_IFACE


def test_k3s_node_ip_handler_routes_from_contract() -> None:
    """The node contract's handler routing resolves to the working handler."""
    contract_path = (
        REPO_ROOT
        / "src/omnibase_infra/nodes/node_k3s_lab_preflight_effect/contract.yaml"
    )
    contract: dict[str, Any] = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    handler_spec = contract["handler_routing"]["handlers"][0]["handler"]
    module = importlib.import_module(handler_spec["module"])
    handler_cls = getattr(module, handler_spec["name"])
    result = handler_cls().handle(
        ModelK3sLabPreflightRequest(
            config_text=rendered_config_text(node_ip=NODE_IP, iface=NODE_IFACE),
            interface_table=TABLE,
            require_rendered=True,
        )
    )
    assert result.verdict == "pass"
    assert result.live_interface == NODE_IFACE
