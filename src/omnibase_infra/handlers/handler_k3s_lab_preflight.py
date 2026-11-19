# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Refuse to start the lab k3s server on an unpinned or dead node-ip (OMN-24956).

WHY
---
The retired lab k3s ran 188 days with no node-ip pin: it registered against
whatever NIC was live at first registration, that NIC later went down, the node
never re-registered, and CoreDNS and every service-name lookup kept answering
with the dead address. A pin nobody checks is the same failure on a delay.

WHAT THIS DOES
--------------
:func:`check_k3s_server_config` walks a config in one fixed order and returns a
:class:`ModelK3sLabPreflightResult`: the YAML must parse to a mapping,
``node-ip`` and ``flannel-iface`` must be set, the placeholders must already be
rendered when the caller asks for the rendered check, and the pinned node-ip
must be held by a live interface of the ``ip -br addr`` table whose name is
exactly the pinned ``flannel-iface``. ``verdict: refuse`` means the caller must
not start the server; ``reason`` names the failure and ``detail`` the address
or interface.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_k3s_lab_preflight_request import (
    ModelK3sLabPreflightRequest,
)
from omnibase_infra.models.model_k3s_lab_preflight_result import (
    ModelK3sLabPreflightResult,
)

#: An unrendered placeholder of the committed template (``__NODE_IP__``).
PLACEHOLDER_PATTERN = re.compile(r"^__[A-Z0-9_]+__$")

#: Interface states of ``ip -br addr`` that count as live for the node-ip.
LIVE_STATES = frozenset({"UP", "UNKNOWN"})


def live_interface_for(address: str, interface_table: str) -> str:
    """Return the name of the live interface holding ``address``, else ``""``.

    Parses ``ip -br addr`` lines (columns NAME STATE ADDR/PREFIX ...). Only
    lines whose STATE is ``UP`` or ``UNKNOWN`` count as live; a DOWN interface
    holding the address is the retired cluster's failure mode and never counts.
    The ``@...`` suffix of virtual interface names is stripped.
    """
    for line in interface_table.splitlines():
        columns = line.split()
        if len(columns) < 3:
            continue
        name, state = columns[0], columns[1]
        if state not in LIVE_STATES:
            continue
        for column in columns[2:]:
            if column.split("/", 1)[0] == address:
                return name.split("@", 1)[0]
    return ""


def _read_ip_br_addr() -> str:
    """Read the host's ``ip -br addr`` table (the live NIC states and addresses)."""
    result = subprocess.run(
        ["ip", "-br", "addr"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def check_k3s_server_config(
    request: ModelK3sLabPreflightRequest,
    read_interfaces: Callable[[], str] | None = None,
) -> ModelK3sLabPreflightResult:
    """Judge a k3s server config; return ``pass`` or ``refuse`` with a reason."""
    try:
        config: object = yaml.safe_load(request.config_text)
    except yaml.YAMLError as exc:
        return ModelK3sLabPreflightResult(
            verdict="refuse",
            reason="config-unparseable",
            detail=str(exc),
        )
    if not isinstance(config, dict):
        return ModelK3sLabPreflightResult(
            verdict="refuse",
            reason="config-unparseable",
            detail=f"config is {type(config).__name__}, not a mapping",
        )

    raw_node_ip = config.get("node-ip")
    if raw_node_ip is None or str(raw_node_ip).strip() == "":
        return ModelK3sLabPreflightResult(
            verdict="refuse",
            reason="node-ip-unset",
            detail="node-ip is missing or blank; the server would register "
            "against whatever NIC k3s auto-detects",
        )
    node_ip = str(raw_node_ip).strip()

    raw_flannel_iface = config.get("flannel-iface")
    if raw_flannel_iface is None or str(raw_flannel_iface).strip() == "":
        return ModelK3sLabPreflightResult(
            verdict="refuse",
            reason="flannel-iface-unset",
            detail=f"flannel-iface is missing or blank while node-ip is {node_ip}",
            node_ip=node_ip,
        )
    flannel_iface = str(raw_flannel_iface).strip()

    if not request.require_rendered:
        return ModelK3sLabPreflightResult(
            verdict="pass",
            node_ip=node_ip,
            flannel_iface=flannel_iface,
        )

    if PLACEHOLDER_PATTERN.match(node_ip) or PLACEHOLDER_PATTERN.match(flannel_iface):
        return ModelK3sLabPreflightResult(
            verdict="refuse",
            reason="node-ip-unrendered",
            detail=f"placeholder left in the config: node-ip {node_ip}, "
            f"flannel-iface {flannel_iface}",
            node_ip=node_ip,
            flannel_iface=flannel_iface,
        )

    read = read_interfaces or _read_ip_br_addr
    table = request.interface_table or read()
    live_interface = live_interface_for(node_ip, table)
    if live_interface == "":
        return ModelK3sLabPreflightResult(
            verdict="refuse",
            reason="node-ip-not-live",
            detail=f"no live (UP) interface holds node-ip {node_ip}",
            node_ip=node_ip,
            flannel_iface=flannel_iface,
        )

    if live_interface != flannel_iface:
        return ModelK3sLabPreflightResult(
            verdict="refuse",
            reason="flannel-iface-mismatch",
            detail=f"node-ip {node_ip} is held by {live_interface}, "
            f"not the pinned flannel-iface {flannel_iface}",
            node_ip=node_ip,
            flannel_iface=flannel_iface,
            live_interface=live_interface,
        )

    return ModelK3sLabPreflightResult(
        verdict="pass",
        node_ip=node_ip,
        flannel_iface=flannel_iface,
        live_interface=live_interface,
    )


class HandlerK3sLabPreflight:
    """The node's single entry point; the lab start path calls :func:`check_k3s_server_config`."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelK3sLabPreflightRequest
    ) -> ModelK3sLabPreflightResult:
        """Judge the k3s server config of ``request`` before the start."""
        return check_k3s_server_config(request)
