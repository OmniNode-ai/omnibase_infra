# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pre-flight locality guard for prod-lane ``node_redeploy_orchestrator``
dispatch (OMN-15181, dev-class fix 1 of the OMN-13418 gated prod redeploy).

## Why this exists

The 2026-07-26 gated prod redeploy dispatch (grant-6dbeae94, OMN-13418) ran
via ``onex skill redeploy --lane prod ...`` from this Mac and HARD-FAILED
closed after two identical ``Kafka flush timed out`` errors. Live-verified
root cause (read-only ssh probes, ledger 2026-07-26T09:05Z / 09:25Z): the
prod redpanda broker's external ``advertised_kafka_api`` listener is a raw
LAN address (``192.168.86.201:49092``) that only routes from clients
physically on that LAN segment -- ``nc -zv 192.168.86.201 49092`` succeeds
when run *on* ``omninode-pc`` itself and fails from this Mac / ``.200`` over
Tailscale-only routes. The bootstrap metadata fetch (via MagicDNS) succeeds
regardless, which is what makes the failure mode a silent produce-path
leader-reconnect timeout rather than an upfront connection error -- the
caller gets no signal until the flush times out.

## What this guards

Only the ``node_redeploy_orchestrator`` dispatch (the ``redeploy`` skill)
with ``runtime_lane == "prod"``. Dev and stability-test lane dispatches are
unaffected -- their brokers do not carry this raw-LAN-only listener
constraint -- so this never adds friction to the everyday dev/stability
path, only to the one lane proven to require on-box locality.

This is intentionally a CLIENT-SIDE, no-network check (``socket.gethostname()``
against a canonical constant) -- it fails fast with an actionable message
*before* the dispatch ever reaches the runtime and burns the flush-timeout
budget, rather than requiring the operator to rediscover this by watching
a second identical failure (feedback_transient_is_bullshit: a repeat of the
same failure is a bug, not a flake -- this guard makes the bug impossible to
repeat).

## What this does NOT do

It does not touch prod redpanda configuration, does not retag/restart any
prod container, and does not touch the promotion-grant resolve/gate chain
(``node_prod_promotion_grant_resolver_effect`` remains a separate, still-open
gap -- OMN-15181 finding 2 -- that this guard does not and cannot fix).
"""

from __future__ import annotations

import socket
from collections.abc import Mapping

import click

__all__ = ["REQUIRED_PROD_DISPATCH_HOSTNAME", "enforce_prod_dispatch_locality"]

# Live-verified 2026-07-26 (ssh omni-201-ts): `hostname` / `hostname -f` /
# `uname -n` all resolve to this literal on the box that carries the prod
# redpanda LAN-only advertised listener. Matches the literal already used
# for `ONEX_BOX_ID` across the prod/stability-test/judge compose files and
# their tests (e.g. docker-compose.prod.yml, test_stability_test_runtime_lane.py).
REQUIRED_PROD_DISPATCH_HOSTNAME = "omninode-pc"

# The one node/lane combination this guard applies to. Declarative here (not
# in skill_mapping.yaml) because the guard is a locality precondition on the
# *client process*, not a node-input payload field.
_GUARDED_NODE_NAME = "node_redeploy_orchestrator"
_GUARDED_LANE_VALUE = "prod"


def enforce_prod_dispatch_locality(
    node_name: str, payload: Mapping[str, object]
) -> None:
    """Fail closed unless a prod-lane redeploy dispatch runs on omninode-pc.

    No-op for every other node and every non-prod ``runtime_lane`` value --
    this must never add friction to dev/stability-test dispatch.

    Raises:
        click.ClickException: when a prod-lane ``node_redeploy_orchestrator``
            dispatch is attempted from a host other than
            :data:`REQUIRED_PROD_DISPATCH_HOSTNAME`.
    """
    if node_name != _GUARDED_NODE_NAME:
        return

    lane = payload.get("runtime_lane")
    if not isinstance(lane, str) or lane.lower() != _GUARDED_LANE_VALUE:
        return

    current_host = socket.gethostname()
    if current_host == REQUIRED_PROD_DISPATCH_HOSTNAME:
        return

    raise click.ClickException(
        "Refusing prod-lane 'redeploy' dispatch from host "
        f"{current_host!r}: prod redpanda's advertised external listener "
        "is a raw-LAN address reachable only from "
        f"{REQUIRED_PROD_DISPATCH_HOSTNAME!r} itself (OMN-15181). Run this "
        "dispatch physically on that box, e.g.:\n"
        '  ssh omni-201-ts "cd <omnibase_infra checkout> && '
        'uv run onex skill redeploy --lane prod ..."\n'
        "Off-box dispatch silently reaches Kafka bootstrap metadata (via "
        "MagicDNS) but then hangs on the produce-path leader reconnect "
        "until the flush timeout fires -- this guard fails fast instead."
    )
