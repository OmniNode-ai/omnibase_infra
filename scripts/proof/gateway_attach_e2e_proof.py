# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""OMN-15753 -- attach ingress end-to-end slice proof (skeleton).

Chain proven: **attach (client-credentials token) -> ACTIVE session ->
heartbeat -> operator disables the tenant's Keycloak client -> next heartbeat
observes revocation -> session torn down**, against
``node_gateway_attach_effect`` (OMN-15750) once it is deployed to onex-dev
(OMN-15754).

Status of this file
--------------------
This is a **skeleton**, matching the convention set by
``scripts/proof/e2e_cloud_workflow_harness.py`` (OMN-10858): every stage
function names exactly which topic/endpoint it asserts against, but the live
bodies raise ``StageNotImplementedError`` until run against a deployed node.
The handler-level equivalent of every stage below is already proven at the
unit level in
``tests/unit/nodes/node_gateway_attach_effect/test_handlers.py`` (see
``test_heartbeat_after_keycloak_revocation_tears_down_session``) -- what this
harness adds is the live bus + live Keycloak round trip, which requires the
node to actually be running somewhere reachable.

Safety posture
---------------
* ``--live`` defaults OFF. With no flags this program prints the stage plan
  and exits 0 -- no network, no Kafka, no Keycloak admin call.
* ``--live`` requires every value below explicitly; nothing defaults or
  falls back silently.
* This never targets prod. onex-dev only, and only via the tenant's own
  client-credentials token -- this harness never uses a Keycloak *admin*
  credential except for the one explicit ``--revoke`` step, which is opt-in
  and logs exactly what it is about to disable before doing it.

Usage
-----
::

    # dry run -- safe anywhere, and the default
    uv run python scripts/proof/gateway_attach_e2e_proof.py

    # live run against onex-dev, house tenant
    uv run python scripts/proof/gateway_attach_e2e_proof.py --live \\
        --edge-instance-id 201-house-tenant-test-lane \\
        --bootstrap-servers <onex-dev redpanda bootstrap> \\
        --attach-request-topic onex.cmd.omnibase-infra.gateway-attach-request.v1 \\
        --heartbeat-request-topic onex.cmd.omnibase-infra.gateway-heartbeat-request.v1 \\
        --detach-request-topic onex.cmd.omnibase-infra.gateway-detach-request.v1 \\
        --session-event-topic onex.evt.omnibase-infra.gateway-session.v1 \\
        --revoke
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime


class StageNotImplementedError(NotImplementedError):
    """Raised by a live stage body that has not been wired to a real transport yet."""


@dataclass(frozen=True)
class HarnessConfig:
    edge_instance_id: str
    bootstrap_servers: str
    attach_request_topic: str
    heartbeat_request_topic: str
    detach_request_topic: str
    session_event_topic: str
    revoke: bool


STAGE_SURFACES: tuple[tuple[str, str], ...] = (
    (
        "attach",
        "publish ModelGatewayAttachRequest (client-credentials token) to "
        "{attach_request_topic}; read back ModelGatewayAttachResponse from "
        "{session_event_topic}, assert status == ACTIVE",
    ),
    (
        "heartbeat_active",
        "publish ModelGatewayHeartbeatRequest to {heartbeat_request_topic}; "
        "assert session_event.event_type == HEARTBEAT_OK, "
        "termination_reason is None",
    ),
    (
        "revoke",
        "Keycloak Admin API: disable the tenant's confidential client "
        "(enabled=false) -- the sole admin-credential call in this harness; "
        "opt-in via --revoke",
    ),
    (
        "heartbeat_after_revoke",
        "publish another ModelGatewayHeartbeatRequest to "
        "{heartbeat_request_topic}; assert session_event.event_type == "
        "REVOKED, termination_reason == REVOKED, and that the session no "
        "longer answers a "
        "subsequent detach with anything but SessionNotFoundError",
    ),
)


def _print_plan(config: HarnessConfig) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] gateway attach e2e proof -- PLAN ONLY")
    for name, template in STAGE_SURFACES:
        if name == "revoke" and not config.revoke:
            print(f"  - {name}: SKIPPED (--revoke not set)")
            continue
        surface = template.format(
            attach_request_topic=config.attach_request_topic,
            heartbeat_request_topic=config.heartbeat_request_topic,
            session_event_topic=config.session_event_topic,
        )
        print(f"  - {name}: {surface}")


def stage_attach(config: HarnessConfig) -> None:
    raise StageNotImplementedError(
        "attach: requires a live Kafka producer against "
        f"{config.bootstrap_servers} / {config.attach_request_topic} -- not "
        "wired until node_gateway_attach_effect is deployed and reachable "
        "(OMN-15754)."
    )


def stage_heartbeat(config: HarnessConfig, *, expect_revoked: bool) -> None:
    raise StageNotImplementedError(
        f"heartbeat(expect_revoked={expect_revoked}): requires a live Kafka "
        f"round trip against {config.heartbeat_request_topic} / "
        f"{config.session_event_topic}."
    )


def stage_revoke(config: HarnessConfig) -> None:
    raise StageNotImplementedError(
        "revoke: requires a live Keycloak Admin API call disabling the "
        "tenant's confidential client -- deliberately not implemented in "
        "this skeleton pass; the admin credential this needs is a distinct, "
        "higher-privilege ref from the tenant's own attach credential."
    )


def run_live(config: HarnessConfig) -> int:
    stage_attach(config)
    stage_heartbeat(config, expect_revoked=False)
    if config.revoke:
        stage_revoke(config)
        stage_heartbeat(config, expect_revoked=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", default=False)
    parser.add_argument("--edge-instance-id", default=None)
    parser.add_argument("--bootstrap-servers", default=None)
    parser.add_argument("--attach-request-topic", default=None)
    parser.add_argument("--heartbeat-request-topic", default=None)
    parser.add_argument("--detach-request-topic", default=None)
    parser.add_argument("--session-event-topic", default=None)
    parser.add_argument("--revoke", action="store_true", default=False)
    args = parser.parse_args(argv)

    if not args.live:
        config = HarnessConfig(
            edge_instance_id=args.edge_instance_id or "<dry-run>",
            bootstrap_servers=args.bootstrap_servers or "<dry-run>",
            attach_request_topic=args.attach_request_topic
            or "onex.cmd.omnibase-infra.gateway-attach-request.v1",
            heartbeat_request_topic=args.heartbeat_request_topic
            or "onex.cmd.omnibase-infra.gateway-heartbeat-request.v1",
            detach_request_topic=args.detach_request_topic
            or "onex.cmd.omnibase-infra.gateway-detach-request.v1",
            session_event_topic=args.session_event_topic
            or "onex.evt.omnibase-infra.gateway-session.v1",
            revoke=args.revoke,
        )
        _print_plan(config)
        return 0

    missing = [
        name
        for name, value in (
            ("--edge-instance-id", args.edge_instance_id),
            ("--bootstrap-servers", args.bootstrap_servers),
            ("--attach-request-topic", args.attach_request_topic),
            ("--heartbeat-request-topic", args.heartbeat_request_topic),
            ("--detach-request-topic", args.detach_request_topic),
            ("--session-event-topic", args.session_event_topic),
        )
        if not value
    ]
    if missing:
        print(f"--live requires all of: {', '.join(missing)}", file=sys.stderr)
        return 2

    config = HarnessConfig(
        edge_instance_id=args.edge_instance_id,
        bootstrap_servers=args.bootstrap_servers,
        attach_request_topic=args.attach_request_topic,
        heartbeat_request_topic=args.heartbeat_request_topic,
        detach_request_topic=args.detach_request_topic,
        session_event_topic=args.session_event_topic,
        revoke=args.revoke,
    )
    return run_live(config)


if __name__ == "__main__":
    raise SystemExit(main())
