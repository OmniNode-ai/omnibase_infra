# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease CLI: a client of ``node_lease_orchestrator`` over the bus (OMN-21424).

``acquire``, ``renew`` and ``release`` publish their command on the orchestrator's
command topic of the named lane and print the orchestrator's answer from its event
topics; only the orchestrator's effect writes etcd. ``show`` is a read of the
store with no command behind it, so it reads etcd directly and writes nothing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import click
from pydantic import BaseModel

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.lease.lease_bus_client import (
    LeaseReplyTimeoutError,
    request_lease,
)
from omnibase_infra.handlers.lease.lease_etcd_gateway import LeaseEtcdGateway
from omnibase_infra.handlers.lease.lease_store import (
    LeaseStore,
    lease_kinds_from_spec,
)
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
    ModelLeaseReply,
)


@dataclass(frozen=True)
class LeaseCliConfig:
    lane: str
    workspace_root: Path | None
    reply_timeout_s: float

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure transport: the CLI publishes commands on the lane's bus."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Every command is a publish and a wait for the orchestrator's answer."""
        return EnumHandlerTypeCategory.EFFECT

    def require_lane(self) -> str:
        """The lane a command is published to; a command without one is refused."""
        if not self.lane.strip():
            raise click.UsageError(
                "name the lane whose bus carries the lease commands: "
                "--lane or ONEX_LEASE_LANE"
            )
        return self.lane.strip()


@click.group(name="lease")
@click.option(
    "--lane",
    envvar="ONEX_LEASE_LANE",
    help="Lane whose bus carries the lease commands (acquire, renew, release).",
)
@click.option(
    "--omnibase-path",
    envvar="OMNIBASE_PATH",
    type=click.Path(path_type=Path),
    help="Workspace root holding the lane declaration. Bound to $OMNIBASE_PATH.",
)
@click.option(
    "--reply-timeout-s",
    type=click.FloatRange(min=0.1),
    default=30.0,
    show_default=True,
    help="How long to wait for the orchestrator's answer.",
)
@click.pass_context
def lease_group(
    ctx: click.Context,
    lane: str | None,
    omnibase_path: Path | None,
    reply_timeout_s: float,
) -> None:
    """Acquire, renew and release generic leases through the lease orchestrator."""
    ctx.obj = LeaseCliConfig(
        lane=lane or "", workspace_root=omnibase_path, reply_timeout_s=reply_timeout_s
    )


def _print(replies: Sequence[ModelLeaseReply]) -> None:
    for reply in replies:
        click.echo(json.dumps(reply.model_dump(mode="json"), sort_keys=True))
    if any(reply.outcome in {"lease-refused", "lease-expired"} for reply in replies):
        raise click.exceptions.Exit(3)


def _command(request: BaseModel) -> None:
    """Publish one command to the orchestrator and print its answer."""
    from omnibase_infra.handlers.lease import lease_lane_bus

    config = click.get_current_context().find_object(LeaseCliConfig)
    if config is None:
        raise click.UsageError("run through the lease command group")
    lane = config.require_lane()

    async def execute() -> ModelLeaseReply:
        async with lease_lane_bus.open_lane_bus(lane, config.workspace_root) as bus:
            return await request_lease(bus, request, timeout_s=config.reply_timeout_s)

    try:
        reply = asyncio.run(execute())
    except (LeaseReplyTimeoutError, lease_lane_bus.LeaseLaneUnresolvedError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print((reply,))


@lease_group.command()
@click.option("--kind", required=True)
@click.option("--resource", required=True)
@click.option("--holder", required=True)
@click.option("--request-id", type=click.UUID, default=lambda: str(uuid4()))
@click.option("--ttl-s", type=click.IntRange(min=2), default=None)
def acquire(
    kind: str,
    resource: str,
    holder: str,
    request_id: UUID,
    ttl_s: int | None,
) -> None:
    """Grant a free slot, or refuse when all slots are held."""
    _command(
        ModelLeaseAcquireRequest(
            kind=kind,
            resource=resource,
            holder=holder,
            request_id=request_id,
            ttl_s=ttl_s,
        )
    )


@lease_group.command()
@click.option("--kind", required=True)
@click.option("--resource", required=True)
@click.option("--lease-id", type=click.UUID, required=True)
def renew(kind: str, resource: str, lease_id: UUID) -> None:
    """Keep a grant alive at the same fence."""
    _command(ModelLeaseRenewRequest(kind=kind, resource=resource, lease_id=lease_id))


@lease_group.command()
@click.option("--kind", required=True)
@click.option("--resource", required=True)
@click.option("--lease-id", type=click.UUID, required=True)
def release(kind: str, resource: str, lease_id: UUID) -> None:
    """Give a grant back."""
    _command(ModelLeaseReleaseRequest(kind=kind, resource=resource, lease_id=lease_id))


@lease_group.command()
@click.option("--kind", required=True)
@click.option("--resource", required=True)
@click.option(
    "--etcd", envvar="ONEX_LEASE_ETCD_ENDPOINT", default="", help="etcd v3 client URL."
)
@click.option(
    "--kinds",
    envvar="ONEX_LEASE_KINDS",
    default="",
    help="Comma list: kind:capacity:ttl_s.",
)
def show(kind: str, resource: str, etcd: str, kinds: str) -> None:
    """Print one JSON line for every live grant of a resource.

    A read of the store with no command behind it: it reads etcd directly and
    writes nothing.
    """
    if not etcd.strip():
        raise click.UsageError("ONEX_LEASE_ETCD_ENDPOINT must not be empty")
    if not kinds.strip():
        raise click.UsageError("ONEX_LEASE_KINDS must not be empty")
    try:
        policies = lease_kinds_from_spec(kinds)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if not any(policy.kind == kind for policy in policies):
        raise click.UsageError(f"lease kind {kind!r} is not declared")

    async def execute() -> Sequence[ModelLeaseReply]:
        handler = await LeaseEtcdGateway.connect(etcd.strip(), 5.0)
        try:
            return await LeaseStore(handler, policies).show(kind, resource)
        finally:
            await handler.aclose()

    _print(asyncio.run(execute()))


__all__ = ["LeaseCliConfig", "lease_group"]
