# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease CLI talking to etcd directly with the same store the orchestrator uses."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

import click

from omnibase_infra.handlers.lease.handler_lease_store import (
    HandlerLeaseStore,
    lease_kinds_from_spec,
)
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseKindPolicy,
    ModelLeaseReply,
)


@dataclass(frozen=True)
class LeaseCliConfig:
    etcd: str
    kinds: str

    def resolve(self) -> tuple[str, tuple[ModelLeaseKindPolicy, ...]]:
        """Validate the endpoint and kinds when a command runs, not at ``--help``."""
        if not self.etcd.strip():
            raise click.UsageError("ONEX_LEASE_ETCD_ENDPOINT must not be empty")
        if not self.kinds.strip():
            raise click.UsageError("ONEX_LEASE_KINDS must not be empty")
        try:
            return self.etcd.strip(), lease_kinds_from_spec(self.kinds)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc


@click.group(name="lease")
@click.option("--etcd", envvar="ONEX_LEASE_ETCD_ENDPOINT", help="etcd v3 client URL.")
@click.option(
    "--kinds", envvar="ONEX_LEASE_KINDS", help="Comma list: kind:capacity:ttl_s."
)
@click.pass_context
def lease_group(ctx: click.Context, etcd: str | None, kinds: str | None) -> None:
    """Acquire, renew, release and show generic leases."""
    ctx.obj = LeaseCliConfig(etcd=etcd or "", kinds=kinds or "")


async def _single(reply: Awaitable[ModelLeaseReply]) -> tuple[ModelLeaseReply]:
    return (await reply,)


def _run(
    kind: str,
    action: Callable[[HandlerLeaseStore], Awaitable[Sequence[ModelLeaseReply]]],
) -> None:
    """Run one store operation, close its client, and print JSON replies."""
    config = click.get_current_context().find_object(LeaseCliConfig)
    if config is None:
        raise click.UsageError("run through the lease command group")
    endpoint, policies = config.resolve()
    if not any(policy.kind == kind for policy in policies):
        raise click.UsageError(f"lease kind {kind!r} is not declared")

    async def execute() -> Sequence[ModelLeaseReply]:
        adapter = await HandlerPlacementEtcd.connect(endpoint, 5.0)
        try:
            return await action(HandlerLeaseStore(adapter, policies))
        finally:
            await adapter.aclose()

    replies = asyncio.run(execute())
    for reply in replies:
        click.echo(json.dumps(reply.model_dump(mode="json"), sort_keys=True))
    if any(reply.outcome in {"lease-refused", "lease-expired"} for reply in replies):
        raise click.exceptions.Exit(3)


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
    request = ModelLeaseAcquireRequest(
        kind=kind,
        resource=resource,
        holder=holder,
        request_id=request_id,
        ttl_s=ttl_s,
    )
    _run(kind, lambda store: _single(store.acquire(request)))


@lease_group.command()
@click.option("--kind", required=True)
@click.option("--resource", required=True)
@click.option("--lease-id", type=click.UUID, required=True)
def renew(kind: str, resource: str, lease_id: UUID) -> None:
    """Keep a grant alive at the same fence."""
    _run(kind, lambda store: _single(store.renew(kind, resource, lease_id)))


@lease_group.command()
@click.option("--kind", required=True)
@click.option("--resource", required=True)
@click.option("--lease-id", type=click.UUID, required=True)
def release(kind: str, resource: str, lease_id: UUID) -> None:
    """Give a grant back."""
    _run(kind, lambda store: _single(store.release(kind, resource, lease_id)))


@lease_group.command()
@click.option("--kind", required=True)
@click.option("--resource", required=True)
def show(kind: str, resource: str) -> None:
    """Print one JSON line for every live grant of a resource."""
    _run(kind, lambda store: store.show(kind, resource))


__all__ = ["LeaseCliConfig", "lease_group"]
