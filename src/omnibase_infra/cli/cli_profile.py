# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Profile CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import click

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.store_developer_profile import StoreDeveloperProfile
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential

__all__ = ["profile_group"]


def _store() -> StoreDeveloperProfile:
    """Return a StoreDeveloperProfile rooted at ~/.onex."""
    return StoreDeveloperProfile(onex_home=Path.home() / ".onex")


def _fail(message: str) -> NoReturn:
    """Print message to stderr and exit with status 1."""
    click.echo(message, err=True)
    raise SystemExit(1)


@click.group(
    name="profile", help="Per-developer defaults in ~/.onex/config.yaml (OMN-19973)."
)
def profile_group() -> None:  # stub-ok: click group
    """Profile commands."""


@profile_group.command(name="bind-lane")
@click.argument("lane")
def bind_lane_cmd(lane: str) -> None:
    """Bind a lane for onex delegate dispatch."""
    store = _store()
    try:
        store.bind_lane(lane)
        lane = store.lane_binding() or lane
    except ModelOnexError as exc:
        _fail(str(exc))

    lane_store = StoreLaneCredential(onex_home=Path.home() / ".onex")
    if lane not in lane_store.declared_lanes():
        click.echo(
            f"WARNING: no bus identity stored for lane '{lane}' yet; "
            "onex delegate will refuse until you run "
            f"'onex auth lane-login --lane {lane} --sasl-username <principal> --sasl-password-stdin'",
            err=True,
        )

    click.echo(
        f"Bound lane '{lane}': onex delegate with no --bus now dispatches to it "
        f"(--bus kafka --lane {lane} --locus deployed-lane). Explicit flags still win."
    )


@profile_group.command(name="unbind-lane")
def unbind_lane_cmd() -> None:
    """Unbind the current lane."""
    store = _store()
    try:
        old = store.lane_binding()
        unbound = store.unbind_lane()
    except ModelOnexError as exc:
        _fail(str(exc))

    if unbound:
        click.echo(f"Unbound lane '{old}'.")
    else:
        click.echo("No lane was bound.")


@profile_group.command(name="show")
def show_cmd() -> None:
    """Show the current lane binding."""
    try:
        lane = _store().lane_binding()
    except ModelOnexError as exc:
        _fail(str(exc))

    if lane is None:
        click.echo("lane_binding: (none)")
    else:
        click.echo(f"lane_binding: {lane}")
