# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex auth`` -- credential in, gateway JWT out (OMN-15922).

Four commands, one credential, no per-harness code. Claude Code, Codex, cursor
and a bare terminal all shell out to the same binary, which is the property
that makes ``onex auth login`` + ``onex delegate`` work identically from all
four: the auth logic lives here, and a marketplace skill stays a thin shim over
it (zero auth logic in skill markdown).

    onex auth login   --tenant-slug S --client-id C --client-secret-stdin
    onex auth status
    onex auth token
    onex auth logout

SECRET HANDLING
    The secret is read from stdin, never from an argv flag. A ``--client-secret
    <value>`` option would put the credential in the process table, in shell
    history, and in any exec log -- three durable copies that outlive the
    session. It is written only to ``~/.onex/credentials.json`` at mode 0600 and
    referenced from ``config.yaml`` by name; ``status`` prints tenant, principal
    and expiry, and never prints secret material.

WHERE THE TRANSPORT COMES FROM
    ``token`` mints over the network, so it needs a concrete
    ``ProtocolGatewayTransport``. That adapter ships in this same distribution
    (``omnibase_infra.gateway.client.gateway_transport_httpx``) and is
    constructed directly. ``login``/``status``/``logout`` touch no network at
    all -- they are pure local file operations over ``~/.onex`` -- so they work
    regardless.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import click

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.gateway.client.gateway_token_minter import (
    GatewayTokenMinter,
)
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)

__all__ = ["auth_group"]


def _store() -> StoreGatewayCredential:
    return StoreGatewayCredential(onex_home=Path.home() / ".onex")


def _fail(message: str) -> NoReturn:
    """Report on stderr and exit non-zero.

    Typed ``NoReturn`` deliberately: every fail-closed branch below relies on
    control not continuing past it, and ``NoReturn`` makes the type checker
    enforce that rather than leaving it to reviewer attention.
    """
    click.echo(f"Error: {message}", err=True)
    sys.exit(1)


def _load_credential() -> ModelGatewayCredential:
    try:
        return _store().load()
    except ModelOnexError as exc:
        _fail(str(exc))


@click.group("auth")
def auth_group() -> None:  # stub-ok
    """Manage the gateway credential and the tokens minted from it."""


@auth_group.command("login")
@click.option(
    "--tenant-slug", required=True, help="Tenant slug the credential belongs to."
)
@click.option(
    "--client-id",
    required=True,
    help="Keycloak clientId of the per-tenant confidential client (this IS the principal_id).",
)
@click.option(
    "--token-endpoint",
    required=True,
    help="Realm token endpoint, e.g. https://<keycloak>/realms/<realm>/protocol/openid-connect/token",
)
@click.option(
    "--base-url", required=True, help="Gateway origin, e.g. https://api.omninode.ai"
)
@click.option(
    "--client-secret-stdin",
    is_flag=True,
    required=True,
    help="Read the client secret from stdin. The only accepted form -- a flag value would leak into the process table and shell history.",
)
@click.option(
    "--edge-instance-id",
    default="",
    help="Host label for session bookkeeping. Defaults to this machine's hostname.",
)
def auth_login(
    tenant_slug: str,
    client_id: str,
    token_endpoint: str,
    base_url: str,
    client_secret_stdin: bool,
    edge_instance_id: str,
) -> None:
    """Store a gateway credential by reference under ~/.onex.

    Writes the secret to ~/.onex/credentials.json (mode 0600) and a
    reference-only block to ~/.onex/config.yaml. Nothing else in config.yaml
    is disturbed.
    """
    if not client_secret_stdin:  # pragma: no cover - click marks the flag required
        _fail("--client-secret-stdin is required; the secret is never taken from argv.")

    secret = sys.stdin.read().strip()
    if not secret:
        _fail(
            "no client secret on stdin. Pipe it, e.g.: pbpaste | onex auth login ... --client-secret-stdin"
        )

    try:
        _store().save(
            tenant_slug=tenant_slug,
            client_id=client_id,
            client_secret=secret,
            token_endpoint=token_endpoint,
            base_url=base_url,
            edge_instance_id=edge_instance_id or socket.gethostname(),
        )
    except ModelOnexError as exc:
        _fail(str(exc))

    click.echo(
        f"Stored gateway credential for tenant '{tenant_slug}' (client_id {client_id})."
    )
    click.echo("Secret written by reference to ~/.onex/credentials.json (mode 0600).")


@auth_group.command("status")
def auth_status() -> None:
    """Print the stored credential's identity and endpoints.

    Never prints secret material -- not the client secret, not a token. This
    command is what an operator pastes into an issue.
    """
    credential = _load_credential()
    click.echo(f"tenant_slug:      {credential.tenant_slug}")
    click.echo(f"principal_id:     {credential.client_id}")
    click.echo(f"token_endpoint:   {credential.token_endpoint}")
    click.echo(f"gateway base_url: {credential.base_url}")
    click.echo(f"edge_instance_id: {credential.edge_instance_id}")
    click.echo("client_secret:    stored by reference (not shown)")


@auth_group.command("token")
def auth_token() -> None:
    """Mint and print a currently-valid gateway access token.

    The escape hatch any harness can shell out to. Emits the raw token on
    stdout and nothing else, so it composes; every diagnostic goes to stderr.
    The token printed is the ATTACH token from POST /v1/auth/gateway-token,
    not the machine token the stored credential grants directly (OMN-16687).
    Exits non-zero if the credential is missing, the grant is refused, the
    exchange refuses the credential, or the exchanged token's audience is not
    exactly the gateway-attach set.
    """
    credential = _load_credential()
    minter = GatewayTokenMinter(
        transport=GatewayTransportHttpx(),
        credential=credential,
    )
    try:
        token = asyncio.run(minter.token_for(now=datetime.now(UTC)))
    except ModelOnexError as exc:
        _fail(str(exc))
    click.echo(token.access_token.get_secret_value())


@auth_group.command("logout")
def auth_logout() -> None:
    """Remove the stored credential and the secret it references."""
    try:
        _store().clear()
    except ModelOnexError as exc:
        _fail(str(exc))
    click.echo("Removed the gateway credential from ~/.onex.")
