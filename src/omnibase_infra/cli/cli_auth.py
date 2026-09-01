# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex auth`` -- credential in, gateway JWT out (OMN-15922).

Four commands, one credential, no per-harness code. Claude Code, Codex, cursor
and a bare terminal all shell out to the same binary, which is the property
that makes ``onex auth login`` + ``onex delegate`` work identically from all
four: the auth logic lives here, and a marketplace skill stays a thin shim over
it (zero auth logic in skill markdown).

    onex auth login   --tenant-slug S --client-id C --client-secret-stdin
    onex auth login   --tenant-slug S --base-url U --api-key-stdin
    onex auth status
    onex auth token
    onex auth logout

TWO CREDENTIAL KINDS, ONE COMMAND (OMN-17205)
    onex-api resolves a caller's tenant from either an OIDC bearer or a tenant
    API key, on equal footing. ``--client-secret-stdin`` stores the first;
    ``--api-key-stdin`` stores the second. They are mutually exclusive: one
    machine holds one gateway credential, and a machine holding both would let
    a read authenticate as an identity nobody chose. The api-key form needs no
    ``--client-id`` and no ``--token-endpoint`` -- an API key is presented
    directly and grants nothing.

SECRET HANDLING
    The secret is read from stdin, never from an argv flag. A ``--client-secret
    <value>`` option would put the credential in the process table, in shell
    history, and in any exec log -- three durable copies that outlive the
    session. It is written only to ``~/.onex/credentials.json`` at mode 0600 and
    referenced from ``config.yaml`` by name; ``status`` prints tenant, principal
    and expiry, and never prints secret material.

WHERE THE TRANSPORT COMES FROM
    ``token`` mints over the network, and ``status`` verifies over it, so both
    need a concrete transport. That adapter ships in this same distribution
    (``omnibase_infra.gateway.client.gateway_transport_httpx``) and is
    constructed directly. ``login`` and ``logout`` touch no network at all --
    they are pure local file operations over ``~/.onex`` -- and
    ``status --no-verify`` keeps that property for status too.

WHAT ``status`` ANSWERS (OMN-17028)
    "Am I authenticated", not "does my config parse". For an API-key
    credential it presents the key to ``GET /v1/whoami`` and reports the tenant
    the GATEWAY resolved, so a revoked key, a key for a different origin, or a
    local label that disagrees with the server all surface as a non-zero exit
    instead of a confident printout. The attach-plane credential is proven by
    ``onex auth token`` instead, and ``status`` says so rather than implying it
    checked.
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
from omnibase_infra.gateway.client.gateway_identity_verifier import (
    GatewayIdentityVerifier,
)
from omnibase_infra.gateway.client.gateway_token_minter import (
    GatewayTokenMinter,
)
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_credential_base import (
    ModelGatewayCredentialBase,
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
    """Resolve the ATTACH-plane credential specifically.

    Used by ``token``, which mints against a realm this credential kind is the
    only one to carry. ``status`` deliberately does NOT go through here: it
    reports on whichever kind the machine holds, and routing it through the
    client-credentials reader is what made it fail on a machine that had just
    successfully stored an API key (OMN-17028).
    """
    try:
        return _store().load()
    except ModelOnexError as exc:
        _fail(str(exc))


def _load_read_credential() -> ModelGatewayCredentialBase:
    """Resolve whichever credential kind this machine actually holds."""
    try:
        return _store().load_read_credential()
    except ModelOnexError as exc:
        _fail(str(exc))


def _verify_api_key(credential: ModelGatewayApiKeyCredential) -> None:
    """Ask the gateway who this key is, and refuse to report a mismatch quietly.

    A stored ``tenant_slug`` that disagrees with the gateway's own answer is a
    config that lies: the slug names the ref the key is filed under and appears
    in every operator-facing message about this credential, so leaving the two
    disagreeing means every later message names the wrong tenant.
    """
    verifier = GatewayIdentityVerifier(
        transport=GatewayTransportHttpx(),
        credential=credential,
    )
    try:
        identity = asyncio.run(verifier.verify())
    except ModelOnexError as exc:
        _fail(str(exc))

    if identity.tenant_slug != credential.tenant_slug:
        _fail(
            f"the stored key authenticates as tenant '{identity.tenant_slug}', "
            f"but this machine has it filed under '{credential.tenant_slug}'. "
            "Re-run onboarding with the slug the dashboard shows for this key."
        )

    click.echo(f"verified:         {credential.base_url} resolved this key")
    click.echo(f"                  as tenant '{identity.tenant_slug}'")


def _read_stdin_secret(what: str, example: str) -> str:
    """Read one secret from stdin, or fail naming how to pipe it."""
    value = sys.stdin.read().strip()
    if not value:
        _fail(f"no {what} on stdin. Pipe it, e.g.: {example}")
    return value


def _login_with_api_key(*, tenant_slug: str, base_url: str) -> None:
    """Store a tenant API key by reference and report without echoing it."""
    api_key = _read_stdin_secret(
        "API key",
        "pbpaste | onex auth login --tenant-slug <slug> --base-url <origin> --api-key-stdin",
    )
    try:
        _store().save_api_key(
            tenant_slug=tenant_slug, api_key=api_key, base_url=base_url
        )
    except ModelOnexError as exc:
        _fail(str(exc))
    click.echo(f"Stored gateway API key for tenant '{tenant_slug}'.")
    click.echo("Key written by reference to ~/.onex/credentials.json (mode 0600).")


@click.group("auth")
def auth_group() -> None:  # stub-ok
    """Manage the gateway credential and the tokens minted from it."""


@auth_group.command("login")
@click.option(
    "--tenant-slug", required=True, help="Tenant slug the credential belongs to."
)
@click.option(
    "--client-id",
    default="",
    help="Keycloak clientId of the per-tenant confidential client (this IS the principal_id). Required for --client-secret-stdin.",
)
@click.option(
    "--token-endpoint",
    default="",
    help="Realm token endpoint, e.g. https://<keycloak>/realms/<realm>/protocol/openid-connect/token. Required for --client-secret-stdin.",
)
@click.option(
    "--base-url", required=True, help="Gateway origin, e.g. https://api.omninode.ai"
)
@click.option(
    "--client-secret-stdin",
    is_flag=True,
    default=False,
    help="Read the client secret from stdin. The only accepted form -- a flag value would leak into the process table and shell history.",
)
@click.option(
    "--api-key-stdin",
    is_flag=True,
    default=False,
    help="Read a tenant API key from stdin instead of a client secret. Same stdin-only rule, same 0600 by-reference storage.",
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
    api_key_stdin: bool,
    edge_instance_id: str,
) -> None:
    """Store a gateway credential by reference under ~/.onex.

    Writes the secret to ~/.onex/credentials.json (mode 0600) and a
    reference-only block to ~/.onex/config.yaml. Nothing else in config.yaml
    is disturbed.
    """
    if client_secret_stdin and api_key_stdin:
        _fail(
            "--client-secret-stdin and --api-key-stdin are mutually exclusive; "
            "one machine holds one gateway credential."
        )
    if not client_secret_stdin and not api_key_stdin:
        _fail(
            "one of --client-secret-stdin or --api-key-stdin is required; a "
            "secret is never taken from argv."
        )

    if api_key_stdin:
        _login_with_api_key(tenant_slug=tenant_slug, base_url=base_url)
        return

    missing = [
        name
        for name, value in (
            ("--client-id", client_id),
            ("--token-endpoint", token_endpoint),
        )
        if not value
    ]
    if missing:
        _fail(f"{' and '.join(missing)} required with --client-secret-stdin.")

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
@click.option(
    "--verify/--no-verify",
    default=True,
    show_default=True,
    help=(
        "Ask the gateway who the stored key is, instead of only printing what "
        "the local config claims. Verification is the default because "
        "'my config file parses' is not the question an operator asking for "
        "status is asking. --no-verify keeps the command purely local, which "
        "is what you want when pasting output into an issue offline."
    ),
)
def auth_status(verify: bool) -> None:
    """Print the stored credential's identity and endpoints.

    Never prints secret material -- not the API key, not the client secret, not
    a token. This command is what an operator pastes into an issue.

    TWO CREDENTIAL KINDS, ONE COMMAND, NO GUESSING (OMN-17028)
        This resolves whichever kind the machine actually holds, because the
        two surfaces are genuinely different and demanding one's fields of the
        other is what made a successfully-onboarded machine report itself
        unauthenticated. An API key belongs to the delegation path
        (``POST /v1/workflows``) and carries no ``edge_instance_id``, no
        principal and no token endpoint -- those are ATTACH-plane fields, and
        requiring them of a key credential asked for four values that path
        never has. The store refuses a machine holding both, so nothing here
        picks a winner.
    """
    credential = _load_read_credential()

    if isinstance(credential, ModelGatewayApiKeyCredential):
        click.echo("credential kind:  tenant API key (delegation path)")
        click.echo(f"tenant_slug:      {credential.tenant_slug}")
        click.echo(f"gateway base_url: {credential.base_url}")
        click.echo("api_key:          stored by reference (not shown)")
        if not verify:
            click.echo("verified:         not attempted (--no-verify)")
            return
        _verify_api_key(credential)
        return

    if not isinstance(credential, ModelGatewayCredential):
        # Not reachable through the store today, and deliberately a refusal
        # rather than a generic printout: a third credential kind arriving here
        # would otherwise be reported with whichever fields happened to exist,
        # which is how a machine gets told it is authenticated for a surface
        # nobody checked.
        _fail(
            "the stored credential is of a kind this command cannot report on. "
            "Re-run 'onex auth login'."
        )

    click.echo("credential kind:  client credentials (gateway attach plane)")
    click.echo(f"tenant_slug:      {credential.tenant_slug}")
    click.echo(f"principal_id:     {credential.client_id}")
    click.echo(f"token_endpoint:   {credential.token_endpoint}")
    click.echo(f"gateway base_url: {credential.base_url}")
    click.echo(f"edge_instance_id: {credential.edge_instance_id}")
    click.echo("client_secret:    stored by reference (not shown)")
    # Stated rather than silently skipped: an attach credential is proven by
    # minting the attach token, which is a different call with a different
    # audience rule. Reporting it "verified" off a whoami would claim a
    # property this command did not check.
    click.echo("verified:         not attempted (run 'onex auth token')")


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
