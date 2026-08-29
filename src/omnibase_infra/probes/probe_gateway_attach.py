# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``gateway_attach`` onboarding verification check (OMN-16036).

WHAT IT PROVES, AND WHY THE EXISTING CHECK IS NOT ENOUGH
    ``http_health`` against the gateway's ``/health`` proves the service is
    reachable and says nothing at all about the credential the onboarding flow
    just wrote. This check closes exactly that gap: it performs the same mint
    the runtime will perform -- a ``client_credentials`` grant for the tenant's
    machine token, then ``POST /v1/auth/gateway-token`` to exchange it for an
    attach-audience token (OMN-16687) -- presents the resulting Bearer to
    ``/v1/gateway/attach``, and only reports a pass once the gateway has opened
    a real, tenant-bound session. A credential that is mistyped, revoked,
    pointed at the wrong realm, or is not the tenant's provisioned machine
    client fails HERE, at onboarding, rather than at first delegation.

WHY IT DETACHES
    Attaching is a mutation of live control-plane state, and a verification
    check that leaves one session behind per run would accumulate them for the
    whole session TTL. The detach is therefore part of the proof, not a
    courtesy: the check reports a pass only when it also closed what it
    opened. A detach that FAILS is surfaced, naming the session id and its
    ceiling -- swallowing it would report a clean, non-destructive pass over a
    session that is still open.

REUSE, NOT A ONE-OFF (OMN-16036 AC4)
    None of the protocol lives here. The grant is ``GatewayTokenMinter``, the
    attach and detach are ``GatewaySessionKeeper``, the credential comes from
    ``StoreGatewayCredential``, and the socket is ``GatewayTransportHttpx`` --
    the same four objects the unattended runtime client drives for the
    OMN-15952 900s re-grant + re-attach cycle. This module is composition and
    result-shaping only, which is what keeps the verification path and the
    runtime path from drifting into two different notions of "attached".
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Final

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.gateway.client.gateway_session_keeper import GatewaySessionKeeper
from omnibase_infra.gateway.client.gateway_token_minter import GatewayTokenMinter
from omnibase_infra.gateway.client.gateway_transport_httpx import GatewayTransportHttpx
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)
from omnibase_infra.protocols.protocol_gateway_transport import (
    ProtocolGatewayTransport,
)

__all__ = ["check_gateway_attach", "prove_gateway_attach"]

# Recorded by the gateway on the session event, so an operator reading session
# history can tell a verification probe apart from a runtime that went away.
_DETACH_REASON: Final[str] = "onboarding verification probe (gateway_attach check)"

# Where ``onex auth login`` writes the credential. Used when a policy's check
# leaves ``target`` blank, which is the common case -- the onboarding step that
# writes the credential and the step that verifies it agree on this path.
_DEFAULT_ONEX_HOME_NAME: Final[str] = ".onex"


async def prove_gateway_attach(
    *,
    credential: ModelGatewayCredential,
    transport: ProtocolGatewayTransport,
    now: datetime,
    rng: random.Random | None = None,
    detach_reason: str = _DETACH_REASON,
) -> str:
    """Grant, attach, detach. Return a proof line; raise on any failure.

    Args:
        credential: The resolved credential to prove.
        transport: POST seam. Injected so the whole proof is driveable against
            an in-memory fake with no network.
        now: Caller-supplied instant, as everywhere in the gateway client.
        rng: Jitter source for the keeper's renewal draw. The proof detaches
            immediately and never renews, so the draw is not observable here;
            a seeded default keeps the composition deterministic.
        detach_reason: Teardown reason recorded on the session event.

    Returns:
        A single operator-facing line naming the tenant, the principal, the
        session that was opened, its ceiling, and the renewal directive the
        gateway declared. Never carries the client secret or the access token.

    Raises:
        ModelOnexError: If the grant is refused, the credential is not the
            tenant's machine client, the attach-token exchange refuses it, the
            exchanged token's audience is not the gateway-attach set, the
            gateway rejects the attach, the attach response omits the required
            renewal directive, or the detach fails. A failed detach names the
            session left open.
    """
    keeper = GatewaySessionKeeper(
        transport=transport,
        credential=credential,
        minter=GatewayTokenMinter(transport=transport, credential=credential),
        rng=rng if rng is not None else random.Random(0),
    )

    attachment = await keeper.attach(now=now)
    session = attachment.session

    try:
        await keeper.detach(now=now, reason=detach_reason)
    except ModelOnexError as exc:
        # The attach succeeded, so the credential IS good -- but the check
        # cannot claim to be non-destructive. Both facts go to the operator,
        # with the identifier they need to clean up.
        raise ModelOnexError(
            f"gateway attach succeeded but the detach did not: {exc}. Session "
            f"{session.session_id} is still open and will remain so until its "
            f"ceiling at {session.expires_at.isoformat()}.",
            error_code=EnumCoreErrorCode.OPERATION_FAILED,
        ) from exc

    return (
        f"gateway attach proof passed for tenant '{credential.tenant_slug}' "
        f"(principal '{credential.client_id}') against "
        f"{credential.base_url.rstrip('/')}: client_credentials grant and "
        f"attach-token exchange accepted, "
        f"session {session.session_id} opened with ceiling "
        f"{session.expires_at.isoformat()} and renewal "
        f"{attachment.renewal.mode.value} at "
        f"{attachment.renewal.renew_at.isoformat()}, then detached -- no "
        "session left open."
    )


async def check_gateway_attach(
    target: str,
    timeout: int,
    *,
    transport: ProtocolGatewayTransport | None = None,
) -> tuple[bool, str]:
    """``gateway_attach`` check_type entry point for the verification executor.

    Args:
        target: Directory holding the credential ``onex auth login`` wrote
            (``config.yaml`` + the 0600 ``credentials.json``). Blank means
            ``~/.onex``, the path the CLI writes. ``${VAR}`` references are
            already expanded by the executor.
        timeout: Budget in seconds for the WHOLE sequence -- grant, exchange,
            attach and detach -- not per request. A policy should allow more
            than the executor's 10s default here: four round trips against a
            cold Keycloak realm do not reliably fit in it.
        transport: Injected POST seam; the real httpx adapter when omitted.

    Returns:
        ``(passed, message)``. Every failure -- missing or mis-permissioned
        credential, refused grant, wrong audience at either hop, refused
        exchange, rejected attach, failed detach, unreachable gateway,
        exhausted budget -- comes back as
        ``(False, <what went wrong and what to do>)``. There is no branch that
        reports a pass without a session having been opened and closed.
    """
    onex_home = (
        Path(target).expanduser()
        if target.strip()
        else Path.home() / _DEFAULT_ONEX_HOME_NAME
    )

    try:
        credential = StoreGatewayCredential(onex_home=onex_home).load()
        proof = await asyncio.wait_for(
            prove_gateway_attach(
                credential=credential,
                transport=(
                    transport
                    if transport is not None
                    else GatewayTransportHttpx(timeout_seconds=float(timeout))
                ),
                now=GatewayTokenMinter.utc_now(),
            ),
            timeout=timeout,
        )
    except TimeoutError:
        return (
            False,
            f"gateway attach proof timed out after {timeout}s (grant + "
            f"exchange + attach + detach against the credential in "
            f"{onex_home}). Raise the check's "
            "timeout_seconds if the control plane is simply slow, or check "
            "whether it is reachable at all.",
        )
    except ModelOnexError as exc:
        # ModelOnexError covers the infra transport errors too -- they subclass
        # it -- so an unreachable gateway and a refused credential both land
        # here and both keep their own remediation text.
        return False, f"gateway attach proof failed: {exc}"

    return True, proof
