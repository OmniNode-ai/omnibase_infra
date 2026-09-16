# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Which credential answers for a selected lane, in one place (OMN-18432).

``resolve_lane_target`` (OMN-16871) answers WHICH BROKER. This module answers
AS WHOM, and it is the only place that decides, because three surfaces reach
the same CLI and they disagree about where a credential lives:

* a developer machine holds the sanctioned by-reference client store
  (``~/.onex``, OMN-15922) and no SASL environment at all;
* the lab host and the CI runners hold the SASL environment and no client
  store;
* a machine holding neither is pointed at a lane that DECLARES SASL.

THE ORDER: STORE, THEN ENVIRONMENT, THEN REFUSE
    The store wins where both exist. The environment on the lab host is shared
    ambient state that every concurrent lane on that host inherits; an identity
    the operator stored explicitly for this machine is a stronger statement of
    intent than a variable somebody exported, and it is the one an audit can
    attribute.

    The environment remains a fallback rather than being deleted, because
    deleting it would change the behaviour of every container and CI runner in
    the fleet in a change whose subject is a developer Mac. Where it answers,
    this module returns ``None`` and the existing environment-sourced
    construction path runs untouched -- the same code, the same fields, the
    same result.

    Neither answering is a REFUSAL, before anything connects. The alternative
    is an anonymous connect against an auth-required listener, which surfaces
    minutes later as a handshake error naming nothing an operator can act on.

ONE CASE IS DELIBERATELY NOT A FALLBACK
    A client store that exists and REFUSES -- a credential file left
    group-readable, a dangling reference, an inline value in the config file --
    does not fall through to the environment. Falling through would mean a
    machine whose own stored identity is unreadable silently authenticates as
    somebody else, which is exactly the substitution the by-reference store
    exists to prevent. The defect is surfaced and the run stops.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.model_delegate_lane_selection import ModelDelegateLaneSelection
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)

__all__ = [
    "ENV_SASL_PASSWORD",
    "ENV_SASL_USERNAME",
    "DelegateLaneCredentialError",
    "resolve_lane_client_transport_for",
]

#: The two environment variables the fleet's containers and CI runners already
#: carry. Named here as NAMES, never read for a value except to decide whether
#: this module should step aside.
ENV_SASL_USERNAME = "KAFKA_SASL_USERNAME"
ENV_SASL_PASSWORD = "KAFKA_SASL_PASSWORD"  # pragma: allowlist secret


class DelegateLaneCredentialError(Exception):
    """No usable identity for a lane that declares an authenticated protocol.

    Raised for absence and for a broken stored credential alike. Both stop the
    run: there is no state of this module that hands back a transport it could
    not fully resolve.
    """


def resolve_lane_client_transport_for(
    *,
    lane_target: ModelDelegateLaneSelection | None,
    onex_home: Path,
    environ: Mapping[str, str],
) -> ModelLaneClientTransport | None:
    """Resolve the complete client transport for a selected lane.

    Args:
        lane_target: The lane the caller selected, or ``None`` when the caller
            stated a raw broker address instead. A raw address states an
            address and claims nothing about the transport, so it binds
            nothing and the existing environment path answers.
        onex_home: The machine's ``~/.onex`` directory. Injected rather than
            derived inside this function so tests drive a real directory.
        environ: The process environment, injected for the same reason.

    Returns:
        The transport to bind, or ``None`` when the existing
        environment-sourced construction path should answer untouched.

    Raises:
        DelegateLaneCredentialError: the lane declares an authenticated
            protocol and no identity resolves, or one resolves broken.
    """
    if lane_target is None:
        return None

    protocol = lane_target.security_protocol
    declared_in = lane_target.declared_in

    if not protocol.upper().startswith("SASL_"):
        # A non-SASL lane still STATES its transport, and stating it is what
        # stops a client from inheriting SASL_PLAINTEXT out of an ambient
        # environment and offering a handshake this listener does not speak.
        return ModelLaneClientTransport(
            lane=lane_target.lane,
            bootstrap_servers=lane_target.bootstrap_servers,
            security_protocol=protocol,
            declared_in=declared_in,
        )

    store = StoreLaneCredential(onex_home=onex_home)
    if onex_home.exists() and lane_target.lane in store.declared_lanes():
        try:
            credential = store.load(lane_target.lane)
        except ModelOnexError as exc:
            message = (
                f"lane {lane_target.lane!r} declares {protocol} and this "
                f"machine holds an identity for it, but that identity cannot "
                f"be read: {exc}. Refusing to fall back to the environment -- "
                "a machine whose stored identity is unreadable must not "
                "silently authenticate as somebody else."
            )
            raise DelegateLaneCredentialError(message) from exc
        return ModelLaneClientTransport(
            lane=lane_target.lane,
            bootstrap_servers=lane_target.bootstrap_servers,
            security_protocol=protocol,
            sasl_mechanism=lane_target.sasl_mechanism,
            sasl_username=credential.sasl_username,
            sasl_password=credential.sasl_password,
            declared_in=declared_in,
        )

    env_username = environ.get(ENV_SASL_USERNAME, "").strip()
    env_password = environ.get(ENV_SASL_PASSWORD, "").strip()
    if env_username and env_password:
        # The lab host and CI: step aside and let the construction path that
        # has always answered on those surfaces answer unchanged.
        return None
    if env_username or env_password:
        present = ENV_SASL_USERNAME if env_username else ENV_SASL_PASSWORD
        missing = ENV_SASL_PASSWORD if env_username else ENV_SASL_USERNAME
        message = (
            f"lane {lane_target.lane!r} declares {protocol}, declared in "
            f"{declared_in}, and this environment sets {present} but not "
            f"{missing}. Half a credential is not a credential: refusing "
            f"rather than opening a connection that cannot complete. Store "
            f"this machine's identity instead with 'onex auth lane-login "
            f"--lane {lane_target.lane} --sasl-username <principal> "
            "--sasl-password-stdin'."
        )
        raise DelegateLaneCredentialError(message)

    held = ", ".join(store.declared_lanes()) or "none"
    message = (
        f"lane {lane_target.lane!r} declares {protocol}, declared in "
        f"{declared_in}, and this machine holds no identity for it. Bus "
        f"identities held: {held}. Store one with 'onex auth lane-login "
        f"--lane {lane_target.lane} --sasl-username <principal> "
        "--sasl-password-stdin', which writes the reference to "
        f"{store.config_path} and the value to {store.credentials_path} at "
        "mode 0600. Refusing to connect anonymously to an authenticated "
        "listener."
    )
    raise DelegateLaneCredentialError(message)
