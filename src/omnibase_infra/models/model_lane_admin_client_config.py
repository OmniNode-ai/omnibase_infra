# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""librdkafka admin client config for a declared lane, authentication included (OMN-21529).

``scripts/create_kafka_topics.py`` took a bare ``--bootstrap-servers`` and built
``AdminClient({"bootstrap.servers": ...})``. A broker that requires
``SASL_PLAINTEXT`` with ``SCRAM-SHA-256`` (the dev broker on ``:19092``) refuses
that client, so provisioning a lane's topics needed a scratch wrapper that
carried the identity by hand.

The runtime already answers this question for every other client: the lane
declaration states the address, protocol and mechanism, and the identity comes
from the machine's by-reference store (or the SASL environment on the lab host
and CI). This model reuses those resolvers and renders their answer in
librdkafka's vocabulary, so the provisioning path reads the transport the same
way the runtime does and no protocol, mechanism or secret is a flag literal.
It reads no process environment itself: the caller passes the mapping it read.

A lane that declares SASL and resolves no identity is refused before any
connection, naming the declaration key and the command that stores one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli.delegate_lane import (
    DelegateLaneSelectionError,
    resolve_lane_selection_at,
)
from omnibase_infra.cli.delegate_lane_credentials import (
    ENV_SASL_PASSWORD,
    ENV_SASL_USERNAME,
    DelegateLaneCredentialError,
    resolve_lane_client_transport_for,
)


class ModelLaneAdminClientConfig(BaseModel):
    """A librdkafka client config for one lane. Holds a secret: never logged."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: str = Field(..., min_length=1, description="Lane id in ci_bus_lanes.yaml.")
    client_config: dict[str, str] = Field(
        ..., description="librdkafka client properties, credential included."
    )

    def __repr__(self) -> str:
        return f"ModelLaneAdminClientConfig(lane={self.lane!r}, keys={sorted(self.client_config)})"

    @classmethod
    def resolve(
        cls,
        lane: str,
        environ: Mapping[str, str],
        onex_home: Path | None = None,
    ) -> ModelLaneAdminClientConfig:
        """Resolve ``lane``; connects to nothing.

        ``environ`` locates the lane declaration (``OMNIBASE_PATH`` then
        ``OMNI_HOME``) and carries the lab host and CI identity
        (``KAFKA_SASL_USERNAME`` and ``KAFKA_SASL_PASSWORD``). The stored
        identity is read from ``onex_home``, ``~/.onex`` by default.

        Raises:
            ValueError: the lane is undeclared, or it declares SASL and no
                usable identity resolves. The text names ``lanes.<lane>``.
        """
        workspace = environ.get("OMNIBASE_PATH") or environ.get("OMNI_HOME")
        try:
            selection = resolve_lane_selection_at(
                lane, Path(workspace) if workspace else None
            )
            transport = resolve_lane_client_transport_for(
                lane_target=selection,
                onex_home=onex_home or Path.home() / ".onex",
                environ=environ,
            )
        except (DelegateLaneSelectionError, DelegateLaneCredentialError) as exc:
            message = (
                f"cannot resolve the admin client for lane {lane!r} "
                f"(overlay key lanes.{lane}): {exc}"
            )
            raise ValueError(message) from exc

        config: dict[str, str] = {
            "bootstrap.servers": selection.bootstrap_servers,
            "security.protocol": selection.security_protocol,
        }
        if transport is None:
            # The lab host and CI: the resolver stepped aside because both SASL
            # environment values are set; it has already refused half a pair.
            if selection.security_protocol.upper().startswith("SASL_"):
                if selection.sasl_mechanism is None:
                    message = (
                        f"lane {lane!r} declares {selection.security_protocol} "
                        f"with no sasl_mechanism (overlay key lanes.{lane})"
                    )
                    raise ValueError(message)
                config["sasl.mechanism"] = selection.sasl_mechanism
                config["sasl.username"] = environ[ENV_SASL_USERNAME].strip()
                config["sasl.password"] = environ[ENV_SASL_PASSWORD].strip()
        elif transport.is_sasl and transport.sasl_mechanism is not None:
            config["sasl.mechanism"] = transport.sasl_mechanism
            if transport.sasl_username is not None:
                config["sasl.username"] = transport.sasl_username
            if transport.sasl_password is not None:
                config["sasl.password"] = transport.sasl_password.get_secret_value()
        return cls(lane=lane, client_config=config)
