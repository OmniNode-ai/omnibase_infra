# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed Kafka configuration for the deploy-agent control bus.

TRANSPORT IS DECLARED, NEVER INFERRED (OMN-18012)
-------------------------------------------------

The CI publishers read their control-bus transport out of a checked-in lane
declaration -- omnimarket ``config/ci_bus_lanes.yaml``, keys ``security_protocol``
and ``sasl_mechanism`` beside each lane's broker -- and
``omnibase_infra/scripts/trigger_rebuild_on_merge.py`` enforces that contract
field-for-field. This module is the deploy agent's half of the same contract,
with the same vocabulary:

* ``security_protocol`` -- one of PLAINTEXT | SSL | SASL_PLAINTEXT | SASL_SSL,
  spelled as librdkafka spells it. REQUIRED: the agent's broker is always a
  concrete cross-process ``host:port``, so there is nothing to default to and
  the loader refuses to guess.
* ``sasl_mechanism`` -- REQUIRED when the protocol is SASL_*, and contradictory
  (rejected) beside a non-SASL protocol.

The agent has no overlay file of its own to read: it is a systemd unit on the
.201 host, so its lane declaration surface IS the unit
(``deploy/deploy-agent-dev.service`` / ``deploy/deploy-agent.service``), which
sets these names as ``Environment=`` lines beside the broker it already
declares there.

WHY THE OLD SHAPE BROKE THE DEV LANE
------------------------------------

The previous loader selected ``SASL_SSL`` whenever SASL credentials were
present in the environment and ``PLAINTEXT`` otherwise, and the model admitted
only those two protocols with only the ``PLAIN`` mechanism. When OMN-18012
Phase B enabled SASL/SCRAM-SHA-256 on the .201 dev-lane Redpanda EXTERNAL
listener at ~16:40Z on 2026-09-07 -- SASL over PLAINTEXT, authenticated but not
encrypted, no TLS on :19092 -- the dev agent had no way to say so. It kept
bootstrapping in the clear and the broker dropped every connection
(``socket disconnected`` on each attempt, 53 unit restarts by 2026-09-08T04:12Z),
so ``onex.cmd.deploy.rebuild-requested.v1`` on the dev broker went unconsumed.

Credential PRESENCE is not a statement about transport. The lane is.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENV_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
ENV_SECURITY_PROTOCOL = "KAFKA_SECURITY_PROTOCOL"
ENV_SASL_MECHANISM = "KAFKA_SASL_MECHANISM"
ENV_SASL_USERNAME = "KAFKA_SASL_USERNAME"
ENV_SASL_PASSWORD = "KAFKA_SASL_PASSWORD"
ENV_SASL_ENV_PREFIX = "KAFKA_SASL_ENV_PREFIX"

SecurityProtocol = Literal["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"]
SaslMechanism = Literal["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"]

SASL_PROTOCOLS = frozenset({"SASL_PLAINTEXT", "SASL_SSL"})
NON_SASL_PROTOCOLS = frozenset({"PLAINTEXT", "SSL"})
VALID_SECURITY_PROTOCOLS = SASL_PROTOCOLS | NON_SASL_PROTOCOLS
VALID_SASL_MECHANISMS = frozenset({"PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"})


class ModelDeployAgentKafkaConfig(BaseModel):
    """Single source of truth for deploy-agent consume and publish bus config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_servers: str = Field(..., min_length=1)
    security_protocol: SecurityProtocol
    sasl_mechanism: SaslMechanism | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    sasl_env_prefix: str = ""
    """Prefix the credentials were declared under, for error messages only.

    The .201 operator env file holds the dev lane's SCRAM principal as
    ``DEV_KAFKA_SASL_USERNAME`` / ``DEV_KAFKA_SASL_PASSWORD``, and systemd
    cannot expand one ``Environment=`` value into another. The unit therefore
    declares the prefix instead of duplicating a secret into a unit file, and
    this field is what lets a refusal name the exact variables it looked for.
    """

    @field_validator("security_protocol", mode="before")
    @classmethod
    def normalise_security_protocol(cls, value: object) -> str:
        """Accept only a librdkafka security protocol, normalised to upper case."""
        if not isinstance(value, str):
            raise ValueError(
                f"{ENV_SECURITY_PROTOCOL} must be a string naming a librdkafka "
                f"security protocol; valid values: {sorted(VALID_SECURITY_PROTOCOLS)}"
            )
        protocol = value.strip().upper()
        if protocol not in VALID_SECURITY_PROTOCOLS:
            raise ValueError(
                f"{value!r} is not a librdkafka security protocol; valid values: "
                f"{sorted(VALID_SECURITY_PROTOCOLS)}"
            )
        return protocol

    @field_validator("sasl_mechanism", mode="before")
    @classmethod
    def normalise_sasl_mechanism(cls, value: object) -> str | None:
        """Reject an unknown or empty mechanism rather than passing it to the client."""
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{ENV_SASL_MECHANISM} must name a SASL mechanism when declared; "
                f"valid values: {sorted(VALID_SASL_MECHANISMS)}"
            )
        mechanism = value.strip().upper()
        if mechanism not in VALID_SASL_MECHANISMS:
            raise ValueError(
                f"{value!r} is not a supported SASL mechanism; valid values: "
                f"{sorted(VALID_SASL_MECHANISMS)}"
            )
        return mechanism

    @property
    def _username_env_name(self) -> str:
        return f"{self.sasl_env_prefix}{ENV_SASL_USERNAME}"

    @property
    def _password_env_name(self) -> str:
        return f"{self.sasl_env_prefix}{ENV_SASL_PASSWORD}"

    @model_validator(mode="after")
    def validate_declared_transport(self) -> ModelDeployAgentKafkaConfig:
        """Enforce the lane's transport declaration, never a guess."""
        if bool(self.sasl_username) != bool(self.sasl_password):
            raise ValueError(
                f"{self._username_env_name} and {self._password_env_name} "
                "must be set together"
            )

        if self.security_protocol in SASL_PROTOCOLS:
            if self.sasl_mechanism is None:
                raise ValueError(
                    f"security_protocol {self.security_protocol} requires a "
                    f"{ENV_SASL_MECHANISM} (e.g. SCRAM-SHA-256); none was declared"
                )
            if not self.sasl_username or not self.sasl_password:
                raise ValueError(
                    f"security_protocol {self.security_protocol} requires SASL "
                    f"credentials: set {self._username_env_name} and "
                    f"{self._password_env_name}. A SASL lane with no principal "
                    "fails closed rather than falling back to the clear"
                )
            return self

        if self.sasl_mechanism:
            raise ValueError(
                f"sasl_mechanism {self.sasl_mechanism!r} is declared beside "
                f"security_protocol {self.security_protocol!r}, which carries no "
                "SASL; fix the declaration rather than let the agent pick a half"
            )
        if self.sasl_username:
            raise ValueError(
                f"SASL credentials are declared beside security_protocol "
                f"{self.security_protocol!r}, which carries no SASL. "
                f"{ENV_SECURITY_PROTOCOL} must declare SASL_PLAINTEXT or SASL_SSL "
                "for them to be used; credential presence alone never selects a "
                "transport"
            )
        return self

    def consumer_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "bootstrap_servers": self.bootstrap_servers,
            "security_protocol": self.security_protocol,
        }
        if self.security_protocol in SASL_PROTOCOLS:
            kwargs.update(
                {
                    "sasl_mechanism": self.sasl_mechanism,
                    "sasl_plain_username": self.sasl_username,
                    "sasl_plain_password": self.sasl_password,
                }
            )
        return kwargs

    def producer_kwargs(self) -> dict[str, Any]:
        return self.consumer_kwargs()


def load_deploy_agent_kafka_config_from_env() -> ModelDeployAgentKafkaConfig:
    """Load the required deploy-agent Kafka config from the environment.

    OMN-9713: there is intentionally no localhost fallback. A missing
    ``KAFKA_BOOTSTRAP_SERVERS`` must fail startup rather than silently consuming
    a different bus from the trigger publisher.

    OMN-18012: there is likewise no transport fallback and no inference. The
    unit declares ``KAFKA_SECURITY_PROTOCOL`` (and ``KAFKA_SASL_MECHANISM`` for
    a SASL protocol) beside the broker it already declares; an undeclared
    transport is a refusal.
    """
    bootstrap_servers = os.environ.get(ENV_BOOTSTRAP_SERVERS, "").strip()
    if not bootstrap_servers:
        raise RuntimeError(
            f"{ENV_BOOTSTRAP_SERVERS} is required for deploy-agent; "
            "there is no localhost fallback"
        )

    security_protocol = os.environ.get(ENV_SECURITY_PROTOCOL, "").strip()
    if not security_protocol:
        raise RuntimeError(
            f"{ENV_SECURITY_PROTOCOL} is required for deploy-agent; declare one "
            f"of {sorted(VALID_SECURITY_PROTOCOLS)} on the unit beside "
            f"{ENV_BOOTSTRAP_SERVERS}. Refusing to guess the transport: "
            "credential presence is not a statement about it (OMN-18012)"
        )

    prefix = os.environ.get(ENV_SASL_ENV_PREFIX, "").strip()
    username = os.environ.get(f"{prefix}{ENV_SASL_USERNAME}") or None
    password = os.environ.get(f"{prefix}{ENV_SASL_PASSWORD}") or None

    return ModelDeployAgentKafkaConfig(
        bootstrap_servers=bootstrap_servers,
        security_protocol=security_protocol,
        sasl_mechanism=os.environ.get(ENV_SASL_MECHANISM) or None,
        sasl_username=username,
        sasl_password=password,
        sasl_env_prefix=prefix,
    )
