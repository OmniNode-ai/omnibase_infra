# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The whole transport a client uses for one lane, address included (OMN-18432).

OMN-16871 moved the broker ADDRESS out of the ambient environment and into the
lane declaration, and stopped there. The protocol, the mechanism and the
credential stayed in ``KAFKA_SECURITY_PROTOCOL`` / ``KAFKA_SASL_*``, so a
process could resolve a governed lane's address from a checked-in file and then
assemble the rest of the connection out of whatever the shell exported. The two
halves were split once before -- OMN-18012, where credential PRESENCE was used
to INFER a transport and picked TLS against a listener that speaks none, 93
times in one CI run. An address with no transport beside it is not a usable
statement about a lane.

This model is that statement, whole. It is built only from a lane declaration
plus a credential the machine resolved by reference, and it carries its own
provenance so a refusal can name the file that answered.

THE VALIDATOR IS THE POINT
    A SASL protocol with no mechanism, a SASL protocol with no credential, and
    a plaintext protocol carrying a credential are all refused at construction.
    There is therefore no half-built transport for a caller to pass on and no
    branch downstream that has to decide what a partial one means -- the one
    decision is made once, here, where the declaration and the credential are
    both in hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

__all__ = ["SASL_PROTOCOL_PREFIX", "ModelLaneClientTransport"]

#: librdkafka spells every authenticated protocol with this prefix
#: (``SASL_PLAINTEXT``, ``SASL_SSL``). The lane declaration uses librdkafka's
#: own spelling, so this is a read of the declaration's vocabulary rather than
#: a parallel enum that could drift from it.
SASL_PROTOCOL_PREFIX: Final[str] = "SASL_"


class ModelLaneClientTransport(BaseModel):
    """One lane's complete client transport: where, how, and as whom."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    lane: str = Field(description="Lane id this transport was resolved for")
    bootstrap_servers: str = Field(
        description="Broker the declaration binds to this lane, host:port"
    )
    security_protocol: str = Field(
        description="librdkafka security protocol, exactly as the lane declares it"
    )
    sasl_mechanism: str | None = Field(
        default=None,
        description="Declared SASL mechanism; absent on a non-SASL protocol",
    )
    sasl_username: str | None = Field(
        default=None, description="SASL principal this machine authenticates as"
    )
    sasl_password: SecretStr | None = Field(
        default=None,
        description="Resolved by reference from the machine's 0600 credential "
        "file; wrapped so it cannot reach a log line through an ordinary repr",
    )
    declared_in: Path = Field(
        description="The lane declaration the address and protocol were read from"
    )

    @property
    def is_sasl(self) -> bool:
        return self.security_protocol.upper().startswith(SASL_PROTOCOL_PREFIX)

    @model_validator(mode="after")
    def _refuse_a_half_built_transport(self) -> ModelLaneClientTransport:
        has_credential = (
            self.sasl_username is not None or self.sasl_password is not None
        )

        if self.is_sasl:
            if self.sasl_mechanism is None:
                message = (
                    f"lane {self.lane!r} declares {self.security_protocol} but "
                    "no sasl_mechanism; a SASL protocol with no mechanism "
                    "cannot be connected with."
                )
                raise ValueError(message)
            if self.sasl_username is None or self.sasl_password is None:
                message = (
                    f"lane {self.lane!r} declares {self.security_protocol} but "
                    "carries no resolved credential. A transport is never "
                    "built half-authenticated: the caller refuses instead, "
                    "naming the lane and how to store the identity."
                )
                raise ValueError(message)
            return self

        if self.sasl_mechanism is not None:
            message = (
                f"lane {self.lane!r} declares sasl_mechanism "
                f"{self.sasl_mechanism!r} beside non-SASL protocol "
                f"{self.security_protocol}; that is a contradiction, not a "
                "spare setting."
            )
            raise ValueError(message)
        if has_credential:
            message = (
                f"lane {self.lane!r} declares non-SASL protocol "
                f"{self.security_protocol} but carries a credential. A "
                "credential beside a plaintext protocol is the OMN-18012 "
                "inference this model exists to make impossible."
            )
            raise ValueError(message)
        return self

    def as_client_config_overrides(self) -> dict[str, str]:
        """The Kafka client config fields this transport states.

        Field names are ``ModelKafkaEventBusConfig``'s own, so the result is
        applied with ``model_copy(update=...)`` and a renamed field fails at
        the model rather than silently dropping a credential.

        A non-SASL lane yields exactly one key. An empty string is never
        emitted for an absent credential: a blank username is a value the
        client would try to authenticate with, not an absence.
        """
        overrides: dict[str, str] = {"security_protocol": self.security_protocol}
        if not self.is_sasl:
            return overrides
        # The validator has already refused every partial combination, so all
        # three are present here and mypy's narrowing is preserved by the
        # explicit reads rather than by an assert.
        mechanism = self.sasl_mechanism
        username = self.sasl_username
        password = self.sasl_password
        if mechanism is not None:
            overrides["sasl_mechanism"] = mechanism
        if username is not None:
            overrides["sasl_plain_username"] = username
        if password is not None:
            overrides["sasl_plain_password"] = password.get_secret_value()
        return overrides
