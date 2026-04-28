# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed Kafka configuration for the deploy-agent control bus."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDeployAgentKafkaConfig(BaseModel):
    """Single source of truth for deploy-agent consume and publish bus config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_servers: str = Field(..., min_length=1)
    security_protocol: Literal["PLAINTEXT", "SASL_SSL"] = "PLAINTEXT"
    sasl_mechanism: Literal["PLAIN"] | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None

    @model_validator(mode="after")
    def validate_sasl_pair(self) -> ModelDeployAgentKafkaConfig:
        if bool(self.sasl_username) != bool(self.sasl_password):
            raise ValueError(
                "KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD must be set together"
            )
        if self.security_protocol == "SASL_SSL":
            if not self.sasl_username or not self.sasl_password:
                raise ValueError(
                    "SASL_SSL deploy-agent Kafka config requires "
                    "KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD"
                )
            if self.sasl_mechanism is None:
                object.__setattr__(self, "sasl_mechanism", "PLAIN")
        if self.sasl_username and self.security_protocol != "SASL_SSL":
            raise ValueError(
                "KAFKA_SECURITY_PROTOCOL must be SASL_SSL when SASL credentials are set"
            )
        return self

    def consumer_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "bootstrap_servers": self.bootstrap_servers,
            "security_protocol": self.security_protocol,
        }
        if self.security_protocol == "SASL_SSL":
            kwargs.update(
                {
                    "sasl_mechanism": self.sasl_mechanism or "PLAIN",
                    "sasl_plain_username": self.sasl_username,
                    "sasl_plain_password": self.sasl_password,
                }
            )
        return kwargs

    def producer_kwargs(self) -> dict[str, Any]:
        return self.consumer_kwargs()


def load_deploy_agent_kafka_config_from_env() -> ModelDeployAgentKafkaConfig:
    """Load required deploy-agent Kafka config from environment.

    OMN-9713: there is intentionally no localhost fallback. A missing
    ``KAFKA_BOOTSTRAP_SERVERS`` must fail startup rather than silently consuming
    a different bus from the trigger publisher.
    """
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap_servers:
        raise RuntimeError(
            "KAFKA_BOOTSTRAP_SERVERS is required for deploy-agent; "
            "there is no localhost fallback"
        )

    username = os.environ.get("KAFKA_SASL_USERNAME") or None
    password = os.environ.get("KAFKA_SASL_PASSWORD") or None
    default_protocol = "SASL_SSL" if username or password else "PLAINTEXT"
    security_protocol = os.environ.get("KAFKA_SECURITY_PROTOCOL", default_protocol)

    return ModelDeployAgentKafkaConfig(
        bootstrap_servers=bootstrap_servers,
        security_protocol=security_protocol,
        sasl_mechanism=os.environ.get("KAFKA_SASL_MECHANISM") or None,
        sasl_username=username,
        sasl_password=password,
    )
