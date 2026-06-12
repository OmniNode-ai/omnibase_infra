# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cloud bus config model for the gateway forwarder."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelGatewayCloudBusConfig(BaseModel):
    """Provider-neutral cloud Kafka leg config resolved from contract refs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_provider_id: UUID
    cloud_broker_ref: str = Field(..., min_length=1)
    cloud_auth_ref: str = Field(..., min_length=1)
    acl_provisioner_ref: str = Field(..., min_length=1)
    client_id_ref: str = Field(..., min_length=1)
    client_secret_api_key_ref: str = Field(..., min_length=1)
    security_protocol: Literal["SASL_SSL"] = "SASL_SSL"
    sasl_mechanism: Literal["OAUTHBEARER"] = "OAUTHBEARER"

    @field_validator(
        "cloud_broker_ref",
        "cloud_auth_ref",
        "acl_provisioner_ref",
        "client_id_ref",
        "client_secret_api_key_ref",
    )
    @classmethod
    def _validate_contract_ref(cls, value: str) -> str:
        ref = value.strip()
        if not ref:
            raise ValueError("gateway cloud bus refs must not be empty")
        if ref.startswith("KAFKA_"):
            raise ValueError(
                "gateway cloud bus config must use contract refs, not KAFKA_* env"
            )
        return ref
