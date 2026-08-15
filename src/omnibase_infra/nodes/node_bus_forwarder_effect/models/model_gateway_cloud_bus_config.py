# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cloud bus config model for the gateway forwarder."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelGatewayCloudBusConfig(BaseModel):
    """Provider-neutral cloud Kafka leg config resolved from contract refs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_provider_id: UUID
    cloud_broker_ref: str = Field(..., min_length=1)
    cloud_auth_ref: str = Field(..., min_length=1)
    acl_provisioner_ref: str = Field(..., min_length=1)
    client_id_ref: str | None = Field(default=None, min_length=1)
    client_secret_api_key_ref: str | None = Field(default=None, min_length=1)
    msk_region_ref: str | None = Field(default=None, min_length=1)
    security_protocol: Literal["SASL_SSL"] = "SASL_SSL"
    sasl_mechanism: Literal["OAUTHBEARER", "AWS_MSK_IAM"] = "OAUTHBEARER"

    @field_validator(
        "cloud_broker_ref",
        "cloud_auth_ref",
        "acl_provisioner_ref",
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

    @field_validator(
        "client_id_ref",
        "client_secret_api_key_ref",
        "msk_region_ref",
    )
    @classmethod
    def _validate_optional_contract_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._validate_contract_ref(value)

    @model_validator(mode="after")
    def _validate_auth_refs(self) -> ModelGatewayCloudBusConfig:
        if self.sasl_mechanism == "OAUTHBEARER":
            if self.client_id_ref is None or self.client_secret_api_key_ref is None:
                raise ValueError(
                    "OAUTHBEARER gateway cloud bus requires client_id_ref and "
                    "client_secret_api_key_ref"
                )
            if self.msk_region_ref is not None:
                raise ValueError(
                    "OAUTHBEARER gateway cloud bus must not declare msk_region_ref"
                )
        else:
            if self.msk_region_ref is None:
                raise ValueError(
                    "AWS_MSK_IAM gateway cloud bus requires msk_region_ref"
                )
            if (
                self.client_id_ref is not None
                or self.client_secret_api_key_ref is not None
            ):
                raise ValueError(
                    "AWS_MSK_IAM gateway cloud bus uses the AWS credential chain, "
                    "not OAuth client refs"
                )
        return self
