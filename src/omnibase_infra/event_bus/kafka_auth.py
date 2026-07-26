# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared aiokafka authentication helpers.

Keep direct aiokafka admin/producer/consumer call sites aligned with
``EventBusKafka`` so managed MSK IAM cutover does not leave health checks,
topic provisioning, or lag checks on plaintext-only client construction.
"""

from __future__ import annotations

import asyncio
import ssl

from aiokafka.abc import AbstractTokenProvider

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


class OAuthBearerTokenProvider(AbstractTokenProvider):
    """aiokafka-compatible OAUTHBEARER token provider."""

    def __init__(
        self,
        *,
        token_endpoint_url: str,
        client_id: str,
        client_secret: str,
    ) -> None:
        self._token_endpoint_url = token_endpoint_url
        self._client_id = client_id
        self._client_secret = client_secret

    async def token(self) -> str:
        """Fetch OAuth2 bearer token using client credentials flow."""
        if not self._token_endpoint_url.startswith("https://"):
            raise ValueError("OAuth token endpoint must use https")

        from omnibase_infra.runtime.models.model_http_client_config import (
            ModelHttpClientConfig,
        )
        from omnibase_infra.runtime.providers.provider_http_client import (
            ProviderHttpClient,
        )

        provider = ProviderHttpClient(
            ModelHttpClientConfig(timeout_seconds=30.0, follow_redirects=False)
        )
        client = await provider.create()
        try:
            response = await client.post(
                self._token_endpoint_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            payload = response.json()
            access_token = payload["access_token"]
            if not isinstance(access_token, str):
                raise TypeError("OAuth token response access_token must be a string")
            return access_token
        finally:
            await ProviderHttpClient.close(client)


class MSKTokenProvider(AbstractTokenProvider):
    """aiokafka-compatible token provider for AWS MSK IAM authentication."""

    def __init__(self, region: str) -> None:
        self._region = region

    async def token(self) -> str:
        """Generate a fresh SigV4-backed MSK IAM OAUTHBEARER token."""
        if not self._region.strip():
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="msk_token",
                target_name="kafka_config",
            )
            raise ProtocolConfigurationError(
                "AWS_MSK_IAM requires non-empty msk_region",
                context=context,
                parameter="msk_region",
                value=self._region,
            )
        from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

        loop = asyncio.get_running_loop()
        token, _expiry_ms = await loop.run_in_executor(
            None,
            lambda: MSKAuthTokenProvider.generate_auth_token(self._region),
        )
        return token


def build_aiokafka_auth_kwargs(config: ModelKafkaEventBusConfig) -> dict[str, object]:
    """Build auth/TLS kwargs for aiokafka clients from runtime Kafka config."""
    if config.security_protocol == "PLAINTEXT":
        return {}

    kwargs: dict[str, object] = {"security_protocol": config.security_protocol}

    if config.sasl_mechanism is not None:
        kwargs["sasl_mechanism"] = config.sasl_mechanism

    if config.sasl_mechanism == "OAUTHBEARER":
        kwargs["sasl_oauth_token_provider"] = OAuthBearerTokenProvider(
            token_endpoint_url=str(config.sasl_oauthbearer_token_endpoint_url),
            client_id=str(config.sasl_oauthbearer_client_id),
            client_secret=str(config.sasl_oauthbearer_client_secret),
        )
    elif config.sasl_mechanism == "AWS_MSK_IAM":
        if config.security_protocol != "SASL_SSL":
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="build_aiokafka_auth_kwargs",
                target_name="kafka_config",
            )
            raise ProtocolConfigurationError(
                "AWS_MSK_IAM requires security_protocol='SASL_SSL', "
                f"got {config.security_protocol!r}",
                context=context,
                parameter="security_protocol",
                value=config.security_protocol,
            )
        kwargs["sasl_mechanism"] = "OAUTHBEARER"
        kwargs["sasl_oauth_token_provider"] = MSKTokenProvider(region=config.msk_region)

    if config.ssl_ca_file is not None:
        kwargs["ssl_context"] = ssl.create_default_context(cafile=config.ssl_ca_file)

    return kwargs


def build_aiokafka_auth_kwargs_from_env() -> dict[str, object]:
    """Build auth/TLS kwargs from the standard runtime Kafka env variables."""
    return build_aiokafka_auth_kwargs(ModelKafkaEventBusConfig.default())
