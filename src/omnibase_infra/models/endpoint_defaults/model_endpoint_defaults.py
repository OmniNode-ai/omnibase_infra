# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declared endpoint defaults for infra services (OMN-21690).

The values live in the ``settings`` block of ``endpoint_defaults.yaml``. The model
has no field defaults, so a block that omits a value fails to load rather than
falling back to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_SETTINGS_PATH = Path(__file__).resolve().parent / "endpoint_defaults.yaml"


class ModelEndpointDefaults(BaseModel):
    """Addresses, hosts and ports that infra services use when the environment names none.

    Attributes:
        loopback_address: Address a health or metrics server binds by default.
        bind_all_address: Address a container-facing server binds by default.
        loopback_host_names: Host names that reach the local machine.
        local_host: Host of the Docker-exposed local stack.
        local_kafka_bootstrap_servers: Bootstrap servers of the local broker.
        local_valkey_port: Port the local Valkey publishes.
        runtime_health_port: Port the runtime health server listens on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    loopback_address: str = Field(min_length=1)
    bind_all_address: str = Field(min_length=1)
    loopback_host_names: tuple[str, ...] = Field(min_length=1)
    local_host: str = Field(min_length=1)
    local_kafka_bootstrap_servers: str = Field(min_length=1)
    local_valkey_port: int = Field(ge=1, le=65535)
    runtime_health_port: int = Field(ge=1, le=65535)

    def connect_address(self, bound_host: str) -> str:
        """Return the address a client connects to for a server bound to ``bound_host``.

        A server bound to the bind-all address is not reachable at that address, so a
        local client connects to the loopback address instead; any other host is
        returned unchanged.
        """
        return (
            self.loopback_address if bound_host == self.bind_all_address else bound_host
        )

    def is_loopback(self, host: str) -> bool:
        """Return whether ``host`` is one of the declared loopback host names."""
        return host in self.loopback_host_names

    @classmethod
    def from_file(cls, settings_path: Path) -> ModelEndpointDefaults:
        """Load the defaults from the ``settings`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``settings`` mapping.
        """
        with settings_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("settings") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_endpoint_defaults",
            )
            raise ProtocolConfigurationError(
                f"{settings_path.name} has no settings mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_endpoint_defaults() -> ModelEndpointDefaults:
    """Return the endpoint defaults declared beside this model."""
    return ModelEndpointDefaults.from_file(_SETTINGS_PATH)


__all__: list[str] = ["ModelEndpointDefaults", "load_endpoint_defaults"]
