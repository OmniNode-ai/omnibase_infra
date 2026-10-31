# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared network binding defaults for infra services (OMN-21623).

The values live in the ``settings`` block of ``network_binding_settings.yaml``.
The model has no field defaults, so a block that omits a value fails to load
rather than falling back to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_SETTINGS_PATH = Path(__file__).resolve().parent / "network_binding_settings.yaml"


class ModelNetworkBindingSettings(BaseModel):
    """Addresses and ports that infra services use when the environment names none.

    Attributes:
        loopback_address: Address a health or metrics server binds by default.
        bind_all_address: Address a container-facing server binds by default.
        local_host: Host of the Docker-exposed local stack.
        postgres_port: PostgreSQL port used when a caller names none.
        tcp_port_min: Lowest valid TCP port number.
        tcp_port_max: Highest valid TCP port number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    loopback_address: str = Field(min_length=1)
    bind_all_address: str = Field(min_length=1)
    local_host: str = Field(min_length=1)
    postgres_port: int = Field(ge=1, le=65535)
    tcp_port_min: int = Field(ge=1)
    tcp_port_max: int = Field(le=65535)

    @classmethod
    def from_file(cls, settings_path: Path) -> ModelNetworkBindingSettings:
        """Load the settings from the ``settings`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``settings`` mapping.
        """
        with settings_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("settings") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_network_binding_settings",
            )
            raise ProtocolConfigurationError(
                f"{settings_path.name} has no settings mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_network_binding_settings() -> ModelNetworkBindingSettings:
    """Return the network binding settings declared beside this model."""
    return ModelNetworkBindingSettings.from_file(_SETTINGS_PATH)


__all__: list[str] = ["ModelNetworkBindingSettings", "load_network_binding_settings"]
