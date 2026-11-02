# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared settings of the registration orchestrator.

The registration orchestrator's package is not an EFFECT node, so none of its
modules read the process environment. The two operator switches it honours are
declared in ``node_registration_orchestrator/contract.yaml`` under ``descriptor``
with the ``${env.VAR:default}`` overlay convention and resolved here through
``expand_contract_env_refs``, the overlay package's one sanctioned env-reading
boundary. Dispatchers and handlers take a ``ModelRegistrationSettings`` by
injection; only a caller that injects nothing falls back to the contract.

Related Tickets:
    - OMN-21698: Registration orchestrator clock and settings injection
    - OMN-3444: Auto-ACK switch (local and dev use only)
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

ENV_LIVENESS_INTERVAL_SECONDS: Final[str] = "ONEX_LIVENESS_INTERVAL_SECONDS"
DEFAULT_LIVENESS_INTERVAL_SECONDS: Final[int] = 60

_CONTRACT_PACKAGE: Final[str] = "omnibase_infra.nodes.node_registration_orchestrator"
_CONTRACT_NAME: Final[str] = "contract.yaml"


@cache
def _declared_descriptor() -> tuple[str, str]:
    """Return the contract's raw ``(auto_ack, liveness_interval_seconds)`` declarations.

    The contract is read as package data, so a zipped or relocated install
    resolves it too. Only the raw declarations are cached; ``from_contract``
    expands their environment references on every call.
    """
    with (
        files(_CONTRACT_PACKAGE)
        .joinpath(_CONTRACT_NAME)
        .open(encoding="utf-8") as contract_file
    ):
        raw = yaml.safe_load(contract_file)
    descriptor = raw.get("descriptor") if isinstance(raw, dict) else None
    auto_ack = descriptor.get("registration_auto_ack") if descriptor else None
    interval = descriptor.get("liveness_interval_seconds") if descriptor else None
    if not isinstance(auto_ack, str) or not isinstance(interval, str):
        raise ValueError(
            f"contract {_CONTRACT_PACKAGE}/{_CONTRACT_NAME} must declare string descriptor.registration_auto_ack "
            "and descriptor.liveness_interval_seconds"
        )
    return auto_ack, interval


class ModelRegistrationSettings(BaseModel):
    """The registration orchestrator's operator switches.

    Attributes:
        auto_ack: Publish the ack command directly after an accepted
            registration (local and dev use only; see OMN-3444).
        liveness_interval_seconds: Offset of a node's liveness deadline.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    auto_ack: bool = False
    liveness_interval_seconds: int = Field(default=DEFAULT_LIVENESS_INTERVAL_SECONDS)

    @classmethod
    def from_contract(cls) -> ModelRegistrationSettings:
        """Resolve the settings the contract declares, at call time.

        Raises:
            ProtocolConfigurationError: If the declared interval resolves to a
                value that is not an integer.
        """
        from omnibase_infra.runtime.overlay.contract_env_ref import (
            expand_contract_env_refs,
        )

        auto_ack, interval = _declared_descriptor()
        resolved_interval = expand_contract_env_refs(interval)
        try:
            liveness_interval_seconds = int(resolved_interval)
        except ValueError as e:
            raise ProtocolConfigurationError(
                f"Invalid value for {ENV_LIVENESS_INTERVAL_SECONDS}: "
                f"'{resolved_interval}' is not a valid integer",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="get_liveness_interval_seconds",
                    target_name=f"env.{ENV_LIVENESS_INTERVAL_SECONDS}",
                ),
            ) from e
        return cls(
            auto_ack=expand_contract_env_refs(auto_ack).lower() == "true",
            liveness_interval_seconds=liveness_interval_seconds,
        )


__all__ = [
    "DEFAULT_LIVENESS_INTERVAL_SECONDS",
    "ENV_LIVENESS_INTERVAL_SECONDS",
    "ModelRegistrationSettings",
]
