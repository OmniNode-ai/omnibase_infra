# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract hot-load gate configuration (OMN-20079, plan task H0).

The runtime has two paths that cache and materialize a contract received over
the event bus:

* the post-freeze listener on ``onex.evt.platform.node-registration.v1``
  (``RuntimeHostProcess._start_dynamic_contract_listener``), and
* the baseline subscription on ``onex.evt.platform.contract-registered.v1``
  that ``handler_source.mode: kafka_events`` turns on
  (``RuntimeHostProcess._wire_baseline_subscriptions``).

Both are off unless the runtime config file carries
``contract_hotload: {enabled: true, mode: <mode>}``, read from the same source
as ``handler_source``. The default is off, and on in a production environment
is a boot error.

Related:
    - OMN-20079: interim gate (plan task H0)
    - OMN-21318: contract hot-loading epic
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

__all__: list[str] = ["ModelContractHotloadConfig"]

# Environment names for which an enabled switch is a boot error. Matched
# case-insensitively; ``onex-prod`` is the value the production runtime
# configmap sets for ONEX_ENVIRONMENT.
_FORBIDDEN_ENVIRONMENTS: frozenset[str] = frozenset(
    {"prod", "production", "onex-prod", "public", "onex-public"}
)


class ModelContractHotloadConfig(BaseModel):
    """The ``contract_hotload`` block of the runtime config file.

    Attributes:
        enabled: Master switch for both hot-load paths. Default ``False``.
        mode: Which load policy applies once enabled. Required when ``enabled``
            is true so that turning the paths on is always a deliberate,
            named choice; ignored while the switch is off.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Master switch for both contract hot-load paths",
    )
    mode: Literal["on_demand", "full_load"] | None = Field(
        default=None,
        description="Load policy; required when the switch is on",
    )

    @model_validator(mode="after")
    def _mode_required_when_enabled(self) -> Self:
        if self.enabled and self.mode is None:
            raise ValueError(
                "contract_hotload.mode is required when contract_hotload.enabled "
                "is true (one of 'on_demand', 'full_load')"
            )
        return self

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: Mapping[str, object] | None,
        *,
        environments: Iterable[str],
    ) -> Self:
        """Read and enforce the block from the runtime config.

        ``environments`` are the environment names the runtime resolved
        (``ONEX_ENVIRONMENT`` and the event bus environment); the switch on
        under any production name is refused.

        Raises:
            ProtocolConfigurationError: The block is malformed, or the switch is
                on while either environment names a production environment.
        """
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="contract_hotload.gate",
        )
        raw = (runtime_config or {}).get("contract_hotload")
        if raw is None:
            return cls()
        try:
            gate = cls.model_validate(raw)
        except ValidationError as exc:
            raise ProtocolConfigurationError(
                f"invalid contract_hotload configuration: {sanitize_error_message(exc)}",
                context=context,
            ) from exc
        named = {name.strip().lower() for name in environments}
        forbidden = sorted(named & _FORBIDDEN_ENVIRONMENTS)
        if gate.enabled and forbidden:
            raise ProtocolConfigurationError(
                "contract_hotload.enabled is true in a production environment "
                f"({', '.join(forbidden)}); contract hot loading is refused there "
                "(OMN-20079)",
                context=context,
            )
        return gate
