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
``contract_hotload: {enabled: true, mode: <mode>, allowed_environments: [...]}``,
read from the same source as ``handler_source``. The default is off. The switch
fails closed: it is honoured only where the runtime's environment is named by
the declared ``allowed_environments`` list, and an environment that is not on
the list, or an unset environment, is a boot error. The list is configuration
(the runtime config file or the overlay that renders it); this module holds no
environment names.

Related:
    - OMN-20079: interim gate (plan task H0)
    - OMN-21318: contract hot-loading epic
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

__all__: list[str] = ["ModelContractHotloadConfig"]


class ModelContractHotloadConfig(BaseModel):
    """The ``contract_hotload`` block of the runtime config file.

    Attributes:
        enabled: Master switch for both hot-load paths. Default ``False``.
        mode: Which load policy applies once enabled. Required when ``enabled``
            is true so that turning the paths on is always a deliberate,
            named choice; ignored while the switch is off.
        allowed_environments: The environment names where hot loading is
            permitted. Required (non-empty) when ``enabled`` is true. Names are
            matched case-insensitively after trimming. Anything not named here,
            including an unset environment, is refused.
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

    allowed_environments: tuple[str, ...] = Field(
        default=(),
        description="Environment names where hot loading is permitted (allowlist)",
    )

    @field_validator("allowed_environments", mode="after")
    @classmethod
    def _normalize_allowed_environments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        names = tuple(name.strip().lower() for name in value)
        if any(not name for name in names):
            raise ValueError(
                "contract_hotload.allowed_environments must not contain a blank name"
            )
        return names

    @model_validator(mode="after")
    def _mode_and_allowlist_required_when_enabled(self) -> Self:
        if self.enabled and self.mode is None:
            raise ValueError(
                "contract_hotload.mode is required when contract_hotload.enabled "
                "is true (one of 'on_demand', 'full_load')"
            )
        if self.enabled and not self.allowed_environments:
            raise ValueError(
                "contract_hotload.allowed_environments is required when "
                "contract_hotload.enabled is true (the environments where hot "
                "loading is permitted)"
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
        (``ONEX_ENVIRONMENT`` and the event bus environment). With the switch on,
        every one of them must be set and named by ``allowed_environments``.

        Raises:
            ProtocolConfigurationError: The block is malformed, or the switch is
                on while an environment is unset or is not on the allowlist.
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
        if not gate.enabled:
            return gate
        named = [name.strip().lower() for name in environments]
        if not named or any(not name for name in named):
            raise ProtocolConfigurationError(
                "contract_hotload.enabled is true but the runtime environment is "
                "unset; contract hot loading is refused unless the environment is "
                "named by contract_hotload.allowed_environments (OMN-20079)",
                context=context,
            )
        outside = sorted(
            {name for name in named if name not in gate.allowed_environments}
        )
        if outside:
            raise ProtocolConfigurationError(
                f"contract_hotload.enabled is true in environment "
                f"({', '.join(outside)}) that contract_hotload.allowed_environments "
                "does not name; contract hot loading is refused there (OMN-20079)",
                context=context,
            )
        return gate
