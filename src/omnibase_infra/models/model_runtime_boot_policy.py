# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime boot policy model (OMN-18087)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelRuntimeBootPolicy(BaseModel):
    """How one runtime process bounds, orders and sizes its boot.

    Declared in ``runtime_policy.contract.yaml`` and rendered into the process
    environment, so no join limit, boot bound, join priority or memory limit is
    a literal in a compose file or in the wiring loop.

    Attributes:
        ready_bound_seconds: Seconds from process start within which ``/ready``
            must first answer 200. Past it the kernel logs the blockers by name
            (it never relaxes the readiness gate or drops a required topic).
        max_concurrent_contract_attach: Contract attaches in flight at once
            during the boot interleave.
        priority_contracts: Contract names whose consumers subscribe before any
            other contract's, in the listed order.
        memory_limit: The container memory limit, a docker size such as ``3072M``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    ready_bound_seconds: int = Field(ge=1)
    max_concurrent_contract_attach: int = Field(ge=1)
    priority_contracts: tuple[str, ...] = ()
    memory_limit: str = Field(pattern=r"^[1-9][0-9]*[MG]$")

    @field_validator("priority_contracts")
    @classmethod
    def _priority_contracts_are_unique_names(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(not name.strip() or "," in name for name in value):
            msg = "boot priority contract names must be non-empty and hold no comma"
            raise ValueError(msg)
        if len(set(value)) != len(value):
            msg = "boot priority contract names must be unique"
            raise ValueError(msg)
        return value

    def render_env(self, prefix: str) -> dict[str, str]:
        """Render this policy as the env vars the compose files reference."""
        return {
            f"{prefix}_READY_BOUND_SECONDS": str(self.ready_bound_seconds),
            f"{prefix}_MAX_CONCURRENT_CONTRACT_ATTACH": str(
                self.max_concurrent_contract_attach
            ),
            f"{prefix}_BOOT_PRIORITY_CONTRACTS": ",".join(self.priority_contracts),
            f"{prefix}_MEMORY": self.memory_limit,
        }


__all__: list[str] = ["ModelRuntimeBootPolicy"]
