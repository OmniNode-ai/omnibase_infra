# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The facts of the stack a runtime proof boots, declared in the plan node's contract.

Ticket: OMN-19572
"""

from __future__ import annotations

import re
from ipaddress import IPv6Address

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.lab_proof.model_lab_proof_step_timeouts import (
    MAX_STEP_SECONDS,
    ModelLabProofStepTimeouts,
)


class ModelLabProofBundlePolicy(BaseModel):
    """The laptop bundle's fixed names today; slot-derived once OMN-19569 lands.

    Every field is required: a contract missing one fails at load naming it,
    never at the first proof that needed it (CLAUDE.md rule 8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle: str = Field(min_length=1)
    clone_base_url: str = Field(pattern=r"^https://[A-Za-z0-9.-]+$")
    infra_repo: str = Field(pattern=r"^OmniNode-ai/[A-Za-z0-9_.-]+$")
    compose_project: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    runtime_image: str = Field(min_length=1)
    runtime_policy_env: str = Field(
        pattern=r"^[A-Za-z0-9_./-]+$",
        description="The runtime policy env file, relative to the infra clone. The "
        "catalog CLI loads it after the bundle env; a bare compose build needs it "
        "named, or the ${VAR:?} guards it satisfies refuse the render.",
    )
    container_runtime_main: str = Field(min_length=1)
    container_runtime_effects: str = Field(min_length=1)
    container_migration_gate: str = Field(min_length=1)
    host_ports: tuple[int, ...] = Field(min_length=1)
    health_interval_seconds: int = Field(ge=1)
    health_deadline_seconds: int = Field(ge=1)
    wiring_failure_patterns: tuple[str, ...] = Field(min_length=1)
    wiring_failure_extract: str = Field(min_length=1)
    consumer_modules: tuple[str, ...] = Field(min_length=1)
    delegation_prompt: str = Field(min_length=1)
    delegation_timeout_seconds: int = Field(ge=1, le=MAX_STEP_SECONDS)
    bus_bootstrap_servers: str = Field(
        min_length=1,
        description="Comma-separated host:port bus addresses the effects container's "
        "delegation step reaches. IPv6 hosts must be bracketed.",
    )
    step_timeouts: ModelLabProofStepTimeouts
    uv_image: str = Field(min_length=1)
    override_dockerfile: str = Field(min_length=1)

    @field_validator("bus_bootstrap_servers")
    @classmethod
    def validate_bus_bootstrap_servers(cls, value: str) -> str:
        """Reject malformed brokers before the first delegation needs the bus."""
        for broker in value.split(","):
            host, separator, port = broker.rpartition(":")
            if (
                not separator
                or re.fullmatch(r"[0-9]{1,5}", port) is None
                or not 1 <= int(port) <= 65535
            ):
                raise ValueError("each bus broker must have a port in 1..65535")
            if host.startswith("[") and host.endswith("]"):
                IPv6Address(host[1:-1])
            elif (
                len(host) > 253
                or re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
                    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*\.?",
                    host,
                )
                is None
            ):
                raise ValueError(
                    "each bus broker must have a valid hostname or IPv6 address"
                )
        return value


__all__ = ["ModelLabProofBundlePolicy"]
