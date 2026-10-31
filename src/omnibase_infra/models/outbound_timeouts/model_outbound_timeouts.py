# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared timeouts for outbound calls (OMN-21623).

The values live in the ``timeouts`` block of ``outbound_timeouts.yaml`` beside
this module. The model has no field defaults, so a block that omits a value
fails to load rather than falling back to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_TIMEOUTS_PATH = Path(__file__).resolve().parent / "outbound_timeouts.yaml"


class ModelOutboundTimeouts(BaseModel):
    """Waits, in seconds, that outbound-call sites apply.

    Attributes:
        git_diff_seconds: Wait for the git subprocess the code-analysis
            enrichment reads a diff from.
        llm_request_seconds: Per-request wait of the code-review and
            test-boilerplate LLM adapters.
        llm_documentation_request_seconds: Per-request wait of the
            documentation-generation LLM adapter.
        llm_coder_max_seconds: Ceiling the HTTP transport accepts for the
            coder-class LLM adapters.
        llm_long_form_max_seconds: Ceiling the HTTP transport accepts for the
            long-form LLM adapters.
        linear_graphql_request_seconds: Wait for a Linear GraphQL call.
        linear_graphql_circuit_reset_seconds: Wait before the Linear GraphQL
            circuit breaker resets.
        linear_ticket_request_seconds: Wait for a Linear ticket-service call.
        close_seconds: Grace an adapter ``close()`` allows.
        audit_poll_seconds: Consumer poll wait of the bus audit.
        audit_topic_discovery_seconds: Wait for the bus audit's topic list.
        audit_offsets_seconds: Wait for one watermark read of the bus audit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    git_diff_seconds: float = Field(gt=0)
    llm_request_seconds: float = Field(gt=0)
    llm_documentation_request_seconds: float = Field(gt=0)
    llm_coder_max_seconds: float = Field(gt=0)
    llm_long_form_max_seconds: float = Field(gt=0)
    linear_graphql_request_seconds: float = Field(gt=0)
    linear_graphql_circuit_reset_seconds: float = Field(gt=0)
    linear_ticket_request_seconds: float = Field(gt=0)
    close_seconds: float = Field(gt=0)
    audit_poll_seconds: float = Field(gt=0)
    audit_topic_discovery_seconds: float = Field(gt=0)
    audit_offsets_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def _request_waits_fit_the_transport_ceiling(self) -> ModelOutboundTimeouts:
        """Refuse a per-request LLM wait above the ceiling its transport accepts."""
        if self.llm_request_seconds > self.llm_long_form_max_seconds:
            raise ValueError("llm_request_seconds exceeds llm_long_form_max_seconds")
        if self.llm_documentation_request_seconds > self.llm_long_form_max_seconds:
            raise ValueError(
                "llm_documentation_request_seconds exceeds llm_long_form_max_seconds"
            )
        return self

    @classmethod
    def from_file(cls, timeouts_path: Path) -> ModelOutboundTimeouts:
        """Load the timeouts from the ``timeouts`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``timeouts`` mapping.
        """
        with timeouts_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("timeouts") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_outbound_timeouts",
            )
            raise ProtocolConfigurationError(
                f"{timeouts_path.name} has no timeouts mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_outbound_timeouts() -> ModelOutboundTimeouts:
    """Return the timeouts declared beside this module."""
    return ModelOutboundTimeouts.from_file(_TIMEOUTS_PATH)


__all__: list[str] = ["ModelOutboundTimeouts", "load_outbound_timeouts"]
