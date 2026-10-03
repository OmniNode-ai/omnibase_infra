# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bus command model for LLM inference requests."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.utils.util_dlq_credential_redaction import is_credential_field_name


class ModelLlmInferenceCommand(BaseModel):
    """Typed command consumed from the node's contract-declared bus topic."""

    # ``hide_input_in_errors``: a refused literal credential must not be echoed
    # back in the ValidationError text, which reaches logs and dead letters.
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        hide_input_in_errors=True,
    )

    correlation_id: UUID = Field(default_factory=uuid4)
    model: str = Field(..., min_length=1)
    messages: tuple[dict[str, JsonType], ...] = Field(default_factory=tuple)
    prompt: str | None = None
    operation_type: EnumLlmOperationType = EnumLlmOperationType.CHAT_COMPLETION
    endpoint_url: str | None = None
    base_url: str | None = None
    provider_config: dict[str, JsonType] = Field(default_factory=dict)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: tuple[str, ...] = Field(default_factory=tuple)
    # OMN-17106 (DR-12): a reference, never a value. This model is consumed
    # off a Kafka topic, so a credential value on it is on the wire (and in the
    # dead-letter topic) for the topic's full retention -- ``SecretStr`` only
    # hid it from ``repr``. The effect boundary resolves the reference
    # immediately before the outbound call (HandlerLlmInferenceCommand).
    api_key_ref: str | None = Field(default=None, min_length=1)
    extra_headers: dict[str, str] = Field(default_factory=dict, repr=False)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)
    gpu_type: str | None = Field(default=None, min_length=1, max_length=64)
    gpu_count: int | None = Field(default=None, ge=1, le=32767)
    compute_usage_source: str | None = None

    @field_validator("provider_config")
    @classmethod
    def _provider_config_carries_no_credential_value(
        cls, value: dict[str, JsonType]
    ) -> dict[str, JsonType]:
        """Refuse a credential-named ``provider_config`` entry (OMN-17106).

        ``provider_config`` is free-form, so no field type can keep a literal
        key out of it; the name is what can be checked. The offending key name
        is reported, never its value. Use ``api_key_ref``.
        """
        offending = sorted(key for key in value if is_credential_field_name(key))
        if offending:
            raise ValueError(
                f"provider_config must not carry a credential value "
                f"(offending keys: {', '.join(offending)}); "
                "use api_key_ref to name a secret resolved at the effect boundary"
            )
        return value

    def provider_value(self, key: str) -> JsonType | None:
        """Return a provider_config value using a validated command key."""
        return self.provider_config.get(key)


__all__: list[str] = ["ModelLlmInferenceCommand"]
