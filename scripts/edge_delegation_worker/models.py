# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed models for the edge delegation worker.

Kept deliberately narrow: this worker consumes/produces a small, fixed
envelope shape and does not attempt to model the full gateway session
lifecycle the control-plane node owns -- only the fields this client needs
to drive attach/heartbeat/renewal and one claim/infer/publish/ack cycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.types import JsonType

EnumWorkerAuthMode = Literal["bearer_token", "client_credentials"]


class ModelWorkerCredential(BaseModel):
    """Credential loaded from the operator-supplied ``--credential-file``.

    Exactly one of the two supported shapes is populated, selected by
    ``auth_mode``:

    - ``bearer_token``: the file holds one pre-issued opaque token used
      verbatim as the ``Authorization: Bearer`` value (the "pre-issued
      tenant key" path -- no token endpoint call).
    - ``client_credentials``: the file holds a JSON object with
      ``client_id`` / ``client_secret`` / ``token_endpoint`` (optionally
      ``scope``) for a Keycloak ``client_credentials`` grant (the ``ga-*``
      gateway-attach client path).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    auth_mode: EnumWorkerAuthMode
    bearer_token: str | None = Field(default=None, repr=False)
    client_id: str | None = None
    client_secret: str | None = Field(default=None, repr=False)
    token_endpoint: str | None = None
    scope: str | None = None

    @model_validator(mode="after")
    def _check_mode_fields(self) -> ModelWorkerCredential:
        if self.auth_mode == "bearer_token":
            if not self.bearer_token:
                raise ValueError(
                    "auth_mode=bearer_token requires a non-empty bearer_token"
                )
        else:
            missing = [
                name
                for name, value in (
                    ("client_id", self.client_id),
                    ("client_secret", self.client_secret),
                    ("token_endpoint", self.token_endpoint),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "auth_mode=client_credentials requires non-empty "
                    f"fields: {', '.join(missing)}"
                )
        return self

    def __repr__(self) -> str:  # pragma: no cover - defensive, exercised via repr()
        # Never let a default dataclass-style repr accidentally interpolate a
        # secret field even though repr=False is already set per-field above.
        return f"ModelWorkerCredential(auth_mode={self.auth_mode!r})"


class ModelGatewayRenewalDirective(BaseModel):
    """Renewal window the attach response declares (OMN-15952 contract)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str
    renew_not_before: datetime
    renew_at: datetime
    session_expires_at: datetime


class ModelGatewaySession(BaseModel):
    """The subset of the attach/heartbeat response this worker acts on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    expires_at: datetime
    heartbeat_interval_seconds: int = Field(gt=0)
    renewal: ModelGatewayRenewalDirective | None = None


class ModelGatewayHeartbeatResult(BaseModel):
    """Result of one heartbeat call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    termination_reason: str | None = None

    @property
    def is_terminated(self) -> bool:
        return self.termination_reason is not None


class ModelDelegationEnvelope(BaseModel):
    """One claimed unit of work from the local mirrored bus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    source_topic: str
    event_type: str
    payload: dict[str, JsonType]
    headers: dict[str, str] = Field(default_factory=dict)


class ModelLocalInferenceRequest(BaseModel):
    """Request built from a claimed envelope, sent to the local model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    model: str
    messages: tuple[dict[str, JsonType], ...] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ModelLocalInferenceResult(BaseModel):
    """Parsed, typed result of one local inference call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    content: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
