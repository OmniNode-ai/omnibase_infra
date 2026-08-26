# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One strict local Bifrost backend binding from the v2 lane overlay."""

from __future__ import annotations

from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ACTIVE_BACKEND_KEYS = frozenset({"local-coder", "local-heavy-reasoning"})
_SERVED_MODEL_NAME = "qwen3.8"
_PARAMETER_COUNT = "27B"
_CONTEXT_WINDOW = 122_880
_LAB_HOST = "192.168.86.201"
_LAB_PORT = 8000
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class ModelBifrostLaneBackendBinding(BaseModel):
    """One active, unauthenticated local delegation backend binding."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
    )

    backend_key: str = Field(
        alias="backend_id",
        serialization_alias="backend_id",
        min_length=1,
    )
    endpoint_url: str = Field(min_length=1)
    advertised_model: str = Field(
        alias="served_model_id",
        serialization_alias="served_model_id",
        min_length=1,
    )
    parameter_count: str = Field(min_length=1)
    context_window: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    timeout_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_lab_binding(self) -> Self:
        if self.backend_key not in _ACTIVE_BACKEND_KEYS:
            raise ValueError(
                "backend_id must be one of the active local delegation backends "
                f"{sorted(_ACTIVE_BACKEND_KEYS)}, got {self.backend_key!r}"
            )
        if self.advertised_model != _SERVED_MODEL_NAME:
            raise ValueError(
                f"served_model_id must be {_SERVED_MODEL_NAME!r}, "
                f"got {self.advertised_model!r}"
            )
        if self.parameter_count != _PARAMETER_COUNT:
            raise ValueError(
                f"parameter_count must be {_PARAMETER_COUNT!r}, "
                f"got {self.parameter_count!r}"
            )
        if self.context_window != _CONTEXT_WINDOW:
            raise ValueError(
                f"context_window must be {_CONTEXT_WINDOW}, got {self.context_window}"
            )
        if self.max_tokens > self.context_window:
            raise ValueError("max_tokens must not exceed context_window")

        parsed = urlsplit(self.endpoint_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint_url host and port must be valid") from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname != _LAB_HOST
            or port != _LAB_PORT
            or parsed.path != _CHAT_COMPLETIONS_PATH
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "endpoint_url must be the authorized lab endpoint "
                f"http://{_LAB_HOST}:{_LAB_PORT}{_CHAT_COMPLETIONS_PATH}; "
                "userinfo, query, and fragment are forbidden"
            )
        return self


__all__ = ["ModelBifrostLaneBackendBinding"]
