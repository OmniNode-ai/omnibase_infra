# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One strict local Bifrost backend binding from the v2 lane overlay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class AuthorizedLabBinding(NamedTuple):
    """The one shape a given local backend is allowed to declare on a lane."""

    host: str
    port: int
    served_model_id: str
    parameter_count: str
    context_window: int


# OMN-15807 established the single-endpoint authority for the .201 lab binding.
# OMN-16833 generalises it to a per-backend table: the fleet serves more than one
# local model, and pinning every backend to one host/port/model made the second
# live local rung (``local-ds-v4-flash``) unrepresentable — so the dev lane shipped
# it as ``endpoint_url: null``, it was dropped by ``_load_bifrost_endpoints``, and
# ``escalation``'s large-window local rung silently degraded to the metered ceiling.
#
# Every row below is a LIVE readback, not a doc claim (2026-08-28, OMN-16833):
#   GET http://192.168.86.201:8000/v1/models -> id "qwen3.8", max_model_len 122880
#   GET http://192.168.86.200:8101/v1/models -> id "deepseek-v4-flash",
#       context_length 131072  (server: ds4-server --ctx 131072 --port 8101 serving
#       DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf)
# Both were additionally confirmed reachable from inside `omninode-runtime` on .201.
# ``parameter_count`` for DS-V4-Flash carries forward the 284B MoE figure the
# omnimarket contract has declared since OMN-12492; the served metadata does not
# expose a parameter count, so this field is a declaration, not a probe result.
_AUTHORIZED_BINDINGS: Mapping[str, AuthorizedLabBinding] = {
    "local-coder": AuthorizedLabBinding(
        host="192.168.86.201",  # onex-allow-internal-ip OMN-16833 reason="authorized .201 lab binding table"
        port=8000,
        served_model_id="qwen3.8",
        parameter_count="27B",
        context_window=122_880,
    ),
    "local-heavy-reasoning": AuthorizedLabBinding(
        host="192.168.86.201",  # onex-allow-internal-ip OMN-16833 reason="authorized .201 lab binding table"
        port=8000,
        served_model_id="qwen3.8",
        parameter_count="27B",
        context_window=122_880,
    ),
    "local-ds-v4-flash": AuthorizedLabBinding(
        host="192.168.86.200",  # onex-allow-internal-ip OMN-16833 reason="authorized .200 lab binding table"
        port=8101,
        served_model_id="deepseek-v4-flash",
        parameter_count="284B",
        context_window=131_072,
    ),
}

ACTIVE_BACKEND_KEYS = frozenset(_AUTHORIZED_BINDINGS)


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
        authorized = _AUTHORIZED_BINDINGS.get(self.backend_key)
        if authorized is None:
            raise ValueError(
                "backend_id must be one of the active local delegation backends "
                f"{sorted(ACTIVE_BACKEND_KEYS)}, got {self.backend_key!r}"
            )
        if self.advertised_model != authorized.served_model_id:
            raise ValueError(
                f"served_model_id for {self.backend_key!r} must be "
                f"{authorized.served_model_id!r}, got {self.advertised_model!r}"
            )
        if self.parameter_count != authorized.parameter_count:
            raise ValueError(
                f"parameter_count for {self.backend_key!r} must be "
                f"{authorized.parameter_count!r}, got {self.parameter_count!r}"
            )
        if self.context_window != authorized.context_window:
            raise ValueError(
                f"context_window for {self.backend_key!r} must be "
                f"{authorized.context_window}, got {self.context_window}"
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
            or parsed.hostname != authorized.host
            or port != authorized.port
            or parsed.path != _CHAT_COMPLETIONS_PATH
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"endpoint_url for {self.backend_key!r} must be the authorized lab "
                f"endpoint http://{authorized.host}:{authorized.port}"
                f"{_CHAT_COMPLETIONS_PATH}; "
                "userinfo, query, and fragment are forbidden"
            )
        return self


__all__ = ["ACTIVE_BACKEND_KEYS", "ModelBifrostLaneBackendBinding"]
