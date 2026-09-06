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
    """The one shape a given local backend is allowed to declare on a lane.

    ``serving`` is the liveness half, added by OMN-16999. Before it, this table
    could express only "authorized" — so a backend whose endpoint had stopped
    answering had exactly two representations, both wrong: leave it bound (the
    lane advertises a rung that returns ``curl`` exit 7, burning an attempt on
    every escalation) or delete the row (which shrinks the declared set, forces
    every lane overlay to drop the rung, and loses the binding the operator has
    to restore by hand when the endpoint comes back). Declaring the rung and
    marking it not-serving is the third state that was missing.
    """

    host: str
    port: int
    served_model_id: str
    parameter_count: str
    context_window: int
    #: Whether this endpoint is currently answering. ``False`` renders the
    #: backend with ``endpoint_url: null`` — the shape ``_load_bifrost_endpoints``
    #: skips — so routing never offers it, while ``routing_rules`` and
    #: ``default_backends`` that name the backend stay internally consistent.
    serving: bool = True


# OMN-15807 established the single-endpoint authority for the .201 lab binding.
# OMN-16833 generalised it to a per-backend table. OMN-16999 corrects the .201
# served id back to live truth and adds the ``serving`` axis.
#
# Every row below is a LIVE readback, not a doc claim. Re-probed 2026-09-05
# (OMN-16999), each with its paired control so a zero is distinguishable from a
# probe that never ran (CLAUDE.md rule 16):
#
#   GET http://192.168.86.201:8000/v1/models
#     -> http=200 in 0.017s; served ids exactly {"Qwen3.6-35B-A3B"};
#        owned_by "vllm"; root /data/inference/hf-cache/Qwen3.6-35B-A3B-GPTQ-Int4;
#        max_model_len 131072.
#     This is the POSITIVE CONTROL for the probe method as well as the binding.
#
#   GET http://192.168.86.200:8101/v1/models
#     -> curl exit 7, http=000, "Failed to connect ... after 11 ms".
#     Controls proving the host is up and the probe ran: ping .200 = 2/2 packets,
#     0.0% loss, 0.189ms avg; TCP 22 OPEN; TCP 8101 and 8000 CLOSED. An `ssh
#     jonah@192.168.86.200 hostname` returns "Stickybeatz-Studio.local" — .200 is
#     the workstation itself, and `ps aux | grep -i "ds4|llama|vllm|sglang"`
#     returns no rows there. So the ds4 model server is simply NOT RUNNING; the
#     host is healthy and nothing was restarted (read-only scope).
#
# WHY ``qwen3.8`` WAS WRONG AND WHY THIS KEEPS HAPPENING. OMN-16419 recorded a
# 2026-08-23 readback of "qwen3.8" (SGLang, max_model_len 122880) and pinned that
# id here and across eleven omnimarket YAML sites. The endpoint has since been
# redeployed SGLang -> vLLM and serves "Qwen3.6-35B-A3B" at 131072 — the value
# OMN-16419 replaced. That is the SECOND flip of this one field, and the drift is
# not detectable by reading the repo: every static test passed while every local
# delegation attempt was refused by the OMN-16419 fail-closed attribution guard
# ("configured model_name='qwen3.8' is not in the served ids ['Qwen3.6-35B-A3B']")
# and climbed to the metered ceiling. The durable fix is not this edit — it is
# ``tests/unit/runtime/test_bifrost_served_model_probe_fixture.py``, which pins
# every row below to a recorded ``/v1/models`` probe so a third flip fails a test
# instead of silently burning the local tier.
#
# ``parameter_count`` for DS-V4-Flash carries forward the 284B MoE figure the
# omnimarket contract has declared since OMN-12492; the served metadata does not
# expose a parameter count, so this field is a declaration, not a probe result.
_AUTHORIZED_BINDINGS: Mapping[str, AuthorizedLabBinding] = {
    "local-coder": AuthorizedLabBinding(
        host="192.168.86.201",  # onex-allow-internal-ip OMN-16999 reason="authorized .201 lab binding table"
        port=8000,
        served_model_id="Qwen3.6-35B-A3B",
        parameter_count="27B",
        context_window=131_072,
        serving=True,
    ),
    "local-heavy-reasoning": AuthorizedLabBinding(
        host="192.168.86.201",  # onex-allow-internal-ip OMN-16999 reason="authorized .201 lab binding table"
        port=8000,
        served_model_id="Qwen3.6-35B-A3B",
        parameter_count="27B",
        context_window=131_072,
        serving=True,
    ),
    # NOT SERVING as of the 2026-09-05 probe above. The row is kept, not deleted:
    # the endpoint is a workstation-hosted service an operator starts on demand,
    # so this is a stopped process, not retired hardware (contrast the .201:8001
    # reasoner slot, which was physically removed and IS deleted upstream). To
    # restore the rung: start the ds4 server, re-probe GET .200:8101/v1/models,
    # update the fixture with that readback, and flip ``serving`` back to True.
    "local-ds-v4-flash": AuthorizedLabBinding(
        host="192.168.86.200",  # onex-allow-internal-ip OMN-16999 reason="authorized .200 lab binding table"
        port=8101,
        served_model_id="deepseek-v4-flash",
        parameter_count="284B",
        context_window=131_072,
        serving=False,
    ),
}

#: Every backend id a lab lane must DECLARE. Membership is authorization, not
#: liveness — a declared rung may be marked not-serving (OMN-16999).
ACTIVE_BACKEND_KEYS = frozenset(_AUTHORIZED_BINDINGS)

#: The subset whose endpoint is currently answering. This is what routing may
#: actually offer; the difference between the two sets is rendered as
#: ``endpoint_url: null``.
SERVING_BACKEND_KEYS = frozenset(
    key for key, binding in _AUTHORIZED_BINDINGS.items() if binding.serving
)


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
    #: OMN-16999. Must equal the authorized row's ``serving`` value, so a lane
    #: cannot unilaterally re-enable a rung the probe table says is dark, nor
    #: silently disable one that is up. Defaults True: the overwhelmingly common
    #: case stays a one-line binding, and only a not-serving rung has to say so.
    serving: bool = True

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
        if self.serving != authorized.serving:
            raise ValueError(
                f"serving for {self.backend_key!r} must be {authorized.serving} "
                f"(the probe table is the authority on liveness, not the lane "
                f"overlay), got {self.serving}"
            )

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


__all__ = [
    "ACTIVE_BACKEND_KEYS",
    "SERVING_BACKEND_KEYS",
    "AuthorizedLabBinding",
    "ModelBifrostLaneBackendBinding",
]
