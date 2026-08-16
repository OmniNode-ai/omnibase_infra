# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outbound-only HTTP call to a local, OpenAI-compatible chat endpoint.

This is the one genuinely new leg the mapping report identified as missing
end to end: nothing in ``omninode_infra`` or ``omnibase_infra`` calls out to
a LAN inference endpoint on behalf of a mirrored delegation request. This
module is that call, deliberately narrow -- it does not attempt to
replicate the full feature surface of ``node_llm_inference_effect``'s
``handler_llm_openai_compatible`` (tool calling, usage normalization,
multi-provider finish-reason mapping); it implements exactly what one
delegation-inference cycle needs: build the request, POST it, parse the
response, fail closed on anything else.

The model base URL is always caller-supplied (``--model-base``). Nothing
here defaults to ``127.0.0.1:8099`` or any other literal -- the operator
names their own local model's address explicitly, every run.
"""

from __future__ import annotations

import httpx

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraProtocolError,
    InfraRequestRejectedError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)
from scripts.edge_delegation_worker.models import (
    ModelLocalInferenceRequest,
    ModelLocalInferenceResult,
)

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def _context(*, operation: str, target_name: str) -> ModelInfraErrorContext:
    return ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.LLM,
        operation=operation,
        target_name=target_name,
    )


def _build_wire_payload(request: ModelLocalInferenceRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": request.model,
        "messages": list(request.messages),
    }
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    return payload


async def run_chat_completion(
    model_base: str,
    request: ModelLocalInferenceRequest,
    *,
    http_client: httpx.AsyncClient,
) -> ModelLocalInferenceResult:
    """POST one chat-completion request to the local model and parse it.

    Fail-closed behavior:

    - Network failure / timeout -> ``InfraUnavailableError``.
    - 4xx (bad request the local server rejected) -> ``InfraRequestRejectedError``.
    - Any other non-200, or a 200 body missing ``choices[0].message.content``
      -> ``InfraProtocolError``. There is no fallback that returns an empty
      or synthesized completion.
    """
    url = model_base.rstrip("/") + _CHAT_COMPLETIONS_PATH
    try:
        response = await http_client.post(url, json=_build_wire_payload(request))
    except httpx.TimeoutException as exc:
        raise InfraUnavailableError(
            "local model request timed out",
            context=_context(operation="run_chat_completion", target_name=url),
        ) from exc
    except httpx.HTTPError as exc:
        raise InfraUnavailableError(
            f"local model request failed: {exc}",
            context=_context(operation="run_chat_completion", target_name=url),
        ) from exc

    if 400 <= response.status_code < 500:
        raise InfraRequestRejectedError(
            f"local model rejected the request: HTTP {response.status_code}: "
            f"{response.text[:500]}",
            context=_context(operation="run_chat_completion", target_name=url),
        )
    if response.status_code != httpx.codes.OK:
        raise InfraProtocolError(
            f"local model returned HTTP {response.status_code}: {response.text[:500]}",
            context=_context(operation="run_chat_completion", target_name=url),
        )

    try:
        body = response.json()
        choice = body["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason") or "unknown"
        if not isinstance(content, str):
            raise TypeError("choices[0].message.content was not a string")
        usage = body.get("usage") or {}
        return ModelLocalInferenceResult(
            correlation_id=request.correlation_id,
            content=content,
            finish_reason=str(finish_reason),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise InfraProtocolError(
            "local model response did not match the expected "
            "OpenAI-compatible chat-completion shape",
            context=_context(operation="run_chat_completion", target_name=url),
        ) from exc
