# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12664 regression: bifrost endpoint-path preservation.

The cloud-tier Gemini path produced a 404 because the bifrost gateway fed the
backend URL through ``base_url``, so the inference handler appended
``/v1/chat/completions`` again:

- bare-origin base  → ``https://generativelanguage.googleapis.com/v1/chat/completions``
- full-path base    → ``https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions``

The registered endpoint is
``https://generativelanguage.googleapis.com/v1beta/openai/chat/completions``.

The fix: when a backend's configured ``base_url`` is already a complete chat
endpoint, route it through ``endpoint_url`` so ``_build_url`` posts it verbatim.
Non-endpoint base URLs (local OpenAI-compatible servers, GLM origins) stay on
the legacy ``base_url`` path and keep getting the OpenAI path appended exactly
once. These tests fail on the old behavior and pass after the fix.

Related:
    - OMN-12664: cloud-tier Gemini URL path truncation / double-append 404
    - OMN-10489: endpoint_url field — post-as-is for contract-complete endpoints
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.enums import EnumLlmOperationType
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.handler_bifrost_gateway import (
    HandlerBifrostGateway,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
    ModelBifrostConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_request import (
    ModelBifrostRequest,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)

pytestmark = pytest.mark.unit

_CORR = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_TENANT = UUID("12345678-1234-5678-1234-567812345678")
_GEMINI_FULL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_BARE_ORIGIN = "https://generativelanguage.googleapis.com"
_LOCAL_BASE = "http://100.109.203.94:8000"


def _gateway(backend: ModelBifrostBackendConfig) -> HandlerBifrostGateway:
    """Build a gateway whose single backend is the one under test."""
    config = ModelBifrostConfig(
        backends={backend.backend_id: backend},
        default_backends=(backend.backend_id,),
    )
    return HandlerBifrostGateway(
        config=config,
        inference_handler=HandlerLlmOpenaiCompatible(transport=None),
    )


def _chat_request() -> ModelBifrostRequest:
    return ModelBifrostRequest(
        operation_type=EnumLlmOperationType.CHAT_COMPLETION,
        tenant_id=_TENANT,
        messages=[{"role": "user", "content": "hi"}],
        correlation_id=_CORR,
    )


def _build_request_for(base_url: str):  # type: ignore[no-untyped-def]
    backend = ModelBifrostBackendConfig(backend_id="b", base_url=base_url)
    gateway = _gateway(backend)
    return gateway._build_inference_request(
        request=_chat_request(),
        backend_cfg=backend,
        correlation_id=_CORR,
    )


@pytest.mark.unit
def test_full_gemini_endpoint_routes_through_endpoint_url() -> None:
    """A full /v1beta/openai/chat/completions base routes via endpoint_url."""
    req = _build_request_for(_GEMINI_FULL)
    assert req.endpoint_url == _GEMINI_FULL
    # _build_url must post it verbatim — no second path append.
    assert HandlerLlmOpenaiCompatible._build_url(req) == _GEMINI_FULL


@pytest.mark.unit
def test_no_double_append_for_full_chat_endpoint() -> None:
    """Regression: the double-versioned 404 path must NOT be produced."""
    req = _build_request_for(_GEMINI_FULL)
    resolved = HandlerLlmOpenaiCompatible._build_url(req)
    assert resolved != (
        "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
    )
    assert "/v1beta/openai/v1/chat/completions" not in resolved


@pytest.mark.unit
def test_gemini_openai_base_is_not_a_full_endpoint() -> None:
    """A /v1beta/openai base (no /chat/completions) is NOT a full endpoint.

    It stays on the legacy base_url path, where /v1/chat/completions is appended,
    producing the SECOND 404 variant
    (``…/v1beta/openai/v1/chat/completions``). This documents why the Gemini
    default must be the complete registered endpoint, not merely the
    /v1beta/openai base.
    """
    req = _build_request_for(_GEMINI_OPENAI_BASE)
    # Not a complete chat endpoint → stays on legacy base_url path.
    assert req.endpoint_url is None
    resolved = HandlerLlmOpenaiCompatible._build_url(req)
    # The legacy append does NOT reach the registered endpoint for this base.
    assert resolved == (
        "https://generativelanguage.googleapis.com/v1beta/openai/v1/chat/completions"
    )
    assert resolved != _GEMINI_FULL


@pytest.mark.unit
def test_gemini_bare_origin_does_not_truncate_path() -> None:
    """Bare origin appends /v1/chat/completions exactly once (legacy path)."""
    req = _build_request_for(_GEMINI_BARE_ORIGIN)
    assert req.endpoint_url is None
    resolved = HandlerLlmOpenaiCompatible._build_url(req)
    assert resolved == ("https://generativelanguage.googleapis.com/v1/chat/completions")


@pytest.mark.unit
def test_local_openai_backend_unaffected() -> None:
    """Non-Gemini local OpenAI-compatible backend keeps legacy base_url path."""
    req = _build_request_for(_LOCAL_BASE)
    assert req.endpoint_url is None
    assert req.base_url == _LOCAL_BASE
    resolved = HandlerLlmOpenaiCompatible._build_url(req)
    assert resolved == "http://100.109.203.94:8000/v1/chat/completions"


@pytest.mark.unit
def test_local_full_endpoint_routes_through_endpoint_url() -> None:
    """A local backend already declaring a full endpoint posts as-is too.

    The detection is structural (URL shape), not provider-specific, so a local
    server configured with a complete endpoint is handled identically — proving
    no provider sniff was introduced.
    """
    full_local = "http://100.109.203.94:8000/v1/chat/completions"
    req = _build_request_for(full_local)
    assert req.endpoint_url == full_local
    assert HandlerLlmOpenaiCompatible._build_url(req) == full_local


@pytest.mark.unit
def test_trailing_slash_full_endpoint_detected() -> None:
    """A full endpoint with a trailing slash is still detected as complete."""
    req = _build_request_for(_GEMINI_FULL + "/")
    assert req.endpoint_url == _GEMINI_FULL + "/"
    assert HandlerLlmOpenaiCompatible._build_url(req) == _GEMINI_FULL + "/"
