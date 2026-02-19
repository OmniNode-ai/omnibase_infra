# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for AdapterDocumentationGeneration against a live LLM.

These tests call the real DeepSeek-R1 endpoint (LLM_DEEPSEEK_R1_URL,
default http://192.168.86.200:8101) and are intentionally skipped when the
endpoint is unreachable so they never block CI.

Run locally with:
    uv run pytest tests/integration/adapters/llm/test_adapter_documentation_generation_integration.py -v -s

Mark:
    integration  -- auto-applied by tests/integration/conftest.py
    slow         -- each call takes ~10-60 s depending on model load
"""

from __future__ import annotations

import os
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.adapters.llm.adapter_documentation_generation import (
    TASK_TYPE_API_DOC,
    TASK_TYPE_DOCSTRING,
    TASK_TYPE_README,
    AdapterDocumentationGeneration,
)
from omnibase_spi.contracts.delegation.contract_delegated_response import (
    ContractDelegatedResponse,
)

# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------

_ENDPOINT_URL: str = os.environ.get("LLM_DEEPSEEK_R1_URL", "http://192.168.86.200:8101")

# ---------------------------------------------------------------------------
# Session-scoped reachability check -- skip entire module if endpoint is down
# ---------------------------------------------------------------------------


def _endpoint_reachable(url: str) -> bool:
    """Return True if the LLM endpoint responds to a health probe."""
    try:
        resp = httpx.get(f"{url}/health", timeout=5.0)
        return resp.status_code < 500
    except Exception:
        return False


# Evaluated once at collection time; tests are deselected if endpoint is down.
_ENDPOINT_AVAILABLE: bool = _endpoint_reachable(_ENDPOINT_URL)
_SKIP_REASON: str = (
    f"DeepSeek-R1 endpoint unreachable at {_ENDPOINT_URL} "
    "(set LLM_DEEPSEEK_R1_URL to override)"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ENDPOINT_AVAILABLE, reason=_SKIP_REASON),
]

# ---------------------------------------------------------------------------
# Source fixtures -- real snippets kept short to limit latency
# ---------------------------------------------------------------------------

_SIMPLE_FUNCTION = """\
def compute_retry_delay(attempt: int, base_ms: int = 100, max_ms: int = 30_000) -> int:
    delay = min(base_ms * (2 ** attempt), max_ms)
    return delay
"""

_CLASS_SOURCE = """\
class CircuitBreaker:
    def __init__(self, threshold: int, reset_timeout: float) -> None:
        self._threshold = threshold
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._open = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._open = True

    def is_open(self) -> bool:
        return self._open
"""


# ---------------------------------------------------------------------------
# Shared adapter fixture
# ---------------------------------------------------------------------------

# The LLM transport enforces fail-closed HMAC signing (LOCAL_LLM_SHARED_SECRET).
# The vLLM-compatible server ignores the header, but the client won't send the
# request without a non-empty secret.  If the secret is already in the
# environment (from .env), use it; otherwise inject a test-only value.
_SECRET_ENV: str = "LOCAL_LLM_SHARED_SECRET"
_SECRET_INJECTED: bool = _SECRET_ENV not in os.environ


@pytest.fixture(scope="module", autouse=True)
def _inject_hmac_secret() -> None:  # type: ignore[return]
    """Ensure LOCAL_LLM_SHARED_SECRET is set for the duration of the module."""
    if _SECRET_INJECTED:
        os.environ[_SECRET_ENV] = "test-integration-secret"
    yield  # type: ignore[misc]
    if _SECRET_INJECTED:
        os.environ.pop(_SECRET_ENV, None)


@pytest.fixture
def adapter() -> AdapterDocumentationGeneration:
    """Fresh adapter per test so the circuit breaker starts closed each time.

    Uses 256 max_tokens instead of the default 2048 to keep integration test
    latency manageable — we're verifying the adapter plumbing, not output quality.
    """
    return AdapterDocumentationGeneration(
        base_url=_ENDPOINT_URL,
        max_tokens=256,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_valid_response(result: ContractDelegatedResponse, task_type: str) -> None:
    """Assert structural correctness of a ContractDelegatedResponse."""
    assert isinstance(result, ContractDelegatedResponse)

    # rendered_text must be non-empty and not the fallback error message
    assert result.rendered_text, "rendered_text is empty"
    assert "Documentation Unavailable" not in result.rendered_text, (
        f"Adapter returned fallback error message for task_type={task_type!r}: "
        f"{result.rendered_text[:200]}"
    )

    # Attribution fields
    assert result.attribution.model_name, "attribution.model_name is empty"
    assert result.attribution.endpoint_url == _ENDPOINT_URL
    assert result.attribution.latency_ms > 0
    assert result.attribution.prompt_version.startswith("v")
    assert 0.0 <= result.attribution.delegation_confidence <= 1.0

    # structured_json carries task_type through
    assert result.structured_json == {"task_type": task_type}


# ---------------------------------------------------------------------------
# Tests: each task_type
# ---------------------------------------------------------------------------


class TestDocstringGeneration:
    """Live integration tests for task_type='docstring'."""

    @pytest.mark.asyncio
    async def test_generates_docstring_for_simple_function(
        self,
        adapter: AdapterDocumentationGeneration,
    ) -> None:
        """Docstring task returns non-empty text with attribution."""
        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING,
            source=_SIMPLE_FUNCTION,
        )
        _assert_valid_response(result, TASK_TYPE_DOCSTRING)
        # A docstring should mention the function name or parameters
        assert any(
            kw in result.rendered_text
            for kw in ("attempt", "delay", "Args", "Returns", "int")
        ), f"Docstring content looks unrelated: {result.rendered_text[:300]}"

    @pytest.mark.asyncio
    async def test_docstring_with_correlation_id(
        self,
        adapter: AdapterDocumentationGeneration,
    ) -> None:
        """correlation_id is accepted without error; attribution is still populated."""
        correlation_id = uuid4()
        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING,
            source=_SIMPLE_FUNCTION,
            correlation_id=correlation_id,
        )
        _assert_valid_response(result, TASK_TYPE_DOCSTRING)
        # correlation_id is not embedded in attribution (documented trade-off),
        # but the call must succeed without error.
        assert result.attribution.endpoint_url == _ENDPOINT_URL


class TestReadmeGeneration:
    """Live integration tests for task_type='readme'."""

    @pytest.mark.asyncio
    async def test_generates_readme_section_for_class(
        self,
        adapter: AdapterDocumentationGeneration,
    ) -> None:
        """README task returns content referencing the source class.

        Note: DeepSeek-R1 is a chain-of-thought model -- it emits reasoning text
        before the final answer.  With a 256-token budget the reasoning may not
        complete, so we validate content relevance rather than Markdown structure.
        """
        result = await adapter.generate(
            task_type=TASK_TYPE_README,
            source=_CLASS_SOURCE,
        )
        _assert_valid_response(result, TASK_TYPE_README)
        # The model should at least mention the class name or key concepts
        lower = result.rendered_text.lower()
        assert any(kw in lower for kw in ("circuit", "failure", "threshold", "open")), (
            f"README output does not mention CircuitBreaker concepts: "
            f"{result.rendered_text[:300]}"
        )


class TestApiDocGeneration:
    """Live integration tests for task_type='api_doc'."""

    @pytest.mark.asyncio
    async def test_generates_api_doc_with_parameters_section(
        self,
        adapter: AdapterDocumentationGeneration,
    ) -> None:
        """API doc task returns content referencing parameters."""
        result = await adapter.generate(
            task_type=TASK_TYPE_API_DOC,
            source=_SIMPLE_FUNCTION,
        )
        _assert_valid_response(result, TASK_TYPE_API_DOC)
        # API doc should mention Parameters or Arguments
        assert any(
            kw in result.rendered_text
            for kw in ("Parameters", "Arguments", "Args", "attempt", "base_ms")
        ), f"API doc content looks unrelated: {result.rendered_text[:300]}"


# ---------------------------------------------------------------------------
# Tests: adapter-level behaviour (not content)
# ---------------------------------------------------------------------------


class TestAdapterBehaviour:
    """Structural and behavioural integration tests."""

    @pytest.mark.asyncio
    async def test_attribution_latency_is_plausible(
        self,
        adapter: AdapterDocumentationGeneration,
    ) -> None:
        """Latency reported in attribution is > 0 ms and < 300 s."""
        result = await adapter.generate(
            task_type=TASK_TYPE_DOCSTRING,
            source=_SIMPLE_FUNCTION,
        )
        assert 0 < result.attribution.latency_ms < 300_000, (
            f"Latency out of expected range: {result.attribution.latency_ms:.1f} ms"
        )

    @pytest.mark.asyncio
    async def test_all_task_types_succeed(
        self,
        adapter: AdapterDocumentationGeneration,
    ) -> None:
        """All three task_types produce valid ContractDelegatedResponse."""
        for task_type in (TASK_TYPE_DOCSTRING, TASK_TYPE_README, TASK_TYPE_API_DOC):
            result = await adapter.generate(
                task_type=task_type,
                source=_SIMPLE_FUNCTION,
            )
            _assert_valid_response(result, task_type)
